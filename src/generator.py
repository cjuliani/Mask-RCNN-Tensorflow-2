import os
import config
import json
import time
import cv2
import random
import numpy as np
import pandas as pd

from PIL import Image
from src.modules.utils.augmentation import augment_geometry


class InfCounterLoop:
    def __init__(self):
        self.index_counter = 0

    def generate(self, data):
        while self.index_counter < len(data):
            to_return = self.index_counter
            self.index_counter += 1
            yield to_return
        else:
            self.index_counter = 0
            yield self.index_counter


class DataGenerator(InfCounterLoop):
    def __init__(self, data_folder, input_size, categories, names, partition=(80, 10)):
        super(DataGenerator, self).__init__()

        # Training parameters
        self.obj_id = None
        self.index_in_epoch = 0
        self.img_id = 0
        self.img_file = None
        self.images_path = os.path.join(data_folder, "images")
        self.partition = partition
        self.input_size = input_size
        tmp = np.argsort(names).tolist()  # indices to sort class names/categories
        self.class_numbers = [categories[i] for i in tmp]
        self.class_names = [names[i] for i in tmp]

        # Augmentation parameters
        self.translation_rate = 0  # rate of translation in image
        self.direction = 0  # direction of translation
        self.angle = 0  # rotation angle
        self.scaling = 0  # scaling factor of the image

        # Get data
        images_path, _, img_files = os.walk(self.images_path).__next__()

        # Get annotation file and read through it
        anno_file = None
        anno_path, _, anno_files = os.walk(os.path.join(data_folder, "annotations")).__next__()
        for file in anno_files:
            if 'json' in file:
                anno_file = os.path.join(anno_path, file)
                break
        print("opening annotation file")

        with open(anno_file) as f:
            anno_json = json.load(f)

            self.img_dframe = pd.DataFrame.from_dict(anno_json['images'])
            self.anno_dframe = pd.DataFrame.from_dict(anno_json['annotations'])
            f.close()

        # Rebuild annotation dict based on class_names to learn
        # Note: category_id starts from 1, not 0.
        self.anno_dframe = self.anno_dframe[self.anno_dframe['category_id'].isin(self.class_numbers)]

        # Rebuild annotation dict based on class_names to learn
        # Note: category_id starts from 1, not 0.'
        self.train_img_ids, self.val_ids, self.test_ids = [], [], []
        for cat in self.class_numbers:
            tmp_dframe = self.anno_dframe[self.anno_dframe['category_id'].isin([cat])]

            # Get possible image ids
            image_ids = np.unique(tmp_dframe["image_id"]).tolist()
            num_ids = len(image_ids)

            # Randomize data and create partitions
            # np.random.shuffle(self.image_ids)
            train_num = int((num_ids / 100) * self.partition[0])
            val_num = int(np.ceil((num_ids / 100) * self.partition[1]))
            train_ids = image_ids[:train_num]
            val_ids = image_ids[train_num:train_num + val_num]
            test_ids = image_ids[train_num + val_num:]

            self.train_img_ids.append(train_ids)
            self.val_ids.append(val_ids)
            self.test_ids.append(test_ids)

        print("splitting data")
        self.train_img_ids, self.train_cat_ids, self.train_indices = self.set_data_indices(
            img_ids=self.train_img_ids,
            categories=self.class_numbers)
        self.data_size = len(self.train_img_ids)

        self.val_ids, self.val_cat_ids, self.val_indices = self.set_data_indices(
            img_ids=self.val_ids,
            categories=self.class_numbers)

        self.test_ids, self.test_cat_ids, self.test_indices = self.set_data_indices(
            img_ids=self.test_ids,
            categories=self.class_numbers)

    def set_data_indices(self, img_ids, categories):
        # Get maximum number of ids for considered class_names.
        max_num = np.max([len(items) for items in img_ids])

        ids_lists = []
        for ids in img_ids:
            if len(ids) < max_num:
                new_ids = np.random.choice(ids, max_num - len(ids)).tolist()
                new_ids = new_ids + ids
            else:
                new_ids = ids
            ids_lists.append(new_ids)

        new_img_ids = np.concatenate(ids_lists, axis=0)
        num_train = new_img_ids.shape[0]
        img_indices = np.random.choice(range(num_train), num_train, replace=False)
        new_img_ids = new_img_ids[img_indices]

        cat_ids = np.array(categories).reshape((len(categories), 1))
        cat_ids = np.reshape(np.tile(cat_ids, (1, max_num)), (-1))
        cat_ids = cat_ids[img_indices]

        return new_img_ids, cat_ids, img_indices

    def randomize_augment_params(self):
        """Randomize parameters."""
        # Transformation
        self.translation_rate = random.choice(np.arange(*config.TRANSLATION_RATE))
        self.direction = random.choice(config.DIRECTION)
        self.angle = random.choice(np.arange(*config.ANGLE))

        # /!\ clipped_zoom() not stable for a given scale factor.
        # If you change the below parameters, check results visually.
        self.scaling = random.choice(config.SCALING)

    def get_object_mask(self, img_ref, num_objects, obj_data_in_image):
        global_obj_mask_ref = np.zeros((img_ref.size[1], img_ref.size[0]), np.int32)
        for i in range(num_objects):
            id = i + 1
            pos = np.array(obj_data_in_image["segmentation"].iloc[i][0])
            pos = pos.reshape((-1, 2))
            # cols, rows = pos[:, 0], pos[:, 1]
            cv2.fillPoly(global_obj_mask_ref, pts=[pos], color=(id, id, id))  # create object polygon in mask

        # global_sem_mask = (global_obj_mask > 0).astype(np.int32)
        return cv2.resize(
            global_obj_mask_ref,
            dsize=self.input_size,
            interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def get_box_from_obj_mask(mask, id):
        rows, cols = np.where(mask == id)
        x1 = np.min(cols)
        y1 = np.min(rows)
        x2 = np.max(cols)
        y2 = np.max(rows)
        return [x1, y1, x2, y2], x2-x1, y2-y1, rows, cols

    def next_batch_with_cropping(self, num_obj_thresh, training, data_aug,
                                 min_obj_size_ratio, resize, model_input_size):
        # Define the start:end range of data to flush out for training
        # given the batch size
        if training:
            index = self.index_in_epoch
            self.index_in_epoch += 1
            # Re-initialize and randomize training samples after every epoch
            # and continue flushing out batch data repeatedly.
            if self.index_in_epoch >= len(self.train_img_ids):
                index = 0
                self.index_in_epoch = 1

                np.random.shuffle(self.train_indices)
                self.train_img_ids = self.train_img_ids[self.train_indices]
                self.train_cat_ids = self.train_cat_ids[self.train_indices]

            self.img_id = self.train_img_ids[index]
            self.obj_id = self.train_cat_ids[index]
        else:
            # Shuffle validation indices if index iterator
            # reached its counter limit.
            if self.index_counter == 0:
                np.random.shuffle(self.val_indices)
                self.val_ids = self.val_ids[self.val_indices]
                self.val_cat_ids = self.val_cat_ids[self.val_indices]

            # Generate validation indices continuously.
            index = next(self.generate(self.val_ids))
            self.img_id = self.val_ids[index]
            self.obj_id = self.val_cat_ids[index]

        # Get image
        indexed_data = self.img_dframe.loc[self.img_dframe["id"] == self.img_id]
        self.img_file = indexed_data["file_name"].item()

        # Open image and normalize pixels.
        img_ref = Image.open(os.path.join(self.images_path, self.img_file)).convert("RGB")
        img_array = np.asarray(img_ref, dtype=np.int32) / 255

        # Get object properties for current image.
        obj_data_in_image = self.anno_dframe.loc[self.anno_dframe["image_id"] == self.img_id]
        initial_num_obj_in_img = len(obj_data_in_image)

        # Rule out image if number of objects too high. Recall
        # function recursively.
        params = (num_obj_thresh, training, data_aug, min_obj_size_ratio, resize, model_input_size)
        if (initial_num_obj_in_img < num_obj_thresh[0]) or (initial_num_obj_in_img > num_obj_thresh[1]):
            return self.next_batch_with_cropping(*params)

        # Build object mask
        global_obj_mask_ref = np.zeros((img_array.shape[0], img_array.shape[1]), np.int32)
        for i in range(initial_num_obj_in_img):
            obj_id = i + 1
            pos = np.array(obj_data_in_image["segmentation"].iloc[i][0])
            pos = pos.reshape((-1, 2))
            # cols, rows = pos[:, 0], pos[:, 1]
            color = (obj_id, obj_id, obj_id)
            cv2.fillPoly(global_obj_mask_ref, pts=[pos], color=color)  # create object polygon in mask

        # Select random object from category and define its
        # position (x, y).
        rand_index = np.random.choice(range(initial_num_obj_in_img))
        pos = np.reshape(np.array(obj_data_in_image["segmentation"].iloc[rand_index][0]), (-1, 2))
        posx, posy = pos[0]  # (2,)

        # Crop image and mask within a squared patch
        # centered at selected point.
        right = posx + (self.input_size[0] // 2)
        left = np.maximum(0, posx - (self.input_size[0] // 2))
        bottom = posy + (self.input_size[1] // 2)
        top = np.maximum(0, posy - (self.input_size[1] // 2))
        img_array = img_array[top:bottom, left:right]
        sampled_obj_mask = global_obj_mask_ref[top:bottom, left:right]

        # Add extra margin if cropping is not squared
        # (to deal with image borders).
        if img_array.shape[1] < self.input_size[1]:
            right_margin_to_add = self.input_size[1] - img_array.shape[1]
            img_array = np.pad(img_array, ((0, 0), (0, right_margin_to_add), (0, 0)))
            sampled_obj_mask = np.pad(sampled_obj_mask, ((0, 0), (0, right_margin_to_add)))

        if img_array.shape[0] < self.input_size[0]:
            bottom_margin_to_add = self.input_size[0] - img_array.shape[0]
            img_array = np.pad(img_array, ((0, bottom_margin_to_add), (0, 0), (0, 0)))
            sampled_obj_mask = np.pad(sampled_obj_mask, ((0, bottom_margin_to_add), (0, 0)))

        if resize:
            img_array = Image.fromarray(np.uint8(img_array * 255))
            img_array = np.array(img_array.resize(model_input_size, resample=Image.NEAREST)) / 255
            sampled_obj_mask = Image.fromarray(sampled_obj_mask)
            sampled_obj_mask = np.array(sampled_obj_mask.resize(model_input_size, resample=Image.NEAREST))

        if data_aug is True:
            # Expand mask dimension (needed for transform)
            sampled_obj_mask_aug = np.expand_dims(np.array(sampled_obj_mask), axis=-1)  # (W, H, 1)

            # Reset augmentation parameters (randomization)
            self.randomize_augment_params()

            # Augment image
            selects1 = [random.choice(range(3)), random.choice(range(2))]
            img_aug = augment_geometry(
                array=img_array,
                size=self.input_size[0],
                select=selects1,
                angle=self.angle,
                scaling=self.scaling,
                direction=self.direction,
                translation_rate=self.translation_rate)

            # Augment masks
            sampled_obj_mask_aug = augment_geometry(
                array=sampled_obj_mask_aug,
                size=self.input_size[0],
                select=selects1,
                angle=self.angle,
                scaling=self.scaling,
                direction=self.direction,
                translation_rate=self.translation_rate)
            # Format precision and dimension of masks.
            sampled_obj_mask_aug = np.squeeze(np.round(sampled_obj_mask_aug).astype(np.int32), axis=-1)  # (W, H)
        else:
            # No augmentation
            sampled_obj_mask_aug = sampled_obj_mask
            img_aug = img_array

        # Define object numbers remaining in sampled image.
        remaining_objects = np.unique(sampled_obj_mask_aug)[1:]

        # If augmentation removed all object, reload normal image.
        if not list(remaining_objects):
            sampled_obj_mask_aug = sampled_obj_mask
            img_aug = img_array
            remaining_objects = np.unique(sampled_obj_mask)[1:]

        # Get new boxes after resizing.
        box, to_keep = [], []
        shp = (initial_num_obj_in_img, self.input_size[0], self.input_size[1])
        binary_masks = np.zeros(shp, np.int32)  # memory greedy
        for obj_id in remaining_objects:
            index = obj_id - 1

            # Boxes
            sampled_bbox, w, h, rows, cols = self.get_box_from_obj_mask(sampled_obj_mask_aug, obj_id)

            # Rule out sampled box if related object is too occluded
            # w.r.t. the original box of the object (before sampling patch).
            box.append(sampled_bbox)

            # Masks
            binary_masks[index, rows, cols] = 1
            to_keep.append(index)

        box = np.array(box, np.int32)
        to_keep = np.array(to_keep)

        # Get labels
        labels = np.array(obj_data_in_image["category_id"].to_list()).astype(np.int32)

        # Convert labels given provided class_names to learn from
        for i, cat_id in enumerate(self.class_numbers):
            labels = np.where(labels == cat_id, i+1, labels)  # WARNING: Add 1, labels start from 1 (0 is background)

        labels = labels[to_keep]
        binary_masks = binary_masks[to_keep]

        return np.expand_dims(img_aug, axis=0), binary_masks, box, labels

    def get_object_data(self, img_id):
        # Get image
        indexed_data = self.img_dframe.loc[self.img_dframe["id"] == img_id]
        img_file = indexed_data["file_name"].item()

        img_ref = Image.open(os.path.join(self.images_path, img_file)).convert('RGB')
        img = img_ref.resize(self.input_size, resample=Image.NEAREST)  # resize to input dimension
        img_ref, img = img_ref, np.array(img) / 255   # normalize pixels

        # Get object mask
        obj_data_in_image = self.anno_dframe.loc[self.anno_dframe["image_id"] == img_id]
        num_objects = len(obj_data_in_image)

        # Build object mask
        global_obj_mask = self.get_object_mask(img_ref, num_objects, obj_data_in_image)

        # Get new boxes after resizing
        bboxes, to_keep = [], []
        bin_masks = np.zeros((num_objects, self.input_size[0], self.input_size[1]), np.int32)  # memory greedy
        for i in range(num_objects):
            id = i + 1
            try:
                # Boxes
                rows, cols = np.where(global_obj_mask == id)
                x1 = np.min(cols)
                y1 = np.min(rows)
                x2 = np.max(cols)
                y2 = np.max(rows)
                bboxes.append([x1, y1, x2, y2])
                # Masks
                bin_masks[i, rows, cols] = 1

                to_keep.append(i)
            except ValueError:
                # an object possibly disappeared when resizing
                # so rule it out
                global_obj_mask = np.where(global_obj_mask == id, 0, global_obj_mask)

        bboxes = np.array(bboxes, np.int32)
        to_keep = np.array(to_keep)

        # Get labels
        labels = np.array(obj_data_in_image["category_id"].to_list()).astype(np.int32)
        labels = labels[to_keep]
        bin_masks = bin_masks[to_keep]

        return np.expand_dims(img, axis=0), bin_masks, bboxes, labels

    def generate_test_data(self, input_ids=None):
        # Define the start:end range of data to flush out for training
        # given the batch size
        if input_ids is None:
            input_ids = self.test_ids

        for img_id in input_ids:
            yield self.get_object_data(img_id)

    def generate_random_test_data(self):
        # Select random data from test dataset.
        index = random.sample(range(len(self.test_ids)), k=1)[0]
        img_id = self.test_ids[index]
        return self.get_object_data(img_id)


def visualize_boxes(background, coords):
    import matplotlib.pyplot as plt
    from matplotlib import patches
    # Create figure and axes, and display image.
    fig, ax = plt.subplots()
    ax.imshow(background)

    # Add boxes and object probability scores.
    for box in coords:
        top_left = (box[0], box[1])
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle(
            top_left, width, height,
            linewidth=0.5,
            edgecolor='red',
            linestyle='-',
            facecolor='none')
        ax.add_patch(rect)

    plt.show()


def get_boxes_dimensions(generator, n=50):
    # Check box dimensions
    for i in range(n):
        start = time.time()
        img, masks, bbox, labels = generator.next_batch_with_cropping(
            num_obj_thresh=[2, 50],
            training=True,
            data_aug=True,
            min_obj_size_ratio=0.3,
            resize=True,
            model_input_size=(512, 512))

        # Get box dimensions
        x1, y1, x2, y2 = np.split(bbox, indices_or_sections=4, axis=1)
        w = (x2-x1).astype(np.int32)
        h = (y2-y1).astype(np.int32)
        end = time.time()
        print(f"w: {np.mean(w).astype(np.int32)}, "
              f"h: {np.mean(h).astype(np.int32)} --- "
              f"mw: {np.max(w)}, mh: {np.max(w)} (speed: {end - start}')")


if __name__ == '__main__':
    generator = DataGenerator(
        data_folder=r"data/iSAID-DOTAv1",
        input_size=(512, 512),
        names=['plane', 'ship'],
        categories=[4, 5])

    img, masks, bbox, labels = generator.next_batch_with_cropping(
        num_obj_thresh=[2, 50],
        training=True,
        data_aug=True,
        min_obj_size_ratio=0.3,
        resize=True,
        model_input_size=(512, 512))

    visualize_boxes(
        background=img[0],
        coords=bbox)
