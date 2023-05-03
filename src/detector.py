import os.path
import tensorflow as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.generator import DataGenerator
from src.modules.solver import Solver
from src.modules.utils.rcnn.mask import unmold_masks, resize_mask


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages


class Detector:
    """Class of the learning module detecting objects.

    Available methods allow to train, tests and apply a learning
    model.

    Attributes:
        input_size (tuple): size in pixels of inputs to learning model.
        data_generator (object): a module generating train batches.
        solver (object): training module containing the learning model
            and the iterative process for training and testing the model.
    """

    def __init__(self, input_size, batch_size, training_classes, training_categories,
                 data_folder, num_classes, model_to_restore=None,
                 weight_folder_to_restore=None, run_eagerly=False, restore_lr=False, load_backbone=False,
                 load_rpn=False, load_head=False, load_cpkt=False, load_mask=False, save_path=None):
        """Initialize a data generator, a trainer and its respective
        learning model."""
        self.input_size = input_size

        # Run mode.
        tf.config.run_functions_eagerly(run_eagerly)

        # Initiate batch generator.
        self.data_generator = DataGenerator(
            data_folder=data_folder,
            input_size=input_size,
            names=training_classes,
            categories=training_categories)

        # Define trainer object from train generator.
        self.solver = Solver(
            data_generator=self.data_generator,
            num_classes=num_classes,
            batch_size=batch_size)

        # Load model variables.
        if weight_folder_to_restore:
            self.load_model_variables(
                restore_lr=restore_lr,
                training_classes=training_classes,
                load_backbone=load_backbone,
                load_rpn=load_rpn,
                load_head=load_head,
                load_mask=load_mask,
                load_cpkt=load_cpkt,
                save_path=save_path,
                model_to_restore=model_to_restore,
                weight_folder_to_restore=weight_folder_to_restore)
        else:
            # If no checkpoint restored, assign class names
            # of generator to checkpoint that must be saved.
            self.solver.ckpt.class_names.assign(self.data_generator.class_names)
            self.solver.ckpt.class_values.assign(self.data_generator.class_numbers)

    def train_model(self, reset_summary):
        """Trains the available learning model from trainer
        object.

        Args:
            reset_summary (bool): if True, delete summary files of
                previously trained model, so that new ones are
                generated when training.
        """
        # Train model.
        self.solver.train(
            reset_summary_files=reset_summary)

    def test_model(self, input_ids=None):
        """Tests the available learning model from trainer
        object."""
        self.solver.test(input_ids)

    def load_model_variables(self, restore_lr, training_classes, load_backbone,
                             load_rpn, load_head, load_mask, load_cpkt, save_path,
                             model_to_restore, weight_folder_to_restore):
        """Loads variables of a saved learning model.

        Args:
            restore_lr (bool): if True, restore the learning rate.
                This is relevant if one want to continue training
                a model at a specified rate, notably if a learning
                rate decay was applied.
            training_classes (list): the class names considered in
                training.
            load_backbone (bool): if True, load the parameters of the
                backbone network.
            load_rpn (bool): if True, load the parameters of the
                RPN network.
            load_head (bool): if True, load the parameters of head
                bboxe/class network.
            load_mask (bool): if True, load the parameters of head
                mask network.
            load_cpkt (bool): if True, load the model checkpoint.
            save_path (str): path to folder result where to save model
                parameters.
            weight_folder_to_restore (str): name of the model to restore.
            model_to_restore (str): name of the folder containing
                the name to restore.
        """
        # Load model variables.
        self.solver.learning_model.load_variables(
            directory=os.path.join(
                save_path, model_to_restore, weight_folder_to_restore),
            checkpoint=self.solver.ckpt,
            load_learning_rate=restore_lr,
            class_names=training_classes,
            load_backbone=load_backbone,
            load_rpn=load_rpn,
            load_head=load_head,
            load_mask=load_mask,
            load_checkpoint=load_cpkt)

        # Assign class names of checkpoint to the
        # learning model.
        self.solver.learning_model.class_names.assign(self.solver.ckpt.class_names)
        self.solver.learning_model.class_values.assign(self.solver.ckpt.class_values)

    def reset_model_variables(self):
        """Reset the parameter values of subclassed models. Use
        this method if a reset is required after loading the
        parameters of a previously trained model (e.g. when
        testing the detector methods)."""
        for model in self.solver.learning_model.networks:
            for ix, layer in enumerate(model.layers):
                if hasattr(model.layers[ix], 'kernel_initializer') and hasattr(model.layers[ix], 'bias_initializer'):
                    weight_initializer = model.layers[ix].kernel_initializer
                    bias_initializer = model.layers[ix].bias_initializer

                    try:
                        # Weights and biases.
                        old_weights, old_biases = model.layers[ix].get_weights()
                        model.layers[ix].set_weights([
                            weight_initializer(shape=old_weights.shape),
                            bias_initializer(shape=len(old_biases))])
                    except ValueError:
                        # Only weights.
                        old_weights = model.layers[ix].get_weights()
                        try:
                            # Weights of layers not within a list.
                            model.layers[ix].set_weights([
                                weight_initializer(shape=old_weights.shape)])
                        except AttributeError:
                            model.layers[ix].set_weights([
                                weight_initializer(shape=old_weights[0].shape)])

    def get_weights_of_learning_model(self):
        """Returns a list of weights from the model layers."""
        loaded_weights = []
        for model in self.solver.learning_model.networks:
            for ix, layer in enumerate(model.layers):
                if hasattr(model.layers[ix], 'kernel_initializer') and hasattr(model.layers[ix], 'bias_initializer'):
                    try:
                        # Weights and biases.
                        old_weights, old_biases = model.layers[ix].get_weights()
                        loaded_weights += [old_weights] + [old_biases]
                    except ValueError:
                        # Only weights.
                        old_weights = model.layers[ix].get_weights()
                        if type(old_weights) == list:
                            old_weights = old_weights[0]
                        loaded_weights += [old_weights]
        return loaded_weights

    def predict(self, image, nms_top_n, nms_iou_thresh, show_class_names,
                hide_bg_class, hide_below_threshold=False, min_score=0.,
                rpn_detection=False, add_mask_contour=True,
                rpn_obj_thresh=0., soft_nms_sigma=0., mask_threshold=0.5):
        """Detect objects from inputs.

        Args:
            image: the input image to predict from.
            nms_top_n (int): maximum number of objects to detect in from
                input. This parameter is relevant for non-maximum
                suppression in learning model.
            nms_iou_thresh (float): a threshold value above which two
                prediction boxes are considered duplicates if their
                overlap ratio exceeds the value. This parameter is
                relevant when calculating the intersection over union
                (iou).
            show_class_names (bool): if True, display the class name
                of predicted objects.
            hide_bg_class (bool): if True, background predictions are
                not displayed.
            hide_below_threshold (bool): if True, predictions whose
                class scores are below the specified threshold are
                not displayed.
            min_score (float): minimum class score for predictions.
            rpn_detection (bool): if True, only display object
                predictions from the region proposal network (RPN).
            rpn_obj_thresh (float): a threshold value above which a
                prediction is considered valid.
            add_mask_contour (bool): if True, add contour line to generated
                mask.
            mask_threshold (float): threshold value for masking.
            soft_nms_sigma (float): soft-NMS parameter. Deactivated
                if set to 0. if not 0, NMS reduces the score of other
                overlapping boxes instead of directly causing them to
                be pruned. Consequently, it returns the new scores of
                each input box in the second output.
                Read: https://arxiv.org/abs/1704.04503 for info.
        """
        (boxes, boxes_obj_scores,
         classes, scores, img,
         rpn_bbox, rpn_scores, obj_masks) = predict_from_single_image(
            image=image,
            nms_top_n=nms_top_n,
            nms_iou_thresh=nms_iou_thresh,
            model=self.solver.learning_model,
            model_input_size=self.input_size,
            min_obj_score=rpn_obj_thresh,
            predict_from_rpn=rpn_detection,
            soft_nms_sigma=soft_nms_sigma)

        if rpn_detection:
            # Create figure and axes, and display image.
            fig, ax = plt.subplots()
            ax.imshow(img[0])

            # Add boxes and object probability scores.
            for prob, rpn_coord in zip(rpn_scores, rpn_bbox):

                if hide_below_threshold and (prob < min_score):
                    continue

                if prob < min_score:
                    line = "-"
                    color = "salmon"
                    text_color = "dimgrey"
                else:
                    line = "-"
                    color = "red"
                    text_color = "black"

                top_left = (rpn_coord[0], rpn_coord[1])
                width = rpn_coord[2] - rpn_coord[0]
                height = rpn_coord[3] - rpn_coord[1]
                rect = patches.Rectangle(
                    top_left, width, height,
                    linewidth=0.5,
                    edgecolor=color,
                    linestyle=line,
                    facecolor='none')
                ax.add_patch(rect)

                # Convert probability in percent.
                prob_ = int(prob.numpy() * 100)
                prob_ = f"{prob_}%"

                # Add box annotation.
                ax.annotate(prob_,
                            (rpn_coord[0], rpn_coord[1]),
                            color=text_color, fontsize=7.5,
                            ha='left', va='bottom',
                            bbox=dict(
                                facecolor=color,
                                edgecolor=color,
                                pad=0.0))
            plt.show()

        else:
            # Get object labels (without background).
            class_values = self.solver.learning_model.class_values.numpy().tolist()
            class_values = [0] + class_values

            # Define color mapping (per class).
            cmap = get_cmap(len(class_values))

            # Define class names if provided.
            class_names = self.solver.learning_model.class_names.numpy().tolist()
            class_names = [name.decode() for name in class_names]
            if show_class_names:
                class_names = ["background"] + class_names
                class_names = [name[:6] for name in class_names]

            # Create figure and axes, and display image.
            fig, ax = plt.subplots()
            ax.imshow(img[0])

            # Add boxes, labels and probability scores.
            data = [classes, scores, boxes, boxes_obj_scores, obj_masks]
            for cls, prob, box, obj_score, mask in zip(*data):
                if hide_bg_class and cls == 0:
                    continue

                if hide_below_threshold and (prob < min_score):
                    continue

                # Do not display objects whose confidence
                # score is below a given threshold.
                if prob < min_score:
                    line = "--"
                    color = "lightgray"
                else:
                    line = "-"
                    color = cmap(cls)

                # Define rectangle coordinates.
                x1, y1, x2, y2 = box
                top_left = (x1, y1)
                width = x2 - x1
                height = y2 - y1

                # Converts a mask generated by the neural network
                # to its original shape and relocate it to its
                # position in original image.
                to_slice = tf.concat([[0.], box], axis=0)
                resized_mask = resize_mask(
                    box_coordinates=to_slice,
                    pooled_mask=tf.expand_dims(mask, axis=0),
                    output_size=img.shape[1],
                    return_object_mask=False)  # (image_width, image_height)

                # Add object mask to image mask.
                if add_mask_contour and not (cls == 0):
                    _mask = np.where(resized_mask.numpy() >= mask_threshold, 1., np.nan)
                    ax.imshow(_mask, alpha=0.3, cmap="gray")

                    _mask = np.where(resized_mask.numpy() >= mask_threshold, 1., 0)
                    lines = contour_rect(_mask)
                    for line in lines:
                        ax.plot(line[1], line[0], color='k', alpha=0.4, linewidth=1.)
                    # source: https://stackoverflow.com/questions/40892203/can-matplotlib-contours-match-pixel-edges

                # Create a Rectangle patch and add it to axes.
                rect = patches.Rectangle(
                    top_left, width, height,
                    linewidth=1,
                    edgecolor=color,
                    linestyle=line,
                    facecolor='none')
                ax.add_patch(rect)

                # Add class names to boxes, or class values
                # otherwise.
                if show_class_names:
                    # Convert probability in percent.
                    obj_score_ = "{:0.2f}".format(obj_score)
                    prob_ = int(prob * 100)
                    prob_ = f": {prob_}% ({obj_score_})"

                    # Define label name wih probability.
                    cls_text = class_names[cls] + prob_
                    ax.annotate(str(cls_text),
                                (box[0], box[1]),
                                color="black", fontsize=7.5,
                                ha='left', va='bottom',
                                bbox=dict(
                                    facecolor=color,
                                    edgecolor=color,
                                    pad=0.0))
                else:
                    # Convert probability in percent.
                    obj_score_ = "{:0.2f}".format(obj_score)
                    prob_ = int(prob * 100)
                    prob_ = f" ({prob_}% ({obj_score_})"

                    # Define label value with probability.
                    cls_text = str(class_values[cls]) + prob_
                    ax.annotate(cls_text,
                                (box[0], box[1]),
                                color="black", fontsize=8,
                                ha='left', va='top',
                                bbox=dict(
                                    facecolor=color,
                                    edgecolor=color,
                                    pad=0.0))
            plt.show()


def predict_from_single_image(image, nms_top_n, nms_iou_thresh, model, model_input_size,
                              min_obj_score=0., soft_nms_sigma=0., predict_from_rpn=False,
                              mask_threshold=0.5):
    """Returns box and class predictions for a single input image.

    Args:
        image: the input image to predict from.
        nms_top_n (int): maximum number of objects to detect in from
            input. This parameter is relevant for non-maximum
            suppression in learning model.
        nms_iou_thresh (float): a threshold value above which two
            prediction boxes are considered duplicates if their
            overlap ratio exceeds the value. This parameter is
            relevant when calculating the intersection over union
            (iou).
        model (object): the learning model used to generate
            predictions.
        model_input_size (tuple): size in pixels of inputs to the
            learning model.
        min_obj_score (float): minimum object score below which objects
            are not considered.
        soft_nms_sigma (float): soft-NMS parameter. Deactivated
            if set to 0. if not 0, NMS reduces the score of other
            overlapping boxes instead of directly causing them to
            be pruned. Consequently, it returns the new scores of
            each input box in the second output.
            Read: https://arxiv.org/abs/1704.04503 for info.
        predict_from_rpn (bool): if True, inference mode for prediction
            purpose (not testing or training) is considered. As such,
            the maximum number of objects to detect from input is
            specified by a nms (non-maximum suppression) parameter.
        mask_threshold (float): threshold value for masking.

    Returns:
        Lists of box coordinates and related object and class scores.
    """
    # Expand dimension and normalize pixels.
    image = tf.expand_dims(tf.cast(image, tf.float32) / 255., axis=0)

    # Resize.
    image = tf.image.resize(image, size=model_input_size)

    # Predict from model.
    (cls_pred, cls_prob, bbox_pred, bbox_obj_scores,
     rpn_bbox, rpn_scores, masks_prob) = model.predict(
        inputs=image,
        soft_nms_sigma=soft_nms_sigma,
        nms_top_n=nms_top_n,
        nms_iou_thresh=nms_iou_thresh,
        obj_score_thresh=min_obj_score,
        predict_from_rpn=predict_from_rpn)

    obj_masks = unmold_masks(
        pooled_masks=masks_prob,
        boxes=bbox_pred,
        img_size=model_input_size,
        mask_threshold=mask_threshold,
        return_object_mask=True)

    # Convert to list and save predictions in main list.
    return (bbox_pred.numpy(),
            bbox_obj_scores.numpy(), cls_pred.numpy(),
            cls_prob.numpy(), image, rpn_bbox,
            rpn_scores, obj_masks)


def contour_rect(image):
    """Fast version"""
    lines = []
    pad = np.pad(image, [(1, 1), (1, 1)])  # zero padding

    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]

    im0 = np.diff(im0, n=1, axis=1)
    starts = np.argwhere(im0 == 1)
    ends = np.argwhere(im0 == -1)

    lines += [([s[0] - .5, s[0] - .5], [s[1] + .5, k[1] + .5]) for s, k
              in zip(starts, ends)]

    im1 = np.diff(im1, n=1, axis=0).T
    starts = np.argwhere(im1 == 1)
    ends = np.argwhere(im1 == -1)
    lines += [([s[1] + .5, k[1] + .5], [s[0] - .5, s[0] - .5]) for s, k
              in zip(starts, ends)]

    return lines


def get_cmap(n, name='rainbow'):
    """Returns a function that maps each index in 0, 1, ..., n-1
    to a distinct RGB color; the keyword argument name must be a
    standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)
