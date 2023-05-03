import tensorflow as tf


def resize_mask(box_coordinates, pooled_mask, output_size, threshold=None, return_object_mask=False):
    """Resize mask to given dimensions.
    
    Args:
        box_coordinates: bounding boxes coordinates.
        pooled_mask: bounding box mask probabilities.
        output_size (tuple or None): size in pixels of object to resize.
        threshold (float): threshold value to define the mask.
        return_object_mask (bool): if True, returns the mask of
            box object with given dimensions.
    """
    # Get dimension of roi proposal in input image.
    x1 = tf.cast(box_coordinates[1], tf.int32)
    y1 = tf.cast(box_coordinates[2], tf.int32)
    x2 = tf.cast(box_coordinates[3], tf.int32)
    y2 = tf.cast(box_coordinates[4], tf.int32)
    box_index = tf.cast(box_coordinates[0], tf.int32)

    width = tf.cond(x2 - x1 < 1, lambda: 1, lambda: x2 - x1)  # to avoid null dimensions
    height = tf.cond(y2 - y1 < 1, lambda: 1, lambda: y2 - y1)

    # Resize roi mask predicted according to proposal
    # image dimensions.
    mask = tf.image.resize(
        images=tf.expand_dims(pooled_mask[box_index], axis=-1),
        size=[height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        antialias=True)
    mask = tf.squeeze(mask, axis=-1)  # (H, W)

    if return_object_mask is True:
        return mask if threshold is None else tf.where(mask >= threshold, 1., 0.)
    else:
        # Padding to obtain dimension of image.
        offset_left, offset_right = x1, output_size - x2
        offset_top, offset_bottom = y1, output_size - y2
        image_mask = tf.pad(
            tensor=mask,
            paddings=[[offset_top, offset_bottom],
                      [offset_left, offset_right]])
        return image_mask if threshold is None else tf.where(image_mask >= threshold, 1., 0.)


def unmold_masks(pooled_masks, boxes, img_size, mask_threshold,
                 return_object_mask=False):
    """Returns masks of bounding boxes.

    Args:
        pooled_masks: bounding box mask probabilities.
        boxes: bounding boxes coordinates.
        img_size (tuple or None): size in pixels of object to resize.
        mask_threshold (float): threshold value to define the mask.
        return_object_mask (bool): if True, returns the mask of
            box object with given dimensions.
    """
    # Define matrix to iterate (box coordinates merged
    # with batch indices
    batch_size = tf.shape(boxes)[0]
    indices = tf.cast(tf.reshape(tf.range(batch_size), (-1, 1)), tf.int32)
    to_iterate = tf.concat([indices, tf.cast(boxes, tf.int32)], axis=-1)   # (B, 5)

    if return_object_mask:
        # Returns masks of individual bounding boxes.
        masks = []
        for row in to_iterate:
            tmp = resize_mask(
                tf.cast(row, tf.float32),
                pooled_masks, None,
                threshold=mask_threshold,
                return_object_mask=True)
            masks.append(tmp)
        return masks
    else:
        # Returns masks in image frame.
        return tf.map_fn(
            fn=lambda x: resize_mask(
                x, pooled_masks, img_size[0],
                threshold=mask_threshold,
                return_object_mask=False),
            elems=tf.cast(to_iterate, tf.float32))  # (B, tile_size, tile_size)
