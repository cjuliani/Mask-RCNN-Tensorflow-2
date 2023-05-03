import tensorflow as tf


def roi_align(feature_maps, boxes, box_indices, pre_pool_size, image_size,
              crop_coef=2):
    """Returns features cropped by bounding boxes. Sampling is done
    by bilinear interpolation with shape [num_boxes, crop_height,
    crop_width, depth].

    Args:
        feature_maps (list): feature maps returned by the backbone
            network.
        boxes (tensor): bounding boxes with y and x normalized by
            (height -1) and (width -1) respectively. The shape
            is [num_boxes, (y1, x1, y2, x2)].
        box_indices (tensor): indices indicating which feature map
            to crop from the backbone model outputs.
        pre_pool_size (int): the output size.
        image_size (int or list or tuple): the size of the image from which bounding
            boxes were estimated.
        crop_coef (int): cropping coefficient.

    """
    with tf.name_scope('roi_align'):
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)
        boxes_width = (x1 - x0)
        boxes_height = (y1 - y0)
        spacing_w = boxes_width / tf.cast(pre_pool_size, dtype=tf.float32)
        spacing_h = boxes_height / tf.cast(pre_pool_size, dtype=tf.float32)
        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.cast(image_size[1] - 1, dtype=tf.float32)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.cast(image_size[0] - 1, dtype=tf.float32)
        nw = spacing_w * tf.cast(pre_pool_size, dtype=tf.float32) / tf.cast(image_size[1] - 1, dtype=tf.float32)
        nh = spacing_h * tf.cast(pre_pool_size, dtype=tf.float32) / tf.cast(image_size[0] - 1, dtype=tf.float32)

        # Won't be back-propagated to rois anyway, but to save time.
        normalized_bbox = tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

        # Constructed by top-left sample point and bottom-right
        # sample point.
        # Note: crop_and_resize is fully differentiable; we can
        # compute derivatives w.r.t. RoI coordinates if pooling
        # uses bilinear interpolation.
        crop_size = tf.constant([pre_pool_size * crop_coef,
                                 pre_pool_size * crop_coef])

        sampled = tf.image.crop_and_resize(
            image=feature_maps,
            boxes=normalized_bbox,
            box_indices=box_indices,
            crop_size=crop_size,
            method='bilinear',
            name="roi_crops")

        # After sampling,sampled points of each bin need to
        # be averaged.
        return sampled


def select_from_last_axis(index, tensor):
    """Returns tensor whose last axis is the given index."""
    return [tensor[..., tf.cast(index, tf.int32)], index]
