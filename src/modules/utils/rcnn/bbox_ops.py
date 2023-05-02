import tensorflow as tf


def get_head_proposals_from_targets(anchors, targets, image_width, image_height):
    """Returns bounding box proposals after coordinate
    transform and clipping for the HEAD network.

    Args:
        anchors (tensor): coordinates of anchor boxes.
        targets (tensor): formatted coordinates of target boxes to
            learn from.
        image_width (float): width of the image from which
            bounding boxes were estimated.
        image_height (float): height of the image from which
            bounding boxes were estimated.
    """
    proposals = head_targets_to_bbox_transform(
        anchors=tf.cast(anchors, tf.float32),
        targets=targets)  # (W * H * 9, 4)

    return clip_head_boxes(
        boxes=proposals,
        image_width=image_width,
        image_height=image_height)  # (W * H * 9, 4)


def get_rpn_proposals_from_targets(anchors, targets, image_width, image_height):
    """Returns bounding box proposals after coordinate
    transform and clipping for the RPN network.

    Args:
        anchors (tensor): coordinates of anchor boxes.
        targets (tensor): formatted coordinates of target boxes to
            learn from.
        image_width (float): width of the image from which
            bounding boxes were estimated.
        image_height (float): height of the image from which
            bounding boxes were estimated.
    """
    proposals = rpn_targets_to_bbox_transform(
        anchors=tf.cast(anchors, tf.float32),
        targets=targets)  # (W * H * 9, 4)

    return clip_rpn_boxes(
        boxes=proposals,
        image_width=image_width,
        image_height=image_height)  # (W * H * 9, 4)


def rpn_targets_to_bbox_transform(anchors, targets):
    """Returns the formatted coordinates of target bounding
    boxes from the RPN network, based on the dimensions of
    anchor boxes."""
    batch_size = tf.shape(targets)[0]

    # Get anchors center positions and dimensions.
    anchor_w = tf.subtract(anchors[:, 2], anchors[:, 0]) + 1.  # width = (x2 - x1)
    anchor_h = tf.subtract(anchors[:, 3], anchors[:, 1]) + 1.  # height = (y2 - y1)
    anchor_cent_x = anchors[:, 0] + (0.5 * anchor_w)
    anchor_cent_y = anchors[:, 1] + (0.5 * anchor_h)

    anchor_cent_x = tf.tile(tf.reshape(anchor_cent_x, (1, -1)), [batch_size, 1])
    anchor_cent_y = tf.tile(tf.reshape(anchor_cent_y, (1, -1)), [batch_size, 1])
    anchor_w = tf.tile(tf.reshape(anchor_w, (1, -1)), [batch_size, 1])
    anchor_h = tf.tile(tf.reshape(anchor_h, (1, -1)), [batch_size, 1])

    # Get predicted box center coordinates by regression,
    # given the dimensions of anchors, i.e. we calculate
    # an offset from know anchor centers.
    boxes_cent_x = (targets[:, :, 0] * anchor_w) + anchor_cent_x  # ax + b
    boxes_cent_y = (targets[:, :, 1] * anchor_h) + anchor_cent_y

    # exp(x) is a coefficient to rescale the anchors dimensions.
    boxes_w = (tf.math.exp(targets[:, :, 2]) * anchor_w) - 1.
    boxes_h = (tf.math.exp(targets[:, :, 3]) * anchor_h) - 1.

    # Get box corners from center coordinates.
    coord_x1 = boxes_cent_x - (boxes_w * 0.5)  # top-left, (B, obs_n)
    coord_y1 = boxes_cent_y - (boxes_h * 0.5)
    coord_x2 = boxes_cent_x + (boxes_w * 0.5)  # bottom-right
    coord_y2 = boxes_cent_y + (boxes_h * 0.5)
    return tf.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=-1)  # (B, obs_n, 4)


def head_targets_to_bbox_transform(anchors, targets):
    """Returns the formatted coordinates of target bounding
    boxes from the HEAD network, based on the dimensions of
    anchor boxes."""
    # Get anchors center positions and dimensions.
    anchor_w = tf.subtract(anchors[:, 2], anchors[:, 0]) + 1.  # width = (x2 - x1)
    anchor_h = tf.subtract(anchors[:, 3], anchors[:, 1]) + 1.  # height = (y2 - y1)
    anchor_cent_x = anchors[:, 0] + (0.5 * anchor_w)
    anchor_cent_y = anchors[:, 1] + (0.5 * anchor_h)

    # Get predicted box center coordinates by regression,
    # given the dimensions of anchors, i.e. we calculate
    # an offset from know anchor centers.
    boxes_cent_x = anchor_cent_x + (targets[:, 0] * anchor_w)  # ax + b
    boxes_cent_y = anchor_cent_y + (targets[:, 1] * anchor_h)

    # exp(x) is a coefficient to rescale the anchors dimensions.
    boxes_w = (tf.math.exp(targets[:, 2]) * anchor_w) - 1.
    boxes_h = (tf.math.exp(targets[:, 3]) * anchor_h) - 1.

    # Get box corners from center coordinates.
    coord_x1 = boxes_cent_x - (boxes_w * 0.5)  # top-left
    coord_y1 = boxes_cent_y - (boxes_h * 0.5)
    coord_x2 = boxes_cent_x + (boxes_w * 0.5)  # bottom-right
    coord_y2 = boxes_cent_y + (boxes_h * 0.5)
    return tf.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)


def clip_rpn_boxes(boxes, image_width, image_height):
    """Returns coordinates of bounding boxes clipped. Use
    this method to make sure that boxes in image input do
    not exceed the image dimensions. This method processes
    box proposals from the RPN network.
    """
    b0 = tf.maximum(tf.minimum(boxes[:, :, 0], image_width - 1), 0.0)
    b1 = tf.maximum(tf.minimum(boxes[:, :, 1], image_height - 1), 0.0)
    b2 = tf.maximum(tf.minimum(boxes[:, :, 2], image_width - 1), 0.0)
    b3 = tf.maximum(tf.minimum(boxes[:, :, 3], image_height - 1), 0.0)
    return tf.stack([b0, b1, b2, b3], axis=-1)  # (B, obs_n, 4)


def clip_head_boxes(boxes, image_width, image_height):
    """Returns coordinates of bounding boxes clipped. Use
    this method to make sure that boxes in image input do
    not exceed the image dimensions.This method processes
    box proposals from the HEAD network.
    """
    b0 = tf.maximum(tf.minimum(boxes[:, 0], image_width - 1), 0.0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], image_height - 1), 0.0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], image_width - 1), 0.0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], image_height - 1), 0.0)
    return tf.stack([b0, b1, b2, b3], axis=1)
