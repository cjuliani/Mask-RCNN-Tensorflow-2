import tensorflow as tf


@tf.function(input_signature=(
        tf.TensorSpec([None, 4], tf.float32),
        tf.TensorSpec([None, 4], tf.float32)))
def get_iou_matrix(x_boxes, gt_boxes):
    """Returns the IoU matrix calculated from the coordinates
    of some input box proposals (targets) and those of ground
    truth boxes.
    """
    num_y = tf.shape(gt_boxes)[0]  # (gt_num,)
    # Calculate targets areas.
    x_area = tf.multiply(x_boxes[:, 2] - x_boxes[:, 0], x_boxes[:, 3] - x_boxes[:, 1])

    IOUs = tf.TensorArray(tf.float32, size=num_y)
    for i in tf.range(num_y):
        # Get area of ground truth box (x2-x1 * y2-y1).
        y_area = tf.multiply(gt_boxes[i, 2] - gt_boxes[i, 0], gt_boxes[i, 3] - gt_boxes[i, 1])
        # Calculate IOU only for boxes of ground truth and target that match.
        # To get this matching, filter out target boxes outside current ground
        # truth box denoted by <ix> such that if e.g. <iy> is negative, its
        # respective target box is outside the scope (spatially) of the ground
        # truth box.
        overlap_x = tf.minimum(gt_boxes[i, 2], x_boxes[:, 2]) - tf.maximum(gt_boxes[i, 0], x_boxes[:, 0])
        overlap_y = tf.minimum(gt_boxes[i, 3], x_boxes[:, 3]) - tf.maximum(gt_boxes[i, 1], x_boxes[:, 1])
        overlap_x = tf.where(tf.less(overlap_x, 0), 0., overlap_x)
        overlap_y = tf.where(tf.less(overlap_y, 0), 0., overlap_y)
        # Get area of target box and related overlap with ground truth box.
        overlap_area = tf.multiply(overlap_x, overlap_y)
        # Intersect over union calculation.
        IOU_state = tf.cast(overlap_area / (y_area + x_area - overlap_area), tf.float32)
        IOUs = IOUs.write(i, IOU_state)
    return replace_nan(tf.transpose(IOUs.stack()))  # (target_num, gt_num)


@tf.function(input_signature=(
        tf.TensorSpec([None, 4], tf.float32),
        tf.TensorSpec([None, 4], tf.float32)))
def get_iou_vector(x_boxes, y_boxes):
    """Returns the IoU vector calculated from the coordinates
    of some input box proposals (targets) and those of ground
    truth boxes."""
    # Get box areas.
    target_areas = tf.multiply(x_boxes[:, 2] - x_boxes[:, 0], x_boxes[:, 3] - x_boxes[:, 1])
    gt_areas = tf.multiply(y_boxes[:, 2] - y_boxes[:, 0], y_boxes[:, 3] - y_boxes[:, 1])
    # Calculate IOU per box.
    overlap_x = tf.minimum(y_boxes[:, 2], x_boxes[:, 2]) - tf.maximum(y_boxes[:, 0], x_boxes[:, 0])
    overlap_y = tf.minimum(y_boxes[:, 3], x_boxes[:, 3]) - tf.maximum(y_boxes[:, 1], x_boxes[:, 1])
    overlap_x = tf.where(tf.less(overlap_x, 0), 0., overlap_x)
    overlap_y = tf.where(tf.less(overlap_y, 0), 0., overlap_y)
    # Get area of target box and related overlap with
    # ground truth box.
    overlap_area = tf.multiply(overlap_x, overlap_y)
    # Intersect over union calculation.
    IOUs = tf.cast(overlap_area / (gt_areas + target_areas - overlap_area), tf.float32)
    return replace_nan(IOUs)


def replace_nan(array):
    """Returns input array whose NaN entries are replaced
    by 0 values."""
    return tf.where(tf.math.is_nan(array), tf.zeros_like(array), array)
