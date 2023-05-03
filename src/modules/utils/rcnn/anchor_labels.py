import tensorflow as tf
from src.modules.utils.iou import get_iou_matrix


@tf.function
def get_labels(gt_boxes, anchors, anchor_batch_size, overlaps_pos, overlaps_neg, positive_ratio,
               image_width, image_height):
    """Returns anchors labels.

    Args:
        gt_boxes (tensor): ground truth bounding boxes.
        anchors (tensor): coordinates of the anchor boxes.
        anchor_batch_size (int): batch size of anchors.
        overlaps_pos (float): IoU threshold considered for
            positive bounding boxes.
        overlaps_neg (float): IoU threshold considered for
            negative bounding boxes.
        positive_ratio (float): the ratio of batch made of
            positive bounding boxes.
        image_width (float): the width of image from which
            bounding boxes were estimated.
        image_height (float): the height of image from which
            bounding boxes were estimated.
    """
    # Generate anchor scores as vector (-1, 0, +1).
    (pos_neg_anchors, class_anchor_assignment,
     iou_vector, assigned_gt_bbox_ratio) = generate_labels(
        gt_boxes=gt_boxes,
        anchors=anchors,
        overlaps_pos=overlaps_pos,
        overlaps_neg=overlaps_neg,
        image_width=image_width,
        image_height=image_height)
    # Balance number of positives and negatives given batch size.
    pos_neg_anchors = balance_labels(
        anchor_scores=pos_neg_anchors,
        anchor_batch_size=anchor_batch_size,
        foreground_ratio=positive_ratio,
        iou_vector=iou_vector)
    return (tf.cast(pos_neg_anchors, tf.float32), class_anchor_assignment,
            assigned_gt_bbox_ratio)  # (H * W * 9, 1)


def fill_label(labels_to_add, final_vector_size, indices_of_labels, fill_value=-1):
    """Returns a positive/negative label vector for all anchors
    within the image limits, i.e. assign 1 to positives (i.e.
    overlapping annotated objects), and 0 to negatives (without
    objects, or considered as background), and -1 to those
    undefined (not known to have objects or background).

    Args:
        labels_to_add (tensor): negative and positives labels to
            consider in the vector.
        final_vector_size (int): the size of final vector.
        indices_of_labels (tensor): indices of anchors considered
            to make up the vector.
        fill_value (int): the standard value used when building
            the vector.
    """

    # Build fill and weight vectors. Weight vector serves in
    # assign the new label values at specified indices.
    indices_to_replace = tf.cast(indices_of_labels, tf.int32)
    fill_vector = tf.scatter_nd(
        indices=indices_to_replace,
        updates=labels_to_add,
        shape=final_vector_size)

    weights = tf.scatter_nd(
        indices=indices_to_replace,
        updates=tf.fill((tf.shape(labels_to_add)[0],), 1),
        shape=final_vector_size)
    weights += - 1

    # Assign label values to specified anchor indices
    # given weights.
    new_labels = tf.cast(fill_vector, tf.int32) + weights

    # Replace negative values if specified.
    if fill_value > -1:
        new_labels = tf.where(new_labels < 0, fill_value, new_labels)
    return new_labels


def fill_iou(labels_to_add, final_vector_size, indices_of_labels, fill_value):
    """Return fill and weight vectors. Weight vector serves in
    # assigning the new label values at specified indices.

    Args:
        labels_to_add (tensor): negative and positives labels to
            consider in the vector.
        final_vector_size (int): the size of final vector.
        indices_of_labels (tensor): indices of anchors considered
            to make up the vector.
        fill_value (float): the standard value used when building
            the vector.
    """
    indices_to_replace = tf.cast(indices_of_labels, tf.int32)
    fill_vector = tf.scatter_nd(
        indices=indices_to_replace,
        updates=labels_to_add,
        shape=final_vector_size)  # (B, 9)

    weights = tf.scatter_nd(
        indices=indices_to_replace,
        updates=tf.fill((tf.shape(labels_to_add)[0],), 1.),
        shape=final_vector_size)  # (B, 9)
    weights += - 1.

    # Assign label values to specified anchor indices
    # given weights.
    new_labels = fill_vector + weights

    # Replace negative values if specified
    if fill_value > -1:
        new_labels = tf.where(new_labels < 0, fill_value, new_labels)
    return new_labels


def clip_anchors(anchors, image_width, image_height):
    """Returns anchors not exceeding the input image dimensions."""
    clipping_indices = tf.where(
        (anchors[:, 0] > 0.) &
        (anchors[:, 2] < tf.cast(image_width, tf.float32)) &
        (anchors[:, 1] > 0.) &
        (anchors[:, 3] < tf.cast(image_height, tf.float32)))
    return tf.gather_nd(anchors, clipping_indices), clipping_indices


def generate_labels(gt_boxes, anchors, overlaps_pos, overlaps_neg, image_width, image_height):
    """Returns the anchor labels.
    1: positive (anchor with object annotated), 0: negative
    (background), -1: unknown.

    Args:
        gt_boxes (tensor): ground truth bounding boxes.
        anchors (tensor): coordinates of the anchor boxes.
        overlaps_pos (float): IoU threshold considered for
            positive bounding boxes.
        overlaps_neg (float): IoU threshold considered for
            negative bounding boxes.
        image_width (float): the width of image from which
            bounding boxes were estimated.
        image_height (float): the height of image from which
            bounding boxes were estimated.
    """
    # Filter out anchors exceeding input image dimensions.
    clipped_anchors, anchors_in_img_indices = clip_anchors(anchors, image_width, image_height)

    # Create label vectors.
    clipped_pos_neg = tf.cast(tf.fill((tf.shape(clipped_anchors)[0],), value=-1), tf.float32)  # (target_num, 1)

    # Calculate IOU between ground truth and anchor boxes.
    iou_matrix = get_iou_matrix(clipped_anchors, gt_boxes)  # (target_num, gt_boxes_num)

    # Get indices of the highest IOUs (per anchor).
    best_gt_match_index = tf.cast(tf.reshape(tf.argmax(iou_matrix, axis=1), [-1]), tf.int32)  # (target_num,)
    max_iou_vector = tf.reduce_max(iou_matrix, axis=-1)  # (target_num, 1)

    # Assign label 0 to anchors whose IOU is below threshold.
    clipped_pos_neg = tf.where(tf.less_equal(max_iou_vector, overlaps_neg), 0., clipped_pos_neg)  # (target_num, 1)

    # Get indices of the highest IOUs (per ground truth boxes).
    clipped_pos_neg = tf.where(tf.greater_equal(max_iou_vector, overlaps_pos), 1., clipped_pos_neg)  # (target_num, 1)

    # Get ratio of ground truth boxes matching anchors at
    # specified IoU foreground threshold.
    pos_indices = tf.reshape(tf.where(tf.greater(clipped_pos_neg, 0)), [-1])
    unique = tf.unique_with_counts(tf.gather(best_gt_match_index, pos_indices))
    count = tf.cast(tf.shape(unique.count)[0], tf.float32)
    assigned_gt_bbox_ratio = count / tf.cast(tf.shape(gt_boxes)[0], tf.float32)

    # When no anchor has an IoU overlap score higher than given
    # threshold, then search for the anchor with the highest IoU
    # and assign it a positive objectness score.
    if not tf.reduce_any(tf.equal(clipped_pos_neg, 1.)):
        best_iou = tf.reduce_max(max_iou_vector)
        clipped_pos_neg = tf.where(tf.equal(max_iou_vector, best_iou), 1., clipped_pos_neg)  # (target_num, 1)

    # Create positive/negative label vector for all anchors
    # within image limits, i.e. assign (0, 1) to positive
    # or negative anchors, and -1 to those undefined.
    pos_neg_anchors = fill_label(
        labels_to_add=clipped_pos_neg,
        final_vector_size=tf.cast((tf.shape(anchors)[0],), tf.int32),
        indices_of_labels=anchors_in_img_indices)

    # Create object label vector for all anchors within image
    # limits, i.e. assign argmax indices (0, 1, 2, ...) to
    # selected anchors, given the number of bounding boxes,
    # and -1 to anchors not positive nor negative.
    class_anchor_assignment = fill_label(
        labels_to_add=best_gt_match_index,
        final_vector_size=tf.cast((tf.shape(anchors)[0],), tf.int32),
        indices_of_labels=anchors_in_img_indices,
        fill_value=0)

    iou_vector = fill_iou(
        labels_to_add=max_iou_vector,
        final_vector_size=tf.cast((tf.shape(anchors)[0],), tf.int32),
        indices_of_labels=anchors_in_img_indices,
        fill_value=0.)

    return (pos_neg_anchors, tf.cast(class_anchor_assignment, tf.int32),
            iou_vector, assigned_gt_bbox_ratio)


def balance_labels(anchor_scores, anchor_batch_size, foreground_ratio, iou_vector,
                   randomize_indices=True):
    """Returns a ternary vector label with the number of foreground
    and background labels balanced in batch.

    Args:
        anchor_scores (tensor): scores of anchors
        anchor_batch_size (int): batch size of anchors.
        foreground_ratio (float): the ratio of batch made of
            positive bounding boxes.
        iou_vector (tensor): the IoU vector.
        randomize_indices (bool): if True, randomize indices of
            positive labels.
    """
    # Make up x% of batch foreground items.
    max_fg_num = tf.cast(tf.cast(anchor_batch_size, tf.float32) * foreground_ratio, tf.int64)

    # Select half-batch of positive labels and disable
    # indices of foreground labels exceeding the half-batch.
    fg_ind = tf.reshape(tf.where(anchor_scores == 1), [-1])  # (pos, )

    # Positive indices to keep
    if randomize_indices:
        to_keep = tf.random.shuffle(fg_ind)[:max_fg_num]  # (fg,)
    else:
        fg_iou_ind = tf.argsort(tf.gather(iou_vector, fg_ind), direction='DESCENDING')
        to_keep = tf.gather(fg_ind, tf.reshape(fg_iou_ind, [-1]))[:max_fg_num]

    # Make vector of 0 and 1.
    pos_vector = tf.scatter_nd(
        indices=tf.reshape(to_keep, (-1, 1)),
        updates=tf.fill((tf.shape(to_keep)[0],), 2),
        shape=tf.cast((tf.shape(anchor_scores)[0],), tf.int64))  # (B, 9)

    # Same for background anchors.
    max_bg_num = tf.cast(anchor_batch_size - tf.shape(to_keep)[0], tf.int64)  # (bg,)
    bg_inds = tf.reshape(tf.where(anchor_scores == 0), [-1])  # (neg,)

    # Negative indices to keep.
    to_keep = tf.random.shuffle(bg_inds)[:max_bg_num]  # (bg,)

    # Make vector of -1 and 0.
    neg_vector = tf.scatter_nd(
        indices=tf.reshape(to_keep, (-1, 1)),
        updates=tf.fill((tf.shape(to_keep)[0],), -1),
        shape=tf.cast((tf.shape(anchor_scores)[0],), tf.int64))  # (B, 9)
    neg_vector = -(neg_vector + 1)

    # Return final ternary vector.
    return pos_vector + neg_vector  # (W * H * 9, 1)
