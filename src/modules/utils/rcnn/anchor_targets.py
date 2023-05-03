import tensorflow as tf
from src.modules.utils.iou import get_iou_matrix


@tf.function
def build_targets(roi_bbox, roi_scores, gt_bbox, gt_cls, gt_masks, foreground_rate, foreground_thresh,
                  background_thresh, normalize_targets, norm_mean_bbox, norm_std_bbox,
                  batch_size, num_classes, resample_positives):
    """Returns target box proposals with formatted coordinates,
     and related classes, IoU, and object scores, including the
     positive and negative indices.

    Args:
        roi_bbox (tensor): the regions of interest bounding boxes.
        roi_scores (tensor): the regions of interest scores.
        gt_bbox (tensor): ground truth bounding boxes.
        gt_cls (tensor): ground truth classes.
        gt_masks (tensor): ground truth masks.
        foreground_rate (float): foreground rate or ratio of
            batch considered for positive boxes (i.e. which consist
            of objects).
        foreground_thresh (float): foreground threshold, or the
            IoU threshold between ground truth and region of interest
            boxes above which the estimated boxes are considered
            positives (i.e. with an object).
        background_thresh (float): background threshold below
            which the estimated boxes are considered negatives (i.e.
            do not contain any objects).
        normalize_targets (bool): if True, normalize the coordinates
            bounding box targets.
        norm_mean_bbox (tensor): the mean coordinate values considered
            for normalization.
        norm_std_bbox (tensor): the standard deviation of coordinate
            values considered for normalization.
        batch_size (float): the batch size.
        num_classes (int): the number of classes.
        resample_positives (bool): if True, make sure the number of
            positive sample reach the limit defined by the foreground
            rate in the batch. This is done by resampling known
            positives.
    """
    # Get number of foreground rois per batch.
    fg_rois_per_image = tf.math.round(foreground_rate * tf.cast(batch_size, tf.float32))  # 0.25 * 256 = 64

    # Determine background and foreground rois.
    (roi_classes, roi_masks, roi_bbox_, roi_scores_,
     bbox_gt_batch, bg_obj_indices, keep_ind,
     assigned_gt_bbox_ratio, iou_batch) = rois_sampling(
        roi_bbox=roi_bbox,
        roi_scores=roi_scores,
        gt_bbox=gt_bbox,
        gt_cls=gt_cls,
        gt_masks=gt_masks,
        foreground_num=tf.cast(fg_rois_per_image, tf.int32),
        foreground_thresh=foreground_thresh,
        background_thresh=background_thresh,
        batch_size=batch_size,
        resample_positives=resample_positives)

    # Compute targets and box weights.
    reg_targets = define_regression_targets(
        rois=roi_bbox_,
        gt_boxes=bbox_gt_batch,
        norm_mean=norm_mean_bbox,
        norm_std=norm_std_bbox,
        normalize=normalize_targets)

    bbox_targets, bbox_weights_ = get_bbox_regression_targets(
        regression_targets=reg_targets,
        labels=tf.cast(roi_classes, tf.float32),
        num_classes=num_classes)

    return (roi_bbox_, roi_classes, roi_scores_, roi_masks,
            bbox_targets, bbox_weights_, bg_obj_indices, keep_ind,
            assigned_gt_bbox_ratio, iou_batch)


def expand_by_random_sampling(ref_number, indices):
    """Returns a vector indices resampled up to a given reference
    number."""
    min_size = tf.minimum(ref_number, tf.size(indices))
    to_add = tf.maximum(ref_number - min_size, 0)

    if to_add > 0:
        # Add background indices if not enough.
        rand_indices = tf.random.uniform(
            minval=0,
            maxval=tf.shape(indices)[0],
            shape=(to_add, 1),
            dtype=tf.dtypes.int32)
        rand_data = tf.gather_nd(indices, rand_indices)  # (to_add,)
        return tf.concat([indices, rand_data], axis=0)  # (ref_number,)
    else:
        # Select random indices.
        return tf.gather(tf.random.shuffle(indices), tf.range(ref_number))  # (ref_number,)


def rois_sampling(roi_bbox, roi_scores, gt_bbox, gt_cls, gt_masks, foreground_num,
                  foreground_thresh, background_thresh, batch_size,
                  resample_positives=False):
    """Returns RoIs distinguished as foreground and background, as determined
    by the IoU thresholds.

    Args:
        roi_bbox (tensor): the regions of interest bounding boxes.
        roi_scores (tensor): the regions of interest scores.
        gt_bbox (tensor): ground truth bounding boxes.
        gt_cls (tensor): ground truth classes.
        gt_masks (tensor): ground truth binary masks of input
            images.
        foreground_num (int): the number of foreground indices.
        foreground_thresh (float): foreground threshold, or the IoU
            threshold between ground truth and region of interest
            boxes above which the estimated boxes are considered
            positives (i.e. with an object).
        background_thresh (float): background threshold below which
             the estimated boxes are considered negatives (i.e. do
             not contain any objects).
        batch_size (float): the batch size.
        resample_positives (bool): if True, make sure the number of
            positive sample reach the limit defined by the foreground
            rate in the batch. This is done by resampling known
            positives.
    """
    # Get matching of ground truth and predicted boxes.
    iou_matrix = get_iou_matrix(roi_bbox, gt_bbox)  # (nms, gt_boxes_n)
    gt_assignment = tf.argmax(iou_matrix, axis=1)  # (nms, )

    # Determine background/foreground labels per anchor.
    max_overlaps = tf.reduce_max(iou_matrix, axis=1)  # (nms, )

    # Build label vector (starts from class 1).
    # obj_labels = tf.gather(gt_cls, gt_assignment)   # (nms, )
    obj_labels = tf.gather(gt_cls, gt_assignment)  # (nms, )

    # Assign class 0 (background, not object) where IoU is
    # below given background thresholding.
    obj_labels = tf.where(tf.less_equal(max_overlaps, background_thresh), 0, obj_labels)

    # Find foreground and background indices.
    foreground_indices = tf.where(tf.greater_equal(max_overlaps, foreground_thresh))  # (nms, 1)
    background_indices = tf.where(
        tf.less_equal(max_overlaps, background_thresh))  # (nms, 1)

    # Build batch data from indices.
    if tf.greater(tf.size(foreground_indices), 0) and tf.greater(tf.size(background_indices), 0):
        # Limit the number of foregrounds and select respective
        # indices randomly.
        fg_num_ = tf.minimum(foreground_num, tf.size(foreground_indices))
        foreground_indices = tf.gather(tf.random.shuffle(foreground_indices), tf.range(fg_num_))  # (fg, 1)

        # Resample positive instances to balance data (w.r.t. negatives).
        if resample_positives and tf.less(tf.size(foreground_indices), foreground_num):
            fg_num_ = foreground_num
            foreground_indices = expand_by_random_sampling(
                ref_number=tf.cast(foreground_num, tf.int32),
                indices=foreground_indices)

        # Same for background. The rest of the batch is background.
        bg_rois_per_image = batch_size - fg_num_  # 256 - 64
        background_indices = expand_by_random_sampling(
            ref_number=bg_rois_per_image,
            indices=background_indices)  # (bg, 1)

    elif tf.greater(tf.size(foreground_indices), 0):
        # If no background, only include foregrounds.
        # Same for background. The rest of the batch is background.
        foreground_indices = expand_by_random_sampling(
            ref_number=batch_size,
            indices=foreground_indices)  # (B, 1)
        fg_num_ = batch_size

    elif tf.greater(tf.size(background_indices), 0):
        # If no foreground, only include backgrounds.
        # tf.print("WARNING: No foreground indices, only backgrounds.")
        background_indices = expand_by_random_sampling(
            ref_number=batch_size,
            indices=background_indices)  # (B, 1)
        fg_num_ = 0
    else:
        background_indices = tf.zeros((batch_size, 1), dtype=tf.int64)  # (B, 1)
        fg_num_ = 0

    # Get labels of foreground / background.
    keep_ind = tf.reshape(tf.concat([foreground_indices, background_indices], axis=0), [-1])  # (B,)
    pos_keep_ind = tf.reshape(foreground_indices, [-1])  # (fg,)

    # Get labels of foreground / background.
    class_batch = tf.gather(obj_labels, keep_ind)  # (B,)
    iou_batch = tf.gather(max_overlaps, keep_ind)  # (B,)

    # Collect masks given indices.
    roi_mask_batch = tf.gather(gt_masks, tf.gather(gt_assignment, keep_ind))  # (B, im_width, im_height)

    # Set non-foreground mask to class = 0.
    bin_vector = tf.cast(tf.concat([tf.ones(tf.shape(foreground_indices)[0]),
                            tf.zeros(tf.shape(background_indices)[0])], axis=0), tf.int32)
    class_batch = tf.multiply(class_batch, bin_vector)
    roi_mask_batch = tf.multiply(tf.reshape(bin_vector, (-1, 1, 1, 1)), roi_mask_batch)  # (B, im_width, im_height)

    # Define sample weights.
    rois_bg_obj_indices = tf.pad(tf.ones(fg_num_), [[0, batch_size - fg_num_]], "CONSTANT")

    # Sample ROIs and ground truth boxes given the
    # foreground / background indices.
    roi_bbox_batch = tf.gather(roi_bbox, keep_ind)  # (B, 4)
    roi_scores_batch = tf.gather(roi_scores, keep_ind)  # (B,)
    bbox_gt_batch = tf.gather(gt_bbox, tf.gather(gt_assignment, keep_ind))  # (B, 4)

    # Get ratio of ground truth boxes matching anchors at
    # specified IoU foreground threshold.
    unique = tf.unique_with_counts(tf.gather(gt_assignment, pos_keep_ind))
    count = tf.cast(tf.shape(unique.count)[0], tf.float32)
    assigned_gt_bbox_ratio = count / tf.cast(tf.shape(gt_bbox)[0], tf.float32)

    return (class_batch, roi_mask_batch, roi_bbox_batch, roi_scores_batch,
            bbox_gt_batch, rois_bg_obj_indices, keep_ind,
            assigned_gt_bbox_ratio, iou_batch)


def vectorize(inputs, num_classes):
    """Returns binary vectors with four 1s positioned at a
    specified index. This index is the box class itself as
    the target coordinates shall be extracted for this class."""
    row_size = 4 * num_classes
    left = inputs[0] * 4
    right = row_size - (left + 4)
    return tf.concat([tf.pad(inputs[1:5], [[left, right]]),
                      tf.pad(inputs[5:], [[left, right]])], axis=0)


def get_bbox_regression_targets(regression_targets, labels, num_classes):
    """Returns bounding-box regression targets with foreground level (no
    background boxes learnt).

    Targets are stored in compact form b x N x (class, tx, ty, tw, th).
    This function expands those targets into the 4-of-4*K representation
    used by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): b x N x 4K blob of regression targets
        bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
    """
    # Define the data matrix to slice with mapping function.
    to_vectorize = tf.concat([
        tf.reshape(labels, (-1, 1)), regression_targets,
        tf.ones_like(regression_targets)], axis=-1)  # (S, 9)

    # Then, extend the last dimension of this sparse matrix to get
    # dimension 16 (instead of 4) for targets and weights. Here, we
    # use a function mapping each row of the sparse matrix into the
    # dimension wanted (i.e. 8 -> 16+16, the 1st dimension of matrix
    # being the index considered for the position of values in the
    # row).
    to_slice = tf.map_fn(
        fn=lambda x: vectorize(x, tf.cast(num_classes, tf.float32)),
        elems=tf.cast(to_vectorize, tf.float32))
    targets = tf.slice(to_slice, [0, 0], [-1, 4 * num_classes])
    targets_weights = tf.slice(to_slice, [0, 4 * num_classes], [-1, -1])
    return targets, targets_weights


def define_regression_targets(rois, gt_boxes, norm_mean, norm_std, normalize=False):
    """Returns the bounding-box regression targets for an image.

    Args:
        rois (tensor): the regions of interest bounding boxes.
        gt_boxes (tensor): ground truth bounding boxes.
        norm_mean (tensor): the mean coordinate values considered
            for normalization.
        norm_std (tensor): the standard deviation of coordinate
            values considered for normalization.
        normalize (bool): if True, normalize the coordinates of the
            target bounding boxes.
     """
    targets = target_regression_transform(rois, gt_boxes)  # (S, 4)
    if normalize:
        # Optionally normalize regression targets by a precomputed mean
        # and standard deviation.
        targets = ((targets - norm_mean) / norm_std)  # (S, 4)
    return targets


def target_regression_transform(rois, gt_boxes):
    """Returns RoIs with formatted regression targets.

    Args:
        rois (tensor): the regions of interest bounding boxes.
        gt_boxes (tensor): ground truth bounding boxes.
    """
    ex_widths = rois[:, 2] - rois[:, 0] + 1.0  # (x2 - x1)
    ex_heights = rois[:, 3] - rois[:, 1] + 1.0  # (y2 - y1)
    ex_ctr_x = rois[:, 0] + (0.5 * ex_widths)  # center_x = x1 + (0.5 * w)
    ex_ctr_y = rois[:, 1] + (0.5 * ex_heights)  # center_y = x2 + (0.5 * h)

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + (0.5 * gt_widths)
    gt_ctr_y = gt_boxes[:, 1] + (0.5 * gt_heights)

    # Define regression targets.
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = tf.math.log(gt_widths / ex_widths)
    targets_dh = tf.math.log(gt_heights / ex_heights)

    return tf.transpose(tf.stack((targets_dx, targets_dy, targets_dw, targets_dh), axis=0))  # (S, 4)
