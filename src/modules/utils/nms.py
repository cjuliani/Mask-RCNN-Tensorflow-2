import tensorflow as tf


def apply_nms(bboxes, scores_prob, top_n, iou_thresh, image_width, image_height, sigma=0.5):
    """Returns indices of bounding boxes kept after applying
    a non-maximum suppression (NMS).

    NMS takes the list of proposals sorted by score and iterates
    over the sorted list, discarding proposals that have an IoU
    larger than some predefined threshold with a proposal that
    has a higher score. If iou_thresh low (<0.5), we may end up
    missing proposals for objects. If high (>0.7), we may increase
    false positives. 0.6 is the standard value.

    Args:
        bboxes (tensor): coordinates of the bounding box proposals.
        scores_prob (tensor): probability values of object scores
            assigned to the proposals.
        top_n (int): the top N proposals sorted by score and
            returned by NMS.
        iou_thresh (float): a threshold value above which two
            boxes are considered duplicates if their overlap ratio
            exceeds the value.
        image_width (int): width of the image from which the bounding
            boxes were estimated.
        image_height (int): height of the image from which the bounding
            boxes were estimated.
        sigma (float): if not 0, NMS reduces the score of other
            overlapping boxes instead of directly causing them to be
            pruned. Consequently, it returns the new scores of each
            input box in the second output. Note: sigma must be larger
            than 0 to enable soft NMS.
            Read: https://arxiv.org/abs/1704.04503 for info.
    """
    # Bounding boxes are supplied as (y1, x1, y2, x2) to NMS.
    x1 = tf.slice(bboxes, [0, 0], [-1, 1]) / image_width
    y1 = tf.slice(bboxes, [0, 1], [-1, 1]) / image_height
    x2 = tf.slice(bboxes, [0, 2], [-1, 1]) / image_width
    y2 = tf.slice(bboxes, [0, 3], [-1, 1]) / image_height
    formatted_proposals = tf.concat([y1, x1, y2, x2], axis=1)

    # Get positive scores.
    with tf.name_scope('nms'):
        if tf.greater(sigma, 0):
            keep_indices, _ = tf.image.non_max_suppression_with_scores(
                boxes=formatted_proposals,
                scores=scores_prob,
                max_output_size=top_n,
                iou_threshold=iou_thresh,  # WARNING: iou threshold ignored when sigma > 0
                soft_nms_sigma=sigma)
        else:
            keep_indices = tf.image.non_max_suppression(
                boxes=formatted_proposals,
                scores=scores_prob,
                max_output_size=top_n,
                iou_threshold=iou_thresh)

    return keep_indices
