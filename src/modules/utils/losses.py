import tensorflow as tf


def smooth_l1_loss(y_pred, y_true, sample_weights):
    """Returns the smooth L1 regression loss.

    The loss is basically the L1 norm, but when the L1
    error is small enough, defined by a certain sigma,
    the error is considered almost correct and the loss
    diminishes at a faster rate.

    Args:
        y_pred (tensor): predicted scores.
        y_true (tensor): reference scores to learn from.
        sample_weights (tensor): the positive and negative
            scores.
    """

    # Get the loss object.
    # Note: tf.keras.losses.Reduction.NONE applies reduction to
    # last axis once (no additional reduction).
    loss_obj = tf.keras.losses.Huber(
        name="huber_loss",
        reduction=tf.keras.losses.Reduction.NONE)

    # Select foreground box coordinates to calculate loss from.
    pos_indices = tf.where(sample_weights > 0)
    bbox_enc_fg = tf.gather_nd(y_pred, pos_indices)
    bbox_targets_fg = tf.gather_nd(y_true, pos_indices)

    # Calculate loss by applying sample_weights to calculate
    # the loss for foreground boxes only.
    return loss_obj(y_true=bbox_targets_fg, y_pred=bbox_enc_fg)  # (num_fg, 1)


def rpn_score_loss(y_true, y_prob, loss_weights):
    """Returns the weighted cross entropy loss of
    object scores.

    Args:
        y_true (tensor): reference scores to learn from.
        y_prob (tensor): object scores probabilities.
        loss_weights (tensor): the positive and negative
            object scores.
    """
    pos_indices = tf.where(tf.round(y_true) > 0)
    ce_loss = tf.math.multiply(tf.cast(y_true, tf.float32), tf.math.log(y_prob))
    total_loss = -tf.math.reduce_mean(tf.gather_nd(ce_loss, pos_indices)) * loss_weights[1]

    return total_loss


def head_class_loss(y_true, y_logits, bg_obj_indices, bg_obj_weights, num_classes):
    """Calculates the cross-entropy loss for classes assigned
    to estimated boxes that contain objects (positives), no
    objects (negatives).

    Args:
        y_true (tensor): reference scores to learn from.
        y_logits (tensor): output of classifier.
        bg_obj_indices (float): indices of boxes associated
            to the background in the batch.
        bg_obj_weights (tensor): the positive and negative
            scores.
        num_classes (int): the number of classes.
    """
    pos_indices = tf.where(bg_obj_indices > 0.)
    y_one_hot = tf.cast(tf.one_hot(y_true, depth=num_classes), tf.float32)
    pos_y_one_hot = tf.gather_nd(y_one_hot, pos_indices)
    pos_logits = tf.gather_nd(y_logits, pos_indices)

    total_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=pos_logits,
            labels=pos_y_one_hot)) * bg_obj_weights[1]

    return total_loss


def smooth_labels(labels, factor=0.1):
    """Returns smoothed labels."""
    labels *= (1. - factor)
    labels += (factor / tf.cast(tf.shape(labels)[0], tf.float32))
    # returned the smoothed labels
    return labels
