import tensorflow as tf

from tensorflow.keras import layers
from src.modules.utils.rcnn import bbox_ops
from src.modules.utils import metrics, iou, losses
from src.modules.utils.rcnn.anchor_targets import expand_by_random_sampling


class RPN(tf.keras.Model):
    """Instantiates the Region proposal network (RPN).

    RPN learns a convolution filter for bounding boxes at each
    grid location. Weights are learnt by looking across the 512
    last backbone features, and determining which grid cells
    likely contain an object, and the bounding box for objects
    in each grid cell.

    Attributes:
        num_anchors (int): number of anchors.
        image_width (int): width of image from which bounding boxes
            are estimated.
        image_height (int): height of image from which bounding boxes
            are estimated.
        metric_interval (int): interval at which metrics of the model
            are calculated.
        scores_prob (tensor): object score probabilities.
        scores_enc (tensor): network output for object scores.
        bbox_enc (tensor): network output for bounding boxes.
        trainable (bool): if True, this object model is set to training
            mode.
        precision: metric module calculating the precision of the
            classification.
        recall: metric module calculating the recall of the
            classification.
        specificity: metric module calculating the specificity
            of classification.
        f1_score: metric module calculating the f1 score of
            classification.
        iou_metric: metric module specifying the IoU of foreground
            proposal coordinates.
    """

    def __init__(self, num_anchor_types, image_width, image_height,
                 trainable, metric_interval, reg_coef, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.num_anchors = num_anchor_types
        self.image_width = image_width
        self.image_height = image_height
        self.metric_interval = metric_interval
        self.bbox_enc_out = tf.zeros((0, 1, 1, num_anchor_types*4))
        self.scores_prob = tf.zeros((0, 1))
        self.scores_enc = tf.zeros((0, 0, 2))
        self.bbox_enc = tf.zeros((0, 0, 4))
        self.trainable = trainable
        self.precision = metrics.Precision(name="rpn_precision", interval=self.metric_interval)
        self.recall = metrics.Recall(name="rpn_recall", interval=self.metric_interval)
        self.specificity = metrics.Specificity(name="rpn_specificity", interval=self.metric_interval)
        self.f1_score = metrics.F1Score(name="rpn_f1_score", interval=self.metric_interval)
        self.iou_metric = metrics.CustomMetric(name="rpn_iou", interval=self.metric_interval)

        # To detect objects, learn the kernel parameters which combine
        # the context of all 512 feature maps from the last conv. layer of
        # backbone network. Output activations correspond to the grid cells
        # of input image.
        self.rpn_features = layers.Conv2D(
            512, 3, 1,
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
            name='rpn_features')

        # Get parameters of regions of interest (RoIs).
        # If RPN output has dimension e.g. 38x38, each cell is linked to
        # e.g. 9 anchors (3 scales * 3 ratios). Given 2 object probabilities
        # and 4 coordinates to predict per anchor:
        # - rois_cls: <2*num_anchor> = 2*9 = 18 classes per cell.
        # - rois_reg: <4*num_anchor> = 4*9 = 36 coordinates per cell.
        self.rpn_score_predictor = layers.Conv2D(
            2 * self.num_anchors, 1, 1,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
            padding='valid',
            activation=None,
            name='rpn_cls')

        self.rpn_bbox_encoder = layers.Conv2D(
            4 * self.num_anchors, 1, 1,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
            padding='valid',
            activation=None,
            name='rpn_bbox')

    def build_variables_from_call(self, last_dimension, **kwargs):
        # Call model once to access variables (for restoration).
        _ = self.call(tf.ones((1, 1, 1, last_dimension)), tf.zeros((1, 4)))

    @tf.function
    def call(self, inputs, anchors=None):
        """Returns object scores and box regression values."""
        # Get RPN features.
        features = self.rpn_features(inputs)  # (B, W, H, C)
        batch_size = tf.shape(features)[0]
        width = tf.shape(features)[1]
        height = tf.shape(features)[2]

        # Get class and bounding box.
        scores_enc_out = self.rpn_score_predictor(features)  # (B, W, H, 9*2)
        bbox_enc_out = self.rpn_bbox_encoder(features)  # (B, W, H, 9*4)

        # Get object probability per grid cell while filtering
        # out predicted bounding boxes by score.
        self.scores_enc = tf.reshape(
            tensor=scores_enc_out,
            shape=[batch_size, width * height, self.num_anchors * 2])  # (B, W * H, 9 * 2)
        self.scores_enc = tf.reshape(
            tensor=self.scores_enc,
            shape=[batch_size, width * height * self.num_anchors, 2])  # (B, W * H * 9, 2)
        self.scores_prob = tf.nn.softmax(self.scores_enc, axis=-1)  # (B, W * H * 9, 2)

        # Get bbox coordinates per grid cell.
        self.bbox_enc = tf.reshape(
            tensor=bbox_enc_out,
            shape=[batch_size, width * height, self.num_anchors * 4])  # (B, W * H, 9 * 4)
        self.bbox_enc = tf.reshape(
            tensor=self.bbox_enc,
            shape=[batch_size, width * height * self.num_anchors, 4])  # (B, W * H * 9, 4)

        # Get proposals after transform and clipping.
        proposals = bbox_ops.get_rpn_proposals_from_targets(
            anchors=anchors,
            targets=self.bbox_enc,
            image_width=self.image_width,
            image_height=self.image_height)  # (B, obs_n, 4)

        # Stop gradient calculation for proposals because
        # gradients ignored w.r.t. the bounding boxes
        # coordinates. RoI pooling is differentiable, but
        # its gradient calculation makes early training
        # unstable.
        proposals = tf.stop_gradient(proposals)

        if not self.trainable:
            self.scores_prob = tf.stop_gradient(self.scores_prob)
            self.scores_enc = tf.stop_gradient(self.scores_enc)
            self.bbox_enc = tf.stop_gradient(self.bbox_enc)

        return proposals, self.scores_prob, self.scores_enc, self.bbox_enc

    @staticmethod
    def convert_coords(coordinates):
        """Returns center coordinates and box dimensions."""
        x1 = coordinates[:, 0]
        y1 = coordinates[:, 1]
        x2 = coordinates[:, 2]
        y2 = coordinates[:, 3]
        cent_x = tf.cast((x1 + x2) / 2, dtype=tf.float32) + 1.0
        cent_y = tf.cast((y1 + y2) / 2, dtype=tf.float32) + 1.0
        width = tf.cast((x2 - x1), dtype=tf.float32) + 1.0
        height = tf.cast((y2 - y1), dtype=tf.float32) + 1.0
        return cent_x, cent_y, width, height

    @tf.function(input_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.bool),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.bool),
            tf.TensorSpec(shape=None, dtype=tf.float32)))
    def build_rpn_targets(
            self, anchors, gt_index_assignments, gt_boxes, anchor_scores,
            rpn_scores, rpn_bbox_enc, rpn_proposals, normalize, norm_mean,
            norm_std, batch_size, resample_positives, resample_rate):
        """Returns the bounding box targets to learn from.

        Args:
            anchors (tensor): the anchor bounding box coordinates.
            gt_index_assignments (tensor): the ground truth box label
                assigned to anchors based on IoU scores.
            gt_boxes (tensor): ground truth bounding boxes.
            anchor_scores (tensor): object scores (positive and negative)
                assigned to anchor bounding boxes.
            rpn_scores (tensor): the estimated RPN object scores.
            rpn_bbox_enc (tensor): output of the regression network.
            rpn_proposals (tensor): coordinates of the bounding box proposals.
            normalize (bool): if True, normalize regression targets by a
                precomputed mean and standard deviation.
            norm_mean (float): precomputed mean for normalization.
            norm_std (float): precomputed mean for standard deviation.
            batch_size (int): the batch size.
            resample_rate (float): ratio of batch made of positives.
            resample_positives (bool): if True, resample positives
                in batch to balance the negative and positive
                instances.
        """
        # Find indices of positive and negative anchors.
        pos_indices_to_learn = tf.reshape(tf.where(tf.equal(anchor_scores, 1)), [-1])  # (rpn_B,1)
        neg_indices_to_learn = tf.reshape(tf.where(tf.equal(anchor_scores, 0)), [-1])  # (rpn_B,1)

        # Resample positives from current batch to reduce
        # imbalance effect on training.
        if resample_positives:
            max_num = tf.cast(batch_size * resample_rate, tf.int32)
            num_pos = tf.shape(pos_indices_to_learn)[0]
            if tf.less(num_pos, max_num - 1):
                pos_indices_to_learn = expand_by_random_sampling(max_num, pos_indices_to_learn)
                neg_indices_to_learn = neg_indices_to_learn[:max_num]
        batch_indices_to_learn = tf.concat([pos_indices_to_learn, neg_indices_to_learn], axis=0)  # (rpn_B,)

        # Get batch of labels.
        pos_neg_gt = tf.gather(anchor_scores, batch_indices_to_learn)
        pos_neg_gt = tf.cast(pos_neg_gt, dtype=tf.int32)  # (rpn_B,1)

        # Get batch of anchors.
        anchor_batch = tf.gather(anchors, batch_indices_to_learn)  # (rpn_B,4)
        anchor_cent_x, anchor_cent_y, anchor_w, anchor_h = self.convert_coords(anchor_batch)

        # Get ground truth boxes center positions and dimensions.
        gt_assignment = tf.gather(gt_index_assignments, batch_indices_to_learn)  # (rpn_B, 1)
        gt_boxes_batch = tf.gather(gt_boxes, gt_assignment)  # (rpn_B, 4)
        gt_cent_x, gt_cent_y, gt_bbox_w, gt_bbox_h = self.convert_coords(gt_boxes_batch)

        # Get batch of prediction scores.
        pos_neg_enc = tf.gather(rpn_scores, batch_indices_to_learn)
        pos_neg_enc = tf.cast(pos_neg_enc, dtype=tf.float32)  # (rpn_B,2)

        # Get batch of predicted bounding boxes and proposals.
        bbox_enc = tf.gather(rpn_bbox_enc, batch_indices_to_learn)  # (rpn_B, 4)
        proposals = tf.gather(rpn_proposals, batch_indices_to_learn)  # (rpn_B, 4)

        # Build regression targets.
        target_dx = tf.cast((gt_cent_x - anchor_cent_x) / anchor_w, dtype=tf.float32)
        target_dy = tf.cast((gt_cent_y - anchor_cent_y) / anchor_h, dtype=tf.float32)
        target_w = tf.cast(tf.math.log(gt_bbox_w / anchor_w), dtype=tf.float32)
        target_h = tf.cast(tf.math.log(gt_bbox_h / anchor_h), dtype=tf.float32)
        bbox_targets = tf.stack([target_dx, target_dy, target_w, target_h], axis=1)

        if normalize:
            # Optionally normalize regression targets by a
            # precomputed mean and standard deviation.
            bbox_targets = ((bbox_targets - norm_mean) / norm_std)  # (S, 4)

        return (pos_neg_gt, pos_neg_enc, bbox_targets,
                bbox_enc, proposals, gt_assignment)

    def get_loss(self, pos_neg_scores, scores_prob, bbox_targets, bbox_enc,
                 pos_neg_weights, loss_weights, label_smoothing,
                 smooth_factor):
        """Returns the losses of box regressions and object scores.

        Args:
            pos_neg_scores (tensor): ground truth object scores (positive
                and negative) assigned to anchor bounding boxes.
            scores_prob (tensor): object score probabilities.
            bbox_targets (tensor): bounding box targets to learn by the
                regressor.
            bbox_enc (tensor): output of the regression network.
            pos_neg_weights (tensor): weights of object score loss defined
                associated to positives and negatives.
            loss_weights (tensor): loss weights of object scores and
                regression.
            label_smoothing (bool): if True, apply label smoothing to
                classification scores.
            smooth_factor (float): factor used for label smoothing.
        """
        if self.trainable:
            # Smooth classification labels.
            if label_smoothing:
                _labels = losses.smooth_labels(tf.cast(pos_neg_scores, tf.float32), smooth_factor)
            else:
                _labels = pos_neg_scores

            # Get score loss, calculated separately
            # between positives (where predicted boxes are
            # expected to cover known ones), negatives (boxes
            # with low pixel value variations, i.e. background).
            cls_loss = losses.rpn_score_loss(
                y_true=_labels,
                y_prob=scores_prob,
                loss_weights=pos_neg_weights) * loss_weights[0]

            # Get regression loss.
            reg_loss = losses.smooth_l1_loss(
                y_pred=bbox_enc,
                y_true=bbox_targets,
                sample_weights=pos_neg_scores)  # (num_fg, 1)
            reg_loss = tf.reduce_mean(reg_loss) * loss_weights[1]

            # Add eventual regularization losses.
            total_loss = tf.reduce_sum(self.losses) + cls_loss + reg_loss
            return cls_loss, reg_loss, total_loss
        else:
            return tf.constant(0.), tf.constant(0.), tf.constant(0.)

    def get_metrics(self, y_pred, y_true, gt_bbox, proposals):
        """Return object scores and IoU metrics.

        Args:
            y_pred (tensor): predicted scores.
            y_true (tensor): reference scores to learn from.
            gt_bbox (tensor): ground truth bounding boxes.
            proposals (tensor): coordinates of the box proposals.
        """
        self.precision.update_state(y_pred=y_pred, y_true=y_true)  # how good to find a positive
        self.recall.update_state(y_pred=y_pred, y_true=y_true)  # how many positives found (out of all known)
        self.specificity.update_state(y_pred=y_pred, y_true=y_true)  # rate of FP; can be 0 if no TN in image
        self.f1_score.update_state(y_pred=y_pred, y_true=y_true)

        # Get IoU vector.
        iou_vector = iou.get_iou_vector(proposals, gt_bbox)
        self.iou_metric.update_state(
            value=tf.reduce_sum(iou_vector),
            size=tf.shape(iou_vector)[0])

        return [self.precision, self.recall, self.specificity,
                self.f1_score, self.iou_metric]
