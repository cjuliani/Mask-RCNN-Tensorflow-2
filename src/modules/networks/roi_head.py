import tensorflow as tf

from tensorflow.keras import layers
from src.modules.utils import metrics, iou, losses
from src.modules.utils.rcnn import bbox_ops


class RoiHead(tf.keras.Model):
    """Region of Interest (RoI) head network.

    Use this module to bounding box coordinates and box classes.

    Attributes:
        num_classes (int): the number of classes.
        loop_iterator: tensorflow variable used for loop procedures.
        AUCs: a tensorflow variable collecting the areas under the
            curve (AUCs) metric values.
        cls_scores (float): network outputs for classification.
        cls_prob (float): estimated class score probabilities.
        cls_pred (float): estimated class score.
        bbox_enc (float): network outputs for bounding box coordinates.
        macro_f1_score: metric module calculating the f1 score
            per class.
        f1_score: metric module calculating the f1 score of
            classification.
        precision: metric module calculating the precision of
            the classification.
        recall: metric module calculating the recall of the
            classification.
        specificity: metric module calculating the specificity
            of classification.
        iou_metric: metric module specifying the IoU of foreground
            proposal coordinates.
        auc_metric: metric module specifying the average values of
            AUC when estimating the mean average precision (mAP).
        mAP50: metric module specifying the average values of mAP
            calculated from PR.
        mAP_custom: metric module specifying the average  values of
            mAP calculated from IoU and class vectors whose scores
            were limited at a threshold of 0.5.
        trainable (bool): if True, this object model is set to
            training mode.
        iou_thresholds (tensor): IoU value range utilized to calculate
            the AUC metric.
        auc_thresholds (tensor): AUC value range utilized to calculate
            the AUC metric.
        flatten: keras layer used to flatten input tensors.
        fc1: first dense layer making up the classifier.
        fc2: second dense layer making up the classifier.
        cls_scoring (tensor): outputs of the classifier.
        bbox_generator (tensor): outputs of the box regressor.
        max_pool: max pooling layer.
    """

    def __init__(self, num_classes, trainable, metric_interval):
        super(RoiHead, self).__init__()
        self.num_classes = num_classes
        self.loop_iterator = tf.Variable(0, trainable=False)
        self.AUCs = tf.Variable([], trainable=False, validate_shape=False)
        auc_spacing = 0.1
        self.iou_thresholds = tf.constant([0.25, 0.5, 0.75])  # COCO standard: tf.range(0.5, 0.95, 0.05)
        self.auc_thresholds = tf.range(0., 1. + auc_spacing, auc_spacing)  # COCO standard interval (101 thresholds)
        self.cls_scores = tf.zeros((0, num_classes))
        self.cls_prob = tf.zeros((0, num_classes))
        self.cls_pred = tf.zeros((0,))
        self.bbox_enc = tf.zeros((0, num_classes * 4))
        self.macro_f1_score = metrics.MacroF1Score(name="head_macro_f1_score", interval=metric_interval)
        self.f1_score = metrics.F1Score(name="head_f1_score", interval=metric_interval)
        self.precision = metrics.Precision(name="head_precision", interval=metric_interval, multiclass=True)
        self.recall = metrics.Recall(name="head_recall", interval=metric_interval, multiclass=True)
        self.specificity = metrics.Specificity(name="head_specificity", interval=metric_interval, multiclass=True)
        self.iou_metric = metrics.CustomMetric(name="head_iou", interval=metric_interval)
        self.auc_metric = metrics.AUC(
            thresholds=self.auc_thresholds, name="head_auc", interval=metric_interval)
        self.mAP50 = metrics.CustomMetric(name="mAP50", interval=metric_interval)
        self.mAP_custom = metrics.CustomMetric(name="mAP_custom", interval=metric_interval)
        self.trainable = trainable

        # Layers.
        self.flatten = layers.Flatten(name='roi_flatten')  # (B, -1)
        self.fc1 = layers.Dense(
            2048, activation=tf.nn.relu,
            name='roi_fc1')  # WARNING: Memory expensive if above 1024.
        self.fc2 = layers.Dense(
            2048, activation=tf.nn.relu,
            name='roi_fc2')
        self.cls_scoring = layers.Dense(
            num_classes, activation=None,
            name='roi_scoring')
        self.bbox_generator = layers.Dense(
            num_classes * 4,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            activation=None,
            name='roi_bbox_gen')
        self.max_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")

    def build_variables_from_call(self, last_dim_of_input, pool_size):
        """Calls learning model once to access variables (for
        # weights restoration)."""
        _ = self.call(tf.ones((1, pool_size, pool_size, last_dim_of_input)))

    @tf.function
    def call(self, inputs):
        """Returns classification scores and box regression values."""
        # Reduce input dimensions by pooling
        rois_pooled = self.max_pool(inputs)  # (B, P/2, P/2, F)
        # Apply head dense network
        flatten = self.flatten(rois_pooled)  # WARNING: Memory expensive if no prior pooling (dimension reduction).
        fc1 = self.fc1(flatten)  # (B, 1024)
        fc2 = self.fc2(fc1)  # (B, 1024)
        # Get bounding box classes
        self.cls_scores = self.cls_scoring(fc2)  # (B, num_classes)
        self.cls_prob = tf.nn.softmax(self.cls_scores, axis=-1, name="roi_cls_prob")  # (B, num_classes)
        self.cls_pred = tf.argmax(self.cls_prob, axis=-1, name="roi_cls_pred")  # (B, 1)
        # Get bounding boxes coordinates
        self.bbox_enc = self.bbox_generator(fc2)  # (B, 4 * num_classes)

        # Rule out network outputs from gradient calculation
        # if network not trained.
        if not self.trainable:
            self.cls_scores = tf.stop_gradient(self.cls_scores)
            self.cls_prob = tf.stop_gradient(self.cls_prob)
            self.cls_pred = tf.stop_gradient(self.cls_pred)
            self.bbox_enc = tf.stop_gradient(self.bbox_enc)

        return (self.cls_scores, self.cls_prob,
                self.cls_pred, self.bbox_enc)

    @staticmethod
    def slice(inputs):
        """Returns the coordinates of bounding boxes specified
        at given class labels (or indices).

        Define function creating binary vectors with four 1s positioned
        from a specified index. This index is the box class itself as
        the target coordinates shall be extracted for this class."""
        left = tf.cast(inputs[0],  # the label
                       tf.int32) * 4
        return tf.slice(inputs[1:], [left], [4])

    @tf.function
    def get_loss(self, bbox_targets_pred, bbox_targets,
                 cls_logits, gt_cls, bg_obj_indices,
                 bg_obj_weights, loss_weights, label_smoothing,
                 smooth_factor, num_classes, train_cls, train_reg):
        """Returns the losses of box regressions and classifications.

        Args:
            bbox_targets_pred (tensor): output of the regressor.
            bbox_targets (tensor): ground truth bounding box targets.
            cls_logits (tensor): output of the classifier.
            gt_cls (tensor): ground truth classes.
            bg_obj_indices (tensor): indices of background and foreground
                in batch.
            bg_obj_weights (tensor): loss weights assigned to background
                and foreground in batch.
            loss_weights (tensor): loss weights of classification and
                regression.
            label_smoothing (bool): if True, apply label smoothing to
                classification scores.
            smooth_factor (float): the label smoothing factor.
            num_classes (int): the number of classes.
            train_cls (bool): if True, train the classifier.
            train_reg (bool): if True, train the regressor.
        """
        if self.trainable:
            # Apply label smoothing.
            if label_smoothing:
                _labels = losses.smooth_labels(tf.cast(gt_cls, tf.float32), smooth_factor)
            else:
                _labels = gt_cls

            # Get classification loss, calculated separately
            # between positives (boxes with objects whose class
            # is known), negatives (boxes with low pixel
            # value variation, i.e. class background).
            cls_loss = losses.head_class_loss(
                y_true=gt_cls,
                y_logits=cls_logits,
                bg_obj_indices=bg_obj_indices,
                bg_obj_weights=bg_obj_weights,
                num_classes=num_classes) * loss_weights[0]

            if not train_cls:
                cls_loss = tf.stop_gradient(cls_loss)

            # Multiply predictions with weights to calculate the loss
            # only for relevant scalars of the tensor, which is of
            # dimension (B, num_classes * 4) - 4 coordinates per row
            # are relevant for this loss, so we make bbox_pred sparse.
            # Note: bbox_targets is already sparse, so we need to
            # make bbox_pred sparse as well.
            to_slice = tf.concat([
                tf.reshape(tf.cast(gt_cls, tf.float32), (-1, 1)), bbox_targets_pred], axis=-1)
            no_sparse_bbox_pred = tf.map_fn(
                fn=lambda x: self.slice(x),
                elems=tf.cast(to_slice, tf.float32))  # (B, 4)

            to_slice = tf.concat([
                tf.reshape(tf.cast(gt_cls, tf.float32), (-1, 1)), bbox_targets], axis=-1)
            no_sparse_bbox_targets = tf.map_fn(
                fn=lambda x: self.slice(x),
                elems=tf.cast(to_slice, tf.float32))  # (B, 4)

            reg_loss = losses.smooth_l1_loss(
                y_pred=no_sparse_bbox_pred,
                y_true=tf.stop_gradient(no_sparse_bbox_targets),
                sample_weights=bg_obj_indices)  # (num_fg, 1)
            reg_loss = tf.reduce_mean(reg_loss) * loss_weights[1]

            if not train_reg:
                reg_loss = tf.stop_gradient(reg_loss)

            # Add eventual regularization losses.
            total_loss = tf.reduce_sum(self.losses) + cls_loss + reg_loss
            return cls_loss, reg_loss, total_loss
        else:
            return tf.constant(0.), tf.constant(0.), tf.constant(0.)

    def get_class_metrics(self, y_pred, y_true):
        """Returns classification metrics.

        Args:
            y_pred (tensor): output of classifier.
            y_true (tensor): ground truth classes as input to
                the HEAD network.
        """
        # Make one-hot tensors.
        y_true_enc = tf.one_hot(y_true, depth=self.num_classes)  # (B, num_classes)
        y_pred_enc = tf.one_hot(y_pred, depth=self.num_classes)  # (B, num_classes)

        # Update metric values.
        self.macro_f1_score.update_state(y_true=y_true_enc, y_pred=y_pred_enc)
        self.precision.update_state(y_pred=y_pred_enc, y_true=y_true_enc)
        self.specificity.update_state(y_pred=y_pred_enc, y_true=y_true_enc)
        self.recall.update_state(y_pred=y_pred_enc, y_true=y_true_enc)
        self.f1_score.update_state(y_true=y_true_enc, y_pred=y_pred_enc)

        return [self.macro_f1_score, self.f1_score, self.precision,
                self.recall, self.specificity]

    @staticmethod
    def get_custom_AP(iou_vector, gt_cls, cls_prob, iou_thresh):
        """Returns the average precision (AP) calculated
        from IoU and class vectors whose scores were
        limited at a threshold of 0.5.

        Args:
            iou_vector (tensor): the IoU vector estimated.
            gt_cls (tensor): ground truth classes.
            cls_prob (tensor): class probabilities from the
                classifier.
            iou_thresh (float): the IoU threshold determining
                negative and positive samples.
        """
        # Verify conditions
        cls_pred = tf.cast(tf.argmax(tf.nn.softmax(cls_prob, axis=-1), axis=-1), tf.float32)
        iou_bin_vector = tf.where(tf.greater_equal(iou_vector, iou_thresh), 1., 0.)  # iou > thresh?
        cls_bin_vector = tf.cast(tf.equal(tf.cast(gt_cls, tf.float32), cls_pred), tf.float32)  # matching class?
        bin_vector = tf.math.multiply(iou_bin_vector, cls_bin_vector)  # True if IoU and class verified
        # Base metrics
        tp = tf.reduce_sum(bin_vector)  # TP if iou>=0.5 and correct class prediction
        fp = tf.reduce_sum(1. - bin_vector)
        return tp / (tp + fp + tf.keras.backend.epsilon())

    def get_mAP_PASCAL(self, iou_vector, gt_cls, cls_prob):
        """Returns the average precision (AP) calculated
        from PR curves.

        Args:
            iou_vector (tensor): the IoU vector estimated.
            gt_cls (tensor): ground truth classes.
            cls_prob (tensor): class probabilities from the
                classifier.
        """
        def get_class_auc(inputs):
            # Slice data
            detections = inputs[:tf.shape(inputs)[0] // 2]  # row (B,)
            confidence = inputs[tf.shape(inputs)[0] // 2:]
            # Update AUC with varying IoUs.
            self.auc_metric.update_state(y_true=detections, y_pred=confidence)
            return self.auc_metric.result()

        def body(thresholds, container):
            # Get probability class matrix and convert it
            # into vector shape.
            gt_cls_one_hot = tf.one_hot(gt_cls, depth=self.num_classes)  # (B, num_classes)

            # Build false positive/true positive vector based on iou.
            current_thresh = tf.slice(thresholds, [self.loop_iterator], [1])
            TP_FP_vec = tf.where(tf.greater_equal(iou_vector, current_thresh), 1., 0.)  # (B,)

            # Set rows of class matrix to 0 where FP exist.
            gt_cls_one_hot = gt_cls_one_hot * tf.reshape(TP_FP_vec, (-1, 1))

            # Get AUC class for current iou threshold.
            to_check = tf.concat([tf.transpose(gt_cls_one_hot),
                                  tf.transpose(cls_prob)], axis=-1)  # (num_classes, B*2)
            auc_per_class = tf.map_fn(
                fn=lambda x: get_class_auc(x),
                elems=tf.cast(to_check, tf.float32))  # (num_classes,)

            # Stack per-class AUC results
            self.loop_iterator.assign_add(1)
            return thresholds, tf.concat([container, auc_per_class], axis=0)

        def condition(thresholds, container):
            return tf.less(self.loop_iterator, tf.shape(thresholds)[0])

        _, auc_per_iou_thresh = tf.while_loop(
            condition, body, [self.iou_thresholds, self.AUCs],
            shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None])])  # (num_iou, num_classes)
        auc_per_iou_thresh = tf.reshape(
            auc_per_iou_thresh,
            shape=(tf.shape(self.iou_thresholds)[0], tf.constant(self.num_classes)))  # (num_iou, num_classes)

        # Get calculate mAP from positive classes
        # only among ground truth.
        cls_one_hot = tf.one_hot(gt_cls, depth=self.num_classes)
        pos_ind = tf.where(tf.reduce_sum(cls_one_hot, axis=0) > 0)  # (pos_classes)
        auc_per_iou_thresh = tf.gather(tf.transpose(auc_per_iou_thresh), tf.reshape(pos_ind, (-1,)))  # (pos, num_iou)
        auc_per_iou_thresh = tf.transpose(auc_per_iou_thresh)  # (num_iou, pos_classes)

        # Calculate per-iou mAP.
        per_iou_mAP = tf.reduce_mean(auc_per_iou_thresh, axis=1)

        # Reset state of iterator.
        self.loop_iterator.assign(0)

        return per_iou_mAP

    def get_proposals(self, gt_cls, targets, anchors, image_width, image_height):
        """Returns the bounding box proposals.

        Args:
            gt_cls (tensor): ground truth classes.
            targets (tensor): output from the regressor.
            anchors (tensor): coordinates of the anchor boxes.
            image_width (int): the width of image from which
                bounding boxes were estimated.
            image_height (int): the height of image from which
                bounding boxes were estimated.
        """
        # Get extract box targets for predicted object class.
        column_indices = tf.expand_dims(tf.cast(gt_cls, tf.float32), axis=-1)  # (B, 1)
        to_slice = tf.concat([column_indices, targets], axis=-1)  # (B, 4 * num_classes + 1)
        no_sparse_target_mat = tf.map_fn(
            fn=lambda x: self.slice(x),
            elems=tf.cast(to_slice, tf.float32))  # (B, 4)

        # Convert targets into box coordinates.
        return bbox_ops.get_head_proposals_from_targets(
            anchors=anchors,
            targets=no_sparse_target_mat,
            image_width=tf.cast(image_width, tf.float32),
            image_height=tf.cast(image_height, tf.float32))  # (B, 4)

    def get_box_metrics(self, gt_bbox, proposals, gt_cls, iou_thresh,
                        cls_prob, bg_obj_indices):
        """Returns detection metrics.

        Args:
            gt_bbox (tensor): ground truth bounding boxes.
            proposals (tensor): coordinates of the box proposals.
            gt_cls (tensor): ground truth classes.
            iou_thresh (float): the IoU threshold determining
                negative and positive samples.
            cls_prob (tensor): probability scores of object
                classes estimated from the HEAD network.
            bg_obj_indices (tensor): indices of boxes associated
                to the background in the batch.
        """
        # Select foreground box coordinates to calculate loss from.
        obj_indices = tf.reshape(tf.where(bg_obj_indices > 0), [-1])

        # Get iou per box.
        iou_vector = iou.get_iou_vector(proposals, gt_bbox)
        iou_vector_fg = tf.gather(iou_vector, obj_indices)
        self.iou_metric.update_state(
            value=tf.reduce_mean(iou_vector_fg),
            size=tf.constant(1.))

        # Calculate mean average precision (mAP) from PR curves
        # generated at given IoU thresholds and per class. Final
        # mAP corresponds to average class mAP while mAPxx
        # corresponds to mAP at IoU threshold xx.
        per_iou_mAP = self.get_mAP_PASCAL(iou_vector, gt_cls=gt_cls, cls_prob=cls_prob)
        mAP25, mAP50, mAP75 = tf.split(per_iou_mAP, num_or_size_splits=3)
        self.mAP50.update_state(value=mAP50[0], size=tf.constant(1.))

        # Calculate custom mAP.
        custom_AP = self.get_custom_AP(iou_vector, gt_cls, cls_prob, iou_thresh=iou_thresh)
        self.mAP_custom.update_state(value=custom_AP, size=tf.constant(1.))

        return [self.iou_metric, self.mAP50, self.mAP_custom]
