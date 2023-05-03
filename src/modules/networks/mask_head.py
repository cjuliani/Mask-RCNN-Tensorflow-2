import tensorflow as tf

from tensorflow.keras import layers
from src.modules.utils import metrics
from src.modules.utils.rcnn.roi_ops import select_from_last_axis


class MaskHead(tf.keras.Model):
    def __init__(self, num_classes, trainable, metric_interval):
        super(MaskHead, self).__init__()
        # General attributes.
        self.num_classes = num_classes

        # Metric objects.
        self.accuracy = metrics.SegAccuracy(name="mask_accuracy", interval=metric_interval)

        # Layers.
        self.conv11 = layers.Conv2D(256, 3, 1, 'same', activation=tf.nn.relu, name='mask_conv11')
        self.conv12 = layers.Conv2D(256, 3, 1, 'same', activation=tf.nn.relu, name='mask_conv12')
        self.conv13 = layers.Conv2D(256, 3, 1, 'same', activation=tf.nn.relu, name='mask_conv13')
        self.conv14 = layers.Conv2D(256, 3, 1, 'same', activation=tf.nn.relu, name='mask_conv14')
        self.conv2 = layers.Conv2D(256, 3, 1, 'same', activation=tf.nn.relu, name='mask_conv2')
        self.logits = layers.Conv2D(num_classes, 1, 1, 'same', activation=None, name='conv_logits')
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), name='up_sampled')

        # Set as trainable network.
        self.trainable = trainable

    def build_variables_from_call(self, last_dim_of_input, **kwargs):
        # Call model once to access variables (for restoration).
        _ = self.call(tf.ones((1, 1, 1, last_dim_of_input)))

    @tf.function
    def call(self, inputs):
        # Get mask
        x = inputs
        c11 = self.conv11(x)
        c12 = self.conv12(c11)
        c13 = self.conv13(c12)
        c14 = self.conv14(c13)
        ups = self.upsampling(c14)  # (B, P*2, P*2, F)
        c21 = self.conv2(ups)
        x = self.logits(c21)
        # Rule out network output from gradient calculation
        # if network not trained.
        if not self.trainable:
            x = tf.stop_gradient(x)

        return tf.nn.sigmoid(x), x

    @staticmethod
    def get_pooled_masks(roi_pooled_masks_probs, pred_img_bboxes, roi_gt_masks,
                         gt_cls, bg_obj_indices):
        """During training, we scale down the ground-truth masks to 28x28
         to compute the loss, and during modules we scale up the predicted
         masks to the size of the ROI bounding box and that gives us the final
         masks, one per object."""
        # Note: masks are soft masks (with float pixel values) and of size 28x28.
        # Get pool siz|e.
        num_items = tf.shape(roi_pooled_masks_probs)[0]
        pool_size = tf.shape(roi_pooled_masks_probs)[1]

        # Select feature from last axis.
        fg_roi_pooled_masks_prob, _ = tf.map_fn(
            fn=lambda x: select_from_last_axis(x[0], x[1]),
            elems=[tf.cast(gt_cls, tf.float32), roi_pooled_masks_probs])  # (B, 28, 28)

        # Normalize coordinates of bounding box before using
        # tf.image.crop_and_resize
        height = tf.cast(tf.shape(roi_gt_masks)[1], tf.float32)
        width = tf.cast(tf.shape(roi_gt_masks)[2], tf.float32)
        x1, y1, x2, y2 = tf.split(pred_img_bboxes, 4, axis=1)
        bbox_norm = tf.concat(
            [y1/height, x1/width, y2/height, x2/width], axis=-1)

        # Scale down ground truth masks within regions defined
        # by RPN box coordinates, to roi pooled mask dimensions.
        try:
            fg_roi_gt_masks_pooled = tf.image.crop_and_resize(
                image=tf.expand_dims(roi_gt_masks, axis=-1),
                boxes=bbox_norm,
                box_indices=tf.range(num_items),
                crop_size=[pool_size, pool_size],
                method="nearest",
                name="gt_mask_crops")  # (num_items, pool, pool, 1)
        except Exception as err:
            raise Exception(err)
        fg_roi_gt_masks_pooled = tf.stop_gradient(tf.squeeze(fg_roi_gt_masks_pooled, axis=-1))  # (num_item, pool, pool)

        # Select foreground items
        sample_indices = tf.where(bg_obj_indices > 0)  # (fg,)
        fg_roi_pooled_masks_prob = tf.gather_nd(fg_roi_pooled_masks_prob, sample_indices)  # (fg, pool, pool)
        fg_roi_gt_masks_pooled = tf.gather_nd(fg_roi_gt_masks_pooled, sample_indices)  # (fg, pool, pool)
        return fg_roi_gt_masks_pooled, fg_roi_pooled_masks_prob

    @tf.function
    def get_loss(self, y_true, y_prob, loss_weight):
        if self.trainable:
            # Calculate loss
            seg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true,
                logits=y_prob,
                name='ce_mask_loss')
            seg_loss = tf.reduce_mean(seg_loss) * loss_weight

            # Add eventual regularization losses
            total_loss = tf.reduce_sum(self.losses) + seg_loss
            return seg_loss, total_loss
        else:
            return tf.constant(0.), tf.constant(0.)

    def get_metrics(self, y_prob, y_true, prob_thresh=0.5):
        # Reshape maps
        y_pred = tf.cast(tf.greater_equal(y_prob, prob_thresh), tf.float32)  # binarize scalars
        # Calculate average metrics per map
        self.accuracy.update_state(y_pred=y_pred, y_true=tf.cast(y_true, tf.float32))
        return [self.accuracy]
