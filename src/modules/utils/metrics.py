import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    """The metric module calculating the F1 score.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset).
    """
    def __init__(self, name=None, dtype=None, interval=1., **kwargs):
        super(F1Score, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="f1_score", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1

    @staticmethod
    def base_metrics(y_pred, y_true):
        """Returns the true positives, and false positives
        and negatives."""
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        return tp, fp, fn

    def precision(self, y_pred, y_true):
        """Returns the precision value."""
        tp, fp, _ = self.base_metrics(y_pred, y_true)
        return tp / (tp + fp + tf.keras.backend.epsilon())

    def recall(self, y_pred, y_true):
        """Returns the recall value."""
        tp, _, fn = self.base_metrics(y_pred, y_true)
        return tp / (tp + fn + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new F1 score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        recall = self.recall(y_pred, y_true)
        precision = self.precision(y_pred, y_true)
        f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        self.score.assign_add(f1)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class MacroF1Score(tf.keras.metrics.Metric):
    """The metric module calculating the macro F1 score.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset).
    """

    def __init__(self, name=None, dtype=None, interval=1., **kwargs):
        super(MacroF1Score, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="macro_f1_score", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1

    @staticmethod
    def base_metrics(y_pred, y_true):
        # Reduce over axis 0 for class-wise metric.
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        return tp, fp, fn

    def precision(self, y_pred, y_true):
        """Returns the precision value."""
        tp, fp, _ = self.base_metrics(y_pred, y_true)
        return tp / (tp + fp + tf.keras.backend.epsilon())

    def recall(self, y_pred, y_true):
        """Returns the recall value."""
        tp, _, fn = self.base_metrics(y_pred, y_true)
        return tp / (tp + fn + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new macro F1
        score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.reset_state()

        # Get positive class in current state.
        pos_ind = tf.where(tf.reduce_sum(y_true, axis=0) > 0)

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        recall = self.recall(y_pred=tf.cast(y_pred, tf.float32), y_true=tf.cast(y_true, tf.float32))  # (num_classes,)
        precision = self.precision(y_pred=tf.cast(y_pred, tf.float32), y_true=tf.cast(y_true, tf.float32))
        f1 = (2 * precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        self.score.assign_add(tf.reduce_mean(tf.gather(f1, pos_ind)))  # average over classes
        self.counter.assign_add(tf.constant(1.))  # only add 1

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class CustomMetric(tf.keras.metrics.Metric):
    """The metric module calculating a custom score provided
    by a function outside this module.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        state_counter: the counter value specified to
            determine when to reset the state of this
            module.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset).
    """

    def __init__(self, name=None, dtype=None, interval=1., **kwargs):
        super(CustomMetric, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name=name+"_score", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.state_counter = tf.Variable(0., name="state_counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1

    def update_state(self, value, size):
        """Updates the metric state with a new score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.state_counter, self.reset_state_interval), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        self.score.assign_add(value)
        self.counter.assign_add(tf.cast(size, tf.float32))
        self.state_counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class Precision(tf.keras.metrics.Metric):
    """The metric module calculating the precision score.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset)
        multiclass (bool): if True, calculate the precision
            score per class.
    """

    def __init__(self, name=None, dtype=None, interval=1., multiclass=False, **kwargs):
        super(Precision, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="precision", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1
        self.multiclass = multiclass

    @staticmethod
    def precision(y_pred, y_true, axis=None):
        """Returns the precision value."""
        tp = tf.reduce_sum(y_true * y_pred, axis=axis)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
        return tp / (tp + fp + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new precision
        score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        if self.multiclass:
            # Get positive class in current state,
            # from (B, num_classes) matrix.
            pos_ind = tf.where(tf.reduce_sum(y_true, axis=0) > 0)  # (pos_classes)
            precision = self.precision(y_pred, y_true, axis=0)
            precision = tf.reduce_mean(tf.gather(precision, pos_ind))
        else:
            precision = self.precision(y_pred, y_true)

        self.score.assign_add(precision)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class Recall(tf.keras.metrics.Metric):
    """The metric module calculating the recall score.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset)
        multiclass (bool): if True, calculate the recall
            score per class.
    """

    def __init__(self, name=None, dtype=None, interval=1., multiclass=False, **kwargs):
        super(Recall, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="recall", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1
        self.multiclass = multiclass

    @staticmethod
    def recall(y_pred, y_true, axis=None):
        """Returns the recall value."""
        tp = tf.reduce_sum(y_true * y_pred, axis=axis)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=axis)
        return tp / (tp + fn + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new recall score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, 5), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        if self.multiclass:
            # Get positive class in current state,
            # from (B, num_classes) matrix.
            pos_ind = tf.where(tf.reduce_sum(y_true, axis=0) > 0)  # (pos_classes)
            recall = self.recall(y_pred, y_true, axis=0)
            recall = tf.reduce_mean(tf.gather(recall, pos_ind))
        else:
            recall = self.recall(y_pred, y_true)

        self.score.assign_add(recall)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class Specificity(tf.keras.metrics.Metric):
    """The metric module calculating the specificity score.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset)
        multiclass (bool): if True, calculate the specificity
            score per class.
    """

    def __init__(self, name=None, dtype=None, interval=1., multiclass=False, **kwargs):
        super(Specificity, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="specificity", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1
        self.multiclass = multiclass

    @staticmethod
    def specificity(y_pred, y_true, axis=None):
        """Returns the specificity value."""
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=axis)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=axis)
        return tn / (tn + fp + tf.keras.backend.epsilon())

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new specificity
        score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        if self.multiclass:
            # Get positive class in current state,
            # from (B, num_classes) matrix.
            pos_ind = tf.where(tf.reduce_sum(y_true, axis=0) > 0)  # (pos_classes)
            specificity = self.specificity(y_pred, y_true, axis=0)
            specificity = tf.reduce_mean(tf.gather(specificity, pos_ind))
        else:
            specificity = self.specificity(y_pred, y_true)

        self.score.assign_add(specificity)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)


class AUC(tf.keras.metrics.Metric):
    """The metric module calculating the AUC score.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset)
        metric: the keras module calculating the AUC value
            based on the PR curve.
    """

    def __init__(self, thresholds, name=None, dtype=None, interval=1., **kwargs):
        super(AUC, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="seg_accuracy", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1
        self.metric = tf.keras.metrics.AUC(thresholds=thresholds, curve='PR', from_logits=False)

    def update_state(self, y_pred, y_true):
        """Updates the metric state with a new AUC score."""
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.metric.reset_state()

        # Calculate AUC
        self.metric.update_state(y_true, y_pred)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        """Returns the average F1 score."""
        return self.metric.result()

    def reset_state(self):
        """Resets the metric state variables."""
        self.metric.reset_state()
        self.counter.assign(0.)


class SegAccuracy(tf.keras.metrics.Metric):
    """The metric module for estimating the accuracy of segmentation from
     the mask branch.

    Attributes:
        score: the F1 score accumulated.
        counter: the total number of scores accumulated.
            The average F1 score is calculated based on
            this number.
        reset_state_interval (int): the step interval at
            which the average F1 score is set to 0 (reset).
     """

    def __init__(self, name=None, dtype=None, interval=1., **kwargs):
        super(SegAccuracy, self).__init__(name=name, dtype=dtype, **kwargs)

        self.score = tf.Variable(0., name="seg_accuracy", trainable=False)
        self.counter = tf.Variable(0., name="counter", trainable=False)
        self.reset_state_interval = interval if tf.greater(interval, 0) else 1

    @staticmethod
    def base_metrics(y_pred, y_true):
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        return tp, fp, fn

    def update_state(self, y_pred, y_true):
        # Reset state at given interval.
        if tf.equal(tf.math.floormod(self.counter, self.reset_state_interval), 0):
            self.reset_state()

        # Measures the metrics and stores them in the object,
        # so it can later be retrieved with result.
        tp, fp, fn = self.base_metrics(y_pred, y_true)

        # Calculate segmentation accuracy according to VOC 2010 formula.
        mean_accuracy = tp / (tp + fp + fn + tf.keras.backend.epsilon())
        self.score.assign_add(mean_accuracy)
        self.counter.assign_add(tf.constant(1.))

    def result(self):
        return self.score.value() / self.counter

    def reset_state(self):
        """Resets the metric state variables."""
        self.score.assign(0.)
        self.counter.assign(0.)
