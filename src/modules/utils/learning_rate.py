import tensorflow as tf


class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Defines the exponential decay of a learning rate.

    Attributes:
        initial_lr (float): the initial learning rate.
        decay_rate (float): the decaying rate.
        decay_steps (int): the decaying step.
        staircase (bool) if True, apply a staircase type
            of decaying.
        minimum_lr (float): minimum learning rate to
            reach while decaying.
    """

    def __init__(self, initial_lr, decay_rate, decay_steps, minimum_lr, staircase):
        self.initial_lr = initial_lr
        self.decay_rate = tf.cast(decay_rate, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.staircase = staircase
        self.minimum_lr = tf.cast(minimum_lr, tf.float32)

    def __call__(self, step):
        """Returns a decayed learning rate."""
        if self.staircase:
            # Apply staircase function.
            exponent = tf.math.floordiv(tf.cast(step, tf.float32), self.decay_steps)
        else:
            exponent = tf.math.divide(tf.cast(step, tf.float32), self.decay_steps)
        # Apply decaying.
        lr = self.initial_lr * tf.math.pow(self.decay_rate, exponent)
        # Make sure calculated learning rate is not
        # below a given minimum.
        return tf.math.maximum(lr, self.minimum_lr)


class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Defines the cyclic change of a learning rate.

    Attributes:
        base_lr (float): the initial learning rate, or
            the lower boundary in the cycle.
        step_size (int) number of training iterations per
            half cycle.
        max_lr (float): upper boundary of learning rate
            in the cycle.
        gamma (float): function amplitude scaling factor.
    """

    def __init__(self, base_lr, max_lr, step_size, gamma):
        self.base_lr = tf.cast(base_lr, tf.float32)
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.step_size = tf.cast(step_size, tf.float32)
        self.gamma = tf.cast(gamma, tf.float32)

    def __call__(self, step):
        # Apply decaying.
        cycle = tf.math.floor(1. + tf.cast(step, tf.float32) / (2. * self.step_size))
        x = tf.math.abs(tf.cast(step, tf.float32) / self.step_size - 2 * cycle + 1)
        amplitude = tf.math.pow(self.gamma, tf.cast(step, tf.float32))
        return self.base_lr + (self.max_lr - self.base_lr) * tf.math.maximum(0., (1. - x)) * amplitude


class NoDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Defines a normal learning rate.

    Attributes:
        initial_lr (float): the initial learning rate.
    """

    def __init__(self, initial_lr):
        self.initial_lr = initial_lr

    def __call__(self, step):
        """Returns the initial learning rate without change."""
        return self.initial_lr
