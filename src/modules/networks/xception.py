import os
import tensorflow as tf

from tensorflow.keras import layers, models


def Xception(include_weights, strides, reg_coef, weights_folder, pooling=None, **kwargs):
    """Instantiates the Xception architecture.

    Args:
        include_weights (bool): if True, load the ImageNet weights
            to the model.
        strides (list): stride coefficients used in specific layers.
        reg_coef (float): backbone regularization coefficients.
        weights_folder (str): root path to weight folder.
        pooling (bool): optional pooling mode for feature extraction

    Returns
        A Keras model instance.
    """
    img_input = layers.Input(shape=(None, None, 3))

    x = layers.Conv2D(
        32, (3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block1_conv1')(img_input)
    x = layers.BatchNormalization(axis=-1, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(
        64, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=-1, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(
        128, (1, 1),
        strides=(2, 2),
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        use_bias=False)(x)
    residual = layers.BatchNormalization(axis=-1)(residual)

    x = layers.SeparableConv2D(
        128, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=-1, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        128, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=-1, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        256, (1, 1),
        strides=(2, 2),
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        use_bias=False)(x)
    residual = layers.BatchNormalization(axis=-1)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        256, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=-1, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        256, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=-1, name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        728, (1, 1),
        strides=strides[0],
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        use_bias=False)(x)
    residual = layers.BatchNormalization(axis=-1)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=-1, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=-1, name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=strides[0],
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
            name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=-1,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
            name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=-1,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(
            728, (3, 3),
            padding='same',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
            name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=-1,
                                      name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(
        1024, (1, 1),
        strides=strides[1],
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        use_bias=False)(x)
    residual = layers.BatchNormalization(axis=-1)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(
        728, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=-1, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(
        1024, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=-1, name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=strides[1],
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(
        1536, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block14_sepconv1')(x)
    x = layers.BatchNormalization(axis=-1, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(
        2048, (3, 3),
        padding='same',
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='block14_sepconv2')(x)
    x = layers.BatchNormalization(axis=-1, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)

    # Create model.
    model = models.Model(img_input, x, name='xception')

    # Load weights.
    if include_weights:
        weights_path = os.path.join(
            weights_folder,
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
        model.load_weights(weights_path, by_name=True)

    return model


class Model:
    """Module of the instantiated backbone network.

    Attributes:
        model (object): the backbone model to train.
        last_conv_channel_dim: (int): output channels of the
            last convolutional layer.
    """

    def __init__(self, trainable, add_initial_weights, layers_to_change, weights_folder,
                 reg_coef, **kwargs):
        # Define strides.
        two_coef = max(0, 2 - layers_to_change)
        one_coef = min(2, layers_to_change)
        strides = ([2] * two_coef) + ([1] * one_coef)

        # Build model
        self.model = Xception(
            strides=strides,
            include_weights=add_initial_weights,
            weights_folder=weights_folder,
            reg_coef=reg_coef, **kwargs)

        # Freeze layer weights (default: not trainable).
        self.model.trainable = trainable
        self.last_conv_channel_dim = 2048

    def __call__(self, inputs, *args, **kwargs):
        x = self.model(inputs)
        # Rule out network output from gradient calculation
        # if network not trained.
        if not self.model.trainable:
            x = tf.stop_gradient(x)
        return x
