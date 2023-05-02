import os
import tensorflow as tf

from tensorflow.keras import layers, models


def ResNet(stack_fn, preact, use_bias, weights_folder, model_name='resnet',
           include_weights=True, reg_coef=1e-5, **kwargs):
    """Instantiates the ResNet architecture.

    Args:
        stack_fn (function): the convolutional blocks that compose the
            ResNet model (stacked).
        preact (bool): if True, add batch-normalization followed by
            an activation layer after the first convolution layer.
        use_bias (bool): if True, add bias to convolutional layers.
        weights_folder (str): root path to weight folder.
        model_name (str): name of the backbone model.
        include_weights (bool): if True, load the ImageNet weights
            to the model.
        reg_coef (float): backbone regularization coefficients.

    Returns
        A Keras model instance.
    """
    img_input = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(
        64, 7,
        strides=2,
        use_bias=use_bias,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='conv1_conv')(img_input)

    if preact is False:
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)
    # x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name='pool1_pool')(x)
    C2, C3, C4, C5 = stack_fn(x)
    if preact is True:
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                      name='post_bn')(C5)
        C5 = layers.Activation('relu', name='post_relu')(x)

    # Create model.
    model = models.Model(img_input, C5, name=model_name)

    # Load weights.
    if include_weights:
        weights_path = os.path.join(
            weights_folder,
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        model.load_weights(weights_path, by_name=True)
    return model


def block(x, filters, kernel_size=3, stride=1,
          conv_shortcut=True, reg_coef=1e-5, name=None):
    """A residual block.

    Args:
        x (tensor): the input tensor.
        filters (int): filters of the bottleneck layer.
        kernel_size (int): kernel size of the bottleneck layer.
        stride (int): stride of the first layer.
        conv_shortcut (bool): if True, use convolution shortcut,
            otherwise identity shortcut.
        reg_coef (float): regularization coefficients of convolution
            layers.
        name (str): block label.

    Returns
        Output tensor for the residual block.
    """
    if conv_shortcut is True:
        shortcut = layers.Conv2D(
            4 * filters, 1,
            strides=stride,
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
            name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
        filters, 1,
        strides=stride,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(
        4 * filters, 1,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack(x, filters, blocks, reg_coef, stride=2, name=None):
    """A set of stacked residual blocks.

    Args:
        x (tensor): the input tensor.
        filters (int): filters of the bottleneck layer in a block.
        blocks (int): blocks in the stacked blocks.
        reg_coef (float): regularization coefficients of convolution
            layers.
        stride (int): stride of the first layer in the first block.
        name (str): stack label.

    Returns:
        Output tensor for the stacked blocks.
    """
    x = block(x, filters, stride=stride, reg_coef=reg_coef, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block(x, filters, conv_shortcut=False, reg_coef=reg_coef, name=name + '_block' + str(i))
    return x


def ResNet50(strides, include_weights, weights_folder, reg_coef=1e-5, **kwargs):
    """Returns the instantiated ResNet backbone model.

    Args:
        strides (list): stride coefficients used in specific layers.
        include_weights (bool): if True, load the ImageNet weights
            to the model.
        weights_folder (str): root path to weight folder.
        reg_coef (float): backbone regularization coefficients.
    """
    def stack_fn(x):
        C2 = stack(x, 64, 3, stride=1, reg_coef=reg_coef, name='conv2')
        C3 = stack(C2, 128, 4, stride=strides[0], reg_coef=reg_coef, name='conv3')
        C4 = stack(C3, 256, 6, stride=strides[1], reg_coef=reg_coef, name='conv4')
        C5 = stack(C4, 512, 3, stride=strides[2], reg_coef=reg_coef, name='conv5')
        return C2, C3, C4, C5
    return ResNet(stack_fn, False, True, weights_folder, 'resnet50', include_weights, reg_coef, **kwargs)


class Model:
    """Module of the instantiated backbone network.

    Attributes:
        model (object): the backbone model to train.
        last_conv_channel_dim: (int): output channels of the
            last convolutional layer.
    """

    def __init__(self, trainable, add_initial_weights, layers_to_change,
                 reg_coef, weights_folder, **kwargs):
        # Define strides
        two_coef = max(0, 3 - layers_to_change)
        one_coef = min(3, layers_to_change)
        strides = ([2] * two_coef) + ([1] * one_coef)

        # Build model
        self.model = ResNet50(
            strides=strides,
            include_weights=add_initial_weights,
            reg_coef=reg_coef,
            weights_folder=weights_folder, **kwargs)

        # Freeze layer weights (default: not trainable)
        self.model.trainable = trainable
        self.last_conv_channel_dim = 2048

    def __call__(self, inputs, *args, **kwargs):
        x = self.model(inputs)
        # Rule out network output from gradient calculation
        # if network not trained.
        if not self.model.trainable:
            x = tf.stop_gradient(x)
        return x
