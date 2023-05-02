import os
import tensorflow as tf

from tensorflow.keras import layers, models, backend


def DenseNet(num_blocks, strides, include_weights, reg_coef, weights_folder,
             **kwargs):
    """Instantiates the DenseNet architecture.

    Args:
        num_blocks (list): number of convolutional blocks per
            denseblock created.
        strides (list): stride coefficients used in specific layers.
        include_weights (bool): if True, load the ImageNet weights
            to the model.
        reg_coef (float): backbone regularization coefficients.
        weights_folder (str): root path to weight folder.

    Returns
        A Keras model instance..
    """
    img_input = layers.Input(shape=(None, None, 3))

    x = layers.Conv2D(
        64, 7,
        strides=2,
        use_bias=False,
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
        name='conv1/conv')(img_input)
    x = layers.BatchNormalization(
        axis=-1, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)

    C2 = layers.MaxPooling2D(3, strides=2, padding="same", name='pool1')(x)

    x = dense_block(C2, num_blocks[0], reg_coef, name='conv2')
    C3 = transition_block(x, 0.5, strides[0], reg_coef, name='pool2')
    x = dense_block(C3, num_blocks[1], reg_coef, name='conv3')
    C4 = transition_block(x, 0.5, strides[1], reg_coef, name='pool3')
    x = dense_block(C4, num_blocks[2], reg_coef, name='conv4')
    x = transition_block(x, 0.5, strides[2], reg_coef, name='pool4')
    x = dense_block(x, num_blocks[3], reg_coef, name='conv5')

    x = layers.BatchNormalization(
        axis=-1, epsilon=1.001e-5, name='bn')(x)
    C5 = layers.Activation('relu', name='relu')(x)

    model = models.Model(img_input, C5, name='densenet201')

    # Load weights.
    if include_weights:
        weights_path = os.path.join(
            weights_folder,
            'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
        model.load_weights(weights_path, by_name=True)

    return model


def dense_block(x, blocks, reg_coef, name):
    """A dense block.

   Args:
        x (tensor): input tensor.
        blocks (int): the number of building blocks.
        reg_coef (float): the regularization coefficients.
        name (str): the block label.

    Returns:
        Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, reg_coef, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, stride, reg_coef, name):
    """A transition block.

    Args:
        x (tensor): input tensor.
        reduction (float): compression rate at transition layers.
        stride (int): the stride coefficient.
        reg_coef (float): the regularization coefficients.
        name (str): the block label.

    Returns:
        Output tensor for the block.
    """
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[-1] * reduction), 1,
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=stride, padding="same", name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, reg_coef, name):
    """A building block for a dense block.

    Args:
        x (tensor): input tensor.
        growth_rate (float): growth rate at dense layers.
        reg_coef (float): the regularization coefficients.
        name (str): the block label.

    Returns:
        Output tensor for the block.
    """
    x1 = layers.BatchNormalization(axis=-1,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       padding="same",
                       use_bias=False,
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=reg_coef, l2=reg_coef),
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
    return x


def DenseNet201(strides, include_weights, reg_coef, weights_folder, **kwargs):
    """Returns the instantiated ResNet backbone model.

    Args:
        strides (list): stride coefficients used in specific layers.
        include_weights (bool): if True, load the ImageNet weights
            to the model.
        reg_coef (float): backbone regularization coefficients.
        weights_folder (str): root path to weight folder.
    """
    return DenseNet([6, 12, 48, 32], strides, include_weights, reg_coef, weights_folder, **kwargs)


class Model:
    """Module of the instantiated backbone network.

    Attributes:
        model (object): the backbone model to train.
        last_conv_channel_dim: (int): output channels of the
            last convolutional layer.
    """

    def __init__(self, trainable, add_initial_weights, reg_coef, weights_folder,
                 layers_to_change=2, **kwargs):
        # Define strides
        two_coef = max(0, 3 - layers_to_change)
        one_coef = min(3, layers_to_change)
        strides = ([2] * two_coef) + ([1] * one_coef)

        # Build model
        self.model = DenseNet201(
            strides=strides,
            include_weights=add_initial_weights,
            reg_coef=reg_coef,
            weights_folder=weights_folder, **kwargs)

        # Freeze layer weights (default: not trainable)
        self.model.trainable = trainable
        self.last_conv_channel_dim = 1920

    def __call__(self, inputs, *args, **kwargs):
        x = self.model(inputs)
        # Rule out network output from gradient calculation
        # if network not trained.
        if not self.model.trainable:
            x = tf.stop_gradient(x)
        return x
