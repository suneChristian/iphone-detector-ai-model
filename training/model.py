""" model.py
    Defines the model's structure and configuration.
"""
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def _residual_block(x, n_filters, strides):
    """ Produces a residual convolutional block as seen in
        https://en.wikipedia.org/wiki/Residual_neural_network

    Args:
        x: The input tensor
        n_filters (int): The number of filters for the convolutional layers
        strides (int): The downsampling convolutional layers' strides along
                       height and width

    Returns:
        x: The output tensor
    """
    shortcut = x

    x = Conv2D(n_filters, 3, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(n_filters, 3,  padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(n_filters, 1, strides=strides, padding='same', use_bias=False)(shortcut)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)

    x = Add()([shortcut, x])
    return x


def create_model(n_blocks=4, n_filters=16, input_shape=(480, 270, 3)):
    """ Defines and instantiates a model.

    Returns:
        model: The instantiated but uncompiled model.
    """
    img_in = Input(shape=input_shape, name='image_input')

    x = img_in
    for block_index in range(n_blocks):
        x = _residual_block(x, n_filters * 2 ** block_index, strides=1)
        x = _residual_block(x, n_filters * 2 ** (block_index + 1), strides=2)

    side_h = K.int_shape(x)[-2]
    side_w = K.int_shape(x)[-3]
    x = MaxPooling2D((side_w, side_h))(x)
    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)
    phone_pred = Dense(1, activation='sigmoid', name='A_phone_pred')(x)

    return Model(img_in, [phone_pred], name='phone_indicator')
