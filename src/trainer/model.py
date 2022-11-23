import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, Activation
from tensorflow.keras.activations import swish
from tensorflow.keras.utils import get_custom_objects
import numpy as np


def myswish_beta(x):
    beta = tf.Variable(initial_value=1.0, trainable=True, name='swish_beta')
    return x * tf.nn.sigmoid(beta * x)  # trainable parameter beta


def build_lm_preproc_model(local_map_in, name=''):
    local_map_1 = tf.keras.layers.Conv2D(4, 3, activation=None,
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(0 + 1))(
        local_map_in)  # out:(None, 1, 15, 15, 4) 1156->
    norm = tf.keras.layers.LayerNormalization()(local_map_1)
    activation = swish(norm)
    local_map_2 = tf.keras.layers.Conv2D(8, 3, activation=None,
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(1 + 1))(
        activation)  # out:(None, 1, 13, 13, 8)
    norm = tf.keras.layers.LayerNormalization()(local_map_2)
    activation = swish(norm)
    local_map_3 = tf.keras.layers.Conv2D(16, 3, activation=None,
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(2 + 1))(
        activation)  # out:(None, 1, 11, 11, 16)
    norm = tf.keras.layers.LayerNormalization()(local_map_3)
    activation = swish(norm)
    local_map_4 = tf.keras.layers.Conv2D(16, 3, activation=None,
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(3 + 1))(
        activation)  # out:(None, 1, 9, 9, 16)
    norm = tf.keras.layers.LayerNormalization()(local_map_4)
    activation = swish(norm)
    flatten_local = tf.keras.layers.Flatten(name=name + 'local_flatten')(activation)

    model = tf.keras.Model(inputs=[local_map_in], outputs=[flatten_local, local_map_1, local_map_2, local_map_3, local_map_4])

    return model