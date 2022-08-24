import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, Activation
from tensorflow.keras.activations import swish
from tensorflow.keras.utils import get_custom_objects
import numpy as np


def myswish_beta(x):
    beta = tf.Variable(initial_value=1.0, trainable=True, name='swish_beta')
    return x * tf.nn.sigmoid(beta * x)  # trainable parameter beta


# get_custom_objects().update({'swish': Activation(myswish_beta)})


def print_node(x):
    print(x)
    return x


def build_lm_preproc_model(local_map_in, name=''):
    local_map_1 = tf.keras.layers.Conv2D(4, 3, activation=None,
                                         strides=(1, 1),
                                         name=name + 'local_conv_0' + str(0 + 1))(
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

    model = tf.keras.Model(inputs=[local_map_in], outputs=[flatten_local, local_map_2, local_map_3, local_map_4])

    return model


def build_flat_model_masked(states_in, num_actions, initial_mb, path_to_local_pretrained_weights=None,
                            name=''):  # local:17,17,4; global:21:21,4
    '''
     usage: model = build_hl_model(lm[tf.newaxis, ...], gm[tf.newaxis, ...], states_proc[tf.newaxis, ...])
    '''

    local_map_in, global_map_in, states_proc_in = states_in
    # local_map_in_sg = tf.stop_gradient(local_map_in)
    # global_map_in_sg = tf.stop_gradient(global_map_in)
    # states_proc_in_sg = tf.stop_gradient(states_proc_in)

    # states_proc = states_proc_in / initial_mb + 1e-6
    states_proc = states_proc_in / 100

    local_map_model = build_lm_preproc_model(local_map_in, name)
    # if path_to_local_pretrained_weights:
    #     print(f'Loading weights from: {path_to_local_pretrained_weights}')
    #     local_map_model.load_weights(path_to_local_pretrained_weights)
    flatten_local, local_map_2, local_map_3, local_map_4 = local_map_model.output

    # global map processing layers

    global_map_1 = tf.keras.layers.Conv2D(4, 5, activation=None,
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(0 + 1))(global_map_in)  # out:17
    norm = tf.keras.layers.LayerNormalization()(global_map_1)
    norm = swish(norm)
    global_map_2 = tf.keras.layers.Conv2D(8, 5, activation=None,
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(1 + 1))(norm)  # out:13
    norm = tf.keras.layers.LayerNormalization()(global_map_2)
    norm = swish(norm)
    global_map_3 = tf.keras.layers.Conv2D(16, 5, activation=None,
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(2 + 1))(norm)  # out:9
    norm = tf.keras.layers.LayerNormalization()(global_map_3)
    norm = swish(norm)

    flatten_global = tf.keras.layers.Flatten(name=name + 'global_flatten')(norm)

    flatten_map = tf.keras.layers.Concatenate(name=name + 'concat_flatten')(
        [flatten_global, flatten_local, states_proc])

    # layer = tf.keras.layers.Concatenate(name=name + 'concat')([flatten_map, states_proc_in])

    layer_1 = tf.keras.layers.Dense(256, activation=None, name=name + 'hidden_layer_all_hl_' + str(0))(
        flatten_map)
    norm = tf.keras.layers.LayerNormalization()(layer_1)
    norm = swish(norm)
    layer_2 = tf.keras.layers.Dense(256, activation=None, name=name + 'hidden_layer_all_hl_' + str(1))(
        norm)
    norm = tf.keras.layers.LayerNormalization()(layer_2)
    norm = swish(norm)
    # layer_3 = tf.keras.layers.Dense(256, activation='elu', name=name + 'hidden_layer_all_hl_' + str(2))(
    #     layer_1)

    Q_vals = tf.keras.layers.Dense(units=num_actions, activation=None, name=name + 'last_dense_layer_hl')(
        norm)

    # Masking of invalid actions

    lz = 1-local_map_in[:, 8, 8, 2] # negation of landing zone -> if no landing zone on position mask to -inf
    b = tf.cast(tf.stack([local_map_in[:, 9, 8, 0],
                          local_map_in[:, 8, 9, 0],
                          local_map_in[:, 7, 8, 0],
                          local_map_in[:, 8, 7, 0],
                         lz], axis=-1), tf.bool)

    # b = tf.cast((tf.keras.layers.Concatenate(axis=1)([a_1, c])), dtype=tf.bool)

    Q_vals = tf.where(b, -np.inf, Q_vals)

    print(f'Qvals: {Q_vals}')

    model = tf.keras.Model(inputs=[local_map_in, global_map_in, states_proc_in], outputs=Q_vals)
    return model


def build_smaller_flat_model_masked(states_in, num_actions, initial_mb, path_to_local_pretrained_weights=None,
                            name=''):  # local:17,17,4; global:21:21,4
    '''
     usage: model = build_hl_model(lm[tf.newaxis, ...], gm[tf.newaxis, ...], states_proc[tf.newaxis, ...])
    '''

    local_map_in, global_map_in, states_proc_in = states_in
    # local_map_in_sg = tf.stop_gradient(local_map_in)
    # global_map_in_sg = tf.stop_gradient(global_map_in)
    # states_proc_in_sg = tf.stop_gradient(states_proc_in)

    # states_proc = states_proc_in / initial_mb + 1e-6
    states_proc = states_proc_in / 100

    # local_map_model = build_lm_preproc_model(local_map_in, name)
    # if path_to_local_pretrained_weights:
    #     print(f'Loading weights from: {path_to_local_pretrained_weights}')
    #     local_map_model.load_weights(path_to_local_pretrained_weights)
    # flatten_local, local_map_2, local_map_3, local_map_4 = local_map_model.output

    local_map = Conv2D(16, 5, activation='relu',
                       strides=(1, 1),
                       name=name + 'local_conv_' + str(0 + 1))(local_map_in)
    local_map = Conv2D(16, 5, activation='relu',
                       strides=(1, 1),
                       name=name + 'local_conv_' + str(1 + 1))(local_map)

    flatten_local = Flatten(name=name + 'local_flatten')(local_map)

    # for k in range(2):
    global_map = Conv2D(16, 5, activation='relu',
                       strides=(1, 1),
                       name=name + 'local_conv_' + str(0 + 1))(global_map_in)

    global_map = Conv2D(16, 5, activation='relu',
                       strides=(1, 1),
                       name=name + 'local_conv_' + str(1 + 1))(global_map)

    flatten_global = Flatten(name=name + 'local_flatten')(global_map)


    flatten_map = tf.keras.layers.Concatenate(name=name + 'concat_flatten')(
        [flatten_global, flatten_local, states_proc])

    layer = Dense(256, activation='relu', name=name + 'hidden_layer_all_' + str(0))(
        flatten_map)
    layer = Dense(256, activation='relu', name=name + 'hidden_layer_all_' + str(0))(
        layer)
    layer = Dense(256, activation='relu', name=name + 'hidden_layer_all_' + str(0))(
        layer)


    Q_vals = tf.keras.layers.Dense(units=num_actions, activation=None, name=name + 'last_dense_layer_hl')(
        layer)

    # Masking of invalid actions

    lz = 1-local_map_in[:, 8, 8, 2]  # negation of landing zone -> if no landing zone on position mask to -inf
    b = tf.cast(tf.stack([local_map_in[:, 9, 8, 0],
                          local_map_in[:, 8, 9, 0],
                          local_map_in[:, 7, 8, 0],
                          local_map_in[:, 8, 7, 0],
                         lz], axis=-1), tf.bool)

    # b = tf.cast((tf.keras.layers.Concatenate(axis=1)([a_1, c])), dtype=tf.bool)

    Q_vals = tf.where(b, -np.inf, Q_vals)

    print(f'Qvals: {Q_vals}')

    model = tf.keras.Model(inputs=[local_map_in, global_map_in, states_proc_in], outputs=Q_vals)
    return model

def build_flat_model_no_mask(states_in, num_actions, initial_mb, path_to_local_pretrained_weights=None,
                             name=''):  # local:17,17,4; global:21:21,4
    '''
     usage: model = build_hl_model(lm[tf.newaxis, ...], gm[tf.newaxis, ...], states_proc[tf.newaxis, ...])
    '''

    local_map_in, global_map_in, states_proc_in = states_in
    # local_map_in_sg = tf.stop_gradient(local_map_in)
    # global_map_in_sg = tf.stop_gradient(global_map_in)
    # states_proc_in_sg = tf.stop_gradient(states_proc_in)

    # states_proc = states_proc_in / initial_mb + 1e-6
    states_proc = states_proc_in / 100

    local_map_model = build_lm_preproc_model(local_map_in, name)
    # if path_to_local_pretrained_weights:
    #     print(f'Loading weights from: {path_to_local_pretrained_weights}')
    #     local_map_model.load_weights(path_to_local_pretrained_weights)
    flatten_local, local_map_2, local_map_3, local_map_4 = local_map_model.output

    # global map processing layers

    global_map_1 = tf.keras.layers.Conv2D(4, 5, activation=None,
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(0 + 1))(global_map_in)  # out:17
    norm = tf.keras.layers.LayerNormalization()(global_map_1)
    norm = swish(norm)
    global_map_2 = tf.keras.layers.Conv2D(8, 5, activation=None,
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(1 + 1))(norm)  # out:13
    norm = tf.keras.layers.LayerNormalization()(global_map_2)
    norm = swish(norm)
    global_map_3 = tf.keras.layers.Conv2D(16, 5, activation=None,
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(2 + 1))(norm)  # out:9
    norm = tf.keras.layers.LayerNormalization()(global_map_3)
    norm = swish(norm)

    flatten_global = tf.keras.layers.Flatten(name=name + 'global_flatten')(norm)

    flatten_map = tf.keras.layers.Concatenate(name=name + 'concat_flatten')(
        [flatten_global, flatten_local, states_proc])

    # layer = tf.keras.layers.Concatenate(name=name + 'concat')([flatten_map, states_proc_in])

    layer_1 = tf.keras.layers.Dense(256, activation=None, name=name + 'hidden_layer_all_hl_' + str(0))(
        flatten_map)
    norm = tf.keras.layers.LayerNormalization()(layer_1)
    norm = swish(norm)
    layer_2 = tf.keras.layers.Dense(256, activation=None, name=name + 'hidden_layer_all_hl_' + str(1))(
        norm)
    norm = tf.keras.layers.LayerNormalization()(layer_2)
    norm = swish(norm)
    # layer_3 = tf.keras.layers.Dense(256, activation='elu', name=name + 'hidden_layer_all_hl_' + str(2))(
    #     layer_1)

    Q_vals = tf.keras.layers.Dense(units=num_actions, activation=None, name=name + 'last_dense_layer_hl')(
        norm)

    # tf.debugging.assert_all_finite(Q_vals, message='Nan in Q_vals')

    print(f'Qvals: {Q_vals}')

    model = tf.keras.Model(inputs=[local_map_in, global_map_in, states_proc_in], outputs=Q_vals)
    return model
