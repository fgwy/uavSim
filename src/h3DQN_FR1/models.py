import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, Activation
from tensorflow.keras.utils import get_custom_objects
import numpy as np

def myswish_beta(x):
   beta = tf.Variable(initial_value=1.0, trainable=True, name='swish_beta')
   return x*tf.nn.sigmoid(beta*x) #trainable parameter beta

# get_custom_objects().update({'swish': Activation(myswish_beta)})


def print_node(x):
    print(x)
    return x

def build_lm_preproc_model(local_map_in, name=''):
    local_map_1 = tf.keras.layers.Conv2D(4, 3, activation='swish',
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(0 + 1))(
        local_map_in)  # out:(None, 1, 15, 15, 4) 1156->
    norm = tf.keras.layers.BatchNormalization()(local_map_1)
    local_map_2 = tf.keras.layers.Conv2D(8, 3, activation='swish',
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(1 + 1))(
        norm)  # out:(None, 1, 13, 13, 8)
    norm = tf.keras.layers.BatchNormalization()(local_map_2)
    local_map_3 = tf.keras.layers.Conv2D(16, 3, activation='swish',
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(2 + 1))(
        norm)  # out:(None, 1, 11, 11, 16)
    norm = tf.keras.layers.BatchNormalization()(local_map_3)
    local_map_4 = tf.keras.layers.Conv2D(16, 3, activation='swish',
                                         strides=(1, 1),
                                         name=name + 'local_conv_' + str(3 + 1))(
        norm)  # out:(None, 1, 9, 9, 16)
    flatten_local = tf.keras.layers.Flatten(name=name + 'local_flatten')(local_map_4)

    model = tf.keras.Model(inputs=[local_map_in], outputs=[flatten_local, local_map_2, local_map_3, local_map_4])

    return model

def build_dummy_model(states_in, num_actions, initial_mb):
    local_map_in, global_map_in, states_proc_in = states_in
    states_proc = states_proc_in / initial_mb + 1e-6

    local_map_proc = Flatten()(local_map_in)
    local_map_proc = Dense(128)(local_map_proc)

    global_map_proc = Flatten()(global_map_in)
    global_map_proc = Dense(128)(global_map_proc)

    concat = Concatenate()([local_map_proc, global_map_proc, states_proc])
    out = Dense(num_actions, activation='linear')(concat)
    model = Model(inputs=states_in, outputs=out)
    return model

def build_hl_model(states_in, use_skip, initial_mb, path_to_local_pretrained_weights=None, name=''):  # local:17,17,4; global:21:21,4
    '''
     usage: model = build_hl_model(lm[tf.newaxis, ...], gm[tf.newaxis, ...], states_proc[tf.newaxis, ...])
    '''

    local_map_in, global_map_in, states_proc_in = states_in
    # local_map_in_sg = tf.stop_gradient(local_map_in)
    # global_map_in_sg = tf.stop_gradient(global_map_in)
    # states_proc_in_sg = tf.stop_gradient(states_proc_in)

    states_proc = states_proc_in / initial_mb + 1e-6

    local_map_model = build_lm_preproc_model(local_map_in, name)
    if not (path_to_local_pretrained_weights == None or path_to_local_pretrained_weights == ''):
        local_map_model.load_weights(path_to_local_pretrained_weights)
    flatten_local, local_map_2, local_map_3, local_map_4 = local_map_model(local_map_in)

    # global map processing layers

    global_map_1 = tf.keras.layers.Conv2D(4, 5, activation='swish',
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(0 + 1))(global_map_in)  # out:17
    norm = tf.keras.layers.BatchNormalization()(global_map_1)
    global_map_2 = tf.keras.layers.Conv2D(8, 5, activation='swish',
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(1 + 1))(norm)  # out:13
    norm = tf.keras.layers.BatchNormalization()(global_map_2)
    global_map_3 = tf.keras.layers.Conv2D(16, 5, activation='swish',
                                          strides=(1, 1),
                                          name=name + 'global_conv_' + str(2 + 1))(norm)  # out:9
    norm = tf.keras.layers.BatchNormalization()(global_map_3)

    flatten_global = tf.keras.layers.Flatten(name=name + 'global_flatten')(norm)

    flatten_map = tf.keras.layers.Concatenate(name=name + 'concat_flatten')(
        [flatten_global, flatten_local, states_proc])

    # layer = tf.keras.layers.Concatenate(name=name + 'concat')([flatten_map, states_proc_in])

    layer_1 = tf.keras.layers.Dense(256, activation='swish', name=name + 'hidden_layer_all_hl_' + str(0))(
        flatten_map)
    norm = tf.keras.layers.BatchNormalization()(layer_1)
    layer_2 = tf.keras.layers.Dense(256, activation='swish', name=name + 'hidden_layer_all_hl_' + str(1))(
        norm)
    norm = tf.keras.layers.BatchNormalization()(layer_2)
    # layer_3 = tf.keras.layers.Dense(256, activation='elu', name=name + 'hidden_layer_all_hl_' + str(2))(
    #     layer_1)

    output = tf.keras.layers.Dense(units=300, activation='swish', name=name + 'last_dense_layer_hl')(
        norm)
    norm_out = tf.keras.layers.BatchNormalization()(output)

    # value for dueling ddqn
    val = tf.keras.layers.Dense(units=256, activation='swish', name=name + 'value_dense1')(norm)
    val = tf.keras.layers.BatchNormalization()(val)
    value = tf.keras.layers.Dense(units=1, activation='linear', name=name + 'value_out')(val)

    reshape = tf.keras.layers.Reshape((5, 5, 12), name=name + 'last_dense_layer')(norm_out)

    # landing = tf.keras.layers.Dense(units=128, activation='swish', name=name + 'landing_layer_proc_hl')(
    #     layer_1)
    landing = tf.keras.layers.Dense(units=1, activation='linear', name=name + 'landing_layer_hl')(norm)

    # deconvolutional part aiming at 17x17
    # self.dec_model = self.build_goal_decoder(reshape, local_map_2, local_map_3, local_map_4, name=name)
    # deconv_4 = self.dec_model(reshape, local_map_2, local_map_3, local_map_4)

    if use_skip:
        deconv_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, activation='swish',
                                                   name=name + 'deconv_' + str(1))(reshape)
        skip_1 = tf.keras.layers.Concatenate(name=name + '1st_skip_connection_concat', axis=3)(
            [deconv_1, local_map_4])
        norm = tf.keras.layers.BatchNormalization()(skip_1)

        deconv_2 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='swish',
                                                   name=name + 'deconv_' + str(2))(norm)
        skip_2 = tf.keras.layers.Concatenate(name=name + '2nd_skip_connection_concat', axis=3)(
            [deconv_2, local_map_3])
        norm = tf.keras.layers.BatchNormalization()(skip_2)
        deconv_2_1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='swish',
                                                     name=name + 'deconv_' + str(2.1))(norm)
        skip_3 = tf.keras.layers.Concatenate(name=name + '3rd_skip_connection_concat', axis=3)(
            [deconv_2_1, local_map_2])
        norm = tf.keras.layers.BatchNormalization()(skip_3)
        deconv_3 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, activation='swish',
                                                   name=name + 'deconv_' + str(3))(norm)
        deconv_4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=1, activation='linear',
                                                   name=name + 'deconv_' + str(4))(deconv_3)

    else:
        deconv_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, activation='swish',
                                                   name=name + 'deconv_' + str(1))(reshape)
        norm = tf.keras.layers.BatchNormalization()(deconv_1)
        deconv_2 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='swish',
                                                   name=name + 'deconv_' + str(2))(norm)
        norm = tf.keras.layers.BatchNormalization()(deconv_2)
        deconv_2_1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='swish',
                                                     name=name + 'deconv_' + str(2.1))(norm)
        norm = tf.keras.layers.BatchNormalization()(deconv_2_1)
        deconv_3 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, activation='swish',
                                                   name=name + 'deconv_' + str(3))(norm)
        norm = tf.keras.layers.BatchNormalization()(deconv_3)
        deconv_4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=1, activation='linear',
                                                   name=name + 'deconv_' + str(4), dtype=tf.float64)(norm)

    flatten_deconv = tf.keras.layers.Flatten(name=name + 'deconv_flatten')(deconv_4)
    adv = tf.keras.layers.Concatenate(name=name + 'concat_final')([flatten_deconv, landing])
    # adv = tf.keras.layers.BatchNormalization(name=name + 'final_norm')(adv)
    advAverage = tf.reduce_mean(adv, axis=1, keepdims=True)

    Q_vals = value + tf.subtract(adv, advAverage)

    model = tf.keras.Model(inputs=[local_map_in, global_map_in, states_proc_in], outputs=Q_vals)
    return model

# def build_goal_decoder(self, reshape, local_map_2, local_map_3, local_map_4, name=''):
#     if self.params.use_skip:
#         deconv_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, activation='swish',
#                                                    name=name + 'deconv_' + str(1))(reshape)
#         skip_1 = tf.keras.layers.Concatenate(name=name + '1st_skip_connection_concat', axis=3)(
#             [deconv_1, local_map_4])
#         norm = tf.keras.layers.BatchNormalization()(skip_1)
#         deconv_2 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='swish',
#                                                    name=name + 'deconv_' + str(2))(norm)
#         skip_2 = tf.keras.layers.Concatenate(name=name + '2nd_skip_connection_concat', axis=3)(
#             [deconv_2, local_map_3])
#         norm = tf.keras.layers.BatchNormalization()(skip_2)
#         deconv_2_1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='swish',
#                                                      name=name + 'deconv_' + str(2.1))(norm)
#         skip_3 = tf.keras.layers.Concatenate(name=name + '3rd_skip_connection_concat', axis=3)(
#             [deconv_2_1, local_map_2])
#         norm = tf.keras.layers.BatchNormalization()(skip_3)
#         deconv_3 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, activation='swish',
#                                                    name=name + 'deconv_' + str(3))(norm)
#         deconv_4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=1, activation='linear',
#                                                    name=name + 'deconv_' + str(4))(deconv_3)
#
#     else:
#         deconv_1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, activation='swish',
#                                                    name=name + 'deconv_' + str(1))(reshape)
#         norm = tf.keras.layers.BatchNormalization()(deconv_1)
#         deconv_2 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, activation='swish',
#                                                    name=name + 'deconv_' + str(2))(norm)
#         norm = tf.keras.layers.BatchNormalization()(deconv_2)
#         deconv_2_1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, activation='swish',
#                                                      name=name + 'deconv_' + str(2.1))(norm)
#         norm = tf.keras.layers.BatchNormalization()(deconv_2_1)
#         deconv_3 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, activation='swish',
#                                                    name=name + 'deconv_' + str(3))(norm)
#         norm = tf.keras.layers.BatchNormalization()(deconv_3)
#         deconv_4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=1, activation='linear',
#                                                    name=name + 'deconv_' + str(4), dtype=tf.float64)(norm)
#
#     model = Model(inputs=(reshape, local_map_2, local_map_3, local_map_4), outputs=deconv_4)
#     return model
#
# class MetaControllerParams:
#     def __init__(self):
#         # Convolutional part config
#         self.conv_layers = 2
#         self.conv_kernel_size = 5
#         self.conv_kernels = 16
#
#         # Fully Connected config
#         self.hidden_layer_size = 256
#         self.hidden_layer_num = 3
#
#         # Training Params
#         self.learning_rate = 3e-5
#         self.alpha = 0.005
#         self.gamma = 0.95
#
#         # Exploration strategy
#         self.soft_max_scaling = 0.1
#
#     def build_model(self):
#         pass
#         return 0
#
#
# class ControllerParams:
#     def __init__(self):
#         # Convolutional part config
#         self.conv_layers = 2
#         self.conv_kernel_size = 5
#         self.conv_kernels = 16
#
#         # Fully Connected config
#         self.hidden_layer_size = 256
#         self.hidden_layer_num = 3
#
#         # Training Params
#         self.learning_rate = 3e-5
#         self.alpha = 0.005
#         self.gamma = 0.95
#
#         # Exploration strategy
#         self.soft_max_scaling = 0.1
#
#
# class MetaController(object):
#     def __init__(self, params=MetaControllerParams()):
#         self.params = params
#
# class Controller(object):
#     def __init__(self, params=ControllerParams()):
#         self.params = params
#
#     def build_model(self):
#         pass
#         return 0


