import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, Reshape, Activation
import numpy as np
# from tensorflow.keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.utils import get_custom_objects


def print_node(x):
    print(x)
    return x

# def myswish_beta(x):
#    beta = tf.Variable(initial_value=1.0, trainable=True, name='swish_beta')
#    return x*tf.nn.sigmoid(beta*x) #trainable parameter beta

# get_custom_objects().update({'swish': Activation(myswish_beta)})


def identify_idx_highest_val_in_tensor(tensor):
    return tf.math.argmax(tensor)

def check_is_nan_in_here(x, name):
    try:
        for i in x:
            if np.any(tf.math.is_nan(i).numpy()):
                print(f'nan in {name}: {i}')
    except:
        try:
            if np.any(tf.math.is_nan(x).numpy()):
                print(f'nan in {name}: {x}')
        except AssertionError:
            print(f'Couldnt iterate over {x}')


class HL_DDQNAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 3
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Hierarchical Params

        # Training Params
        self.learning_rate = 3e-5
        self.alpha = 0.005
        self.gamma = 0.95

        # Exploration strategy
        self.soft_max_scaling = 0.1

        # Global-Local Map
        self.use_global_local = True
        self.global_map_scaling = 3
        self.local_map_size = 17

        # Printing
        self.print_summary = False

        # Model building
        self.use_skip = True
        self.use_pretrained_local_map_preproc = False
        self.path_to_local_pretrained_weights = ''

        # epsilon
        self.initial_epsilon = 0.3
        self.final_epsilon = 0.0005
        self.eps_steps = 20000

class HL_DDQNAgent(object):

    def __init__(self, params: HL_DDQNAgentParams, example_state, example_action_hl, stats=None):

        self.params = params
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0

        self.training = False

        self.i = 0

        self.var = True

        self.epsilon = self.params.initial_epsilon

        self.counter = 0

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.float_map_shape = example_state.get_float_map_shape()
        self.scalars = example_state.get_num_scalars()
        self.goal_target_shape = example_state.get_goal_target_shape()
        self.num_actions_hl = (self.params.local_map_size ** 2) + 1
        self.example_goal = example_state.get_example_goal()
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]
        self.local_map_shape = example_state.get_local_map_shape()
        self.global_map_shape = example_state.get_global_map_shape(self.params.global_map_scaling)
        self.initial_mb = tf.convert_to_tensor(example_state.get_initial_mb(), dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_hl_input', dtype=tf.float32)
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_hl_input = Input(shape=(), name='reward_hl_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_prime_hl_input = Input(shape=(), name='q_prime_hl_input', dtype=tf.float32)

        local_map_input = Input(shape=self.local_map_shape, name='local_map_input')
        global_map_input = Input(shape=self.global_map_shape, name='global_map_input')

        states_hl = [local_map_input,
                     global_map_input,
                     scalars_input]

        self.q_network_hl = self.build_hl_model(states_hl, self.params.path_to_local_pretrained_weights, 'soft_updated_hl_model_')
        # self.q_network_hl.summary()
        self.target_network_hl = self.build_hl_model(states_hl, 'target_hl_')

        # self.q_network_hl = self.build_dummy_model(states_hl)
        # self.target_network_hl = self.build_dummy_model(states_hl)

        self.hard_update_hl()

        q_values_hl = self.q_network_hl.output
        q_target_values_hl = self.target_network_hl.output

        ########## HIGH-Level Agent ##############

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action_hl = tf.argmax(q_values_hl, axis=1, name='max_action', output_type=tf.int64)
        max_action_target_hl = tf.argmax(q_target_values_hl, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action_hl = tf.one_hot(max_action_hl, depth=self.num_actions_hl, dtype=float, on_value=1.0,
                                           off_value=0.0)
        # one_hot_max_action_hl = tf.squeeze(one_hot_max_action_hl)
        q_prime_hl = tf.reduce_sum(tf.multiply(one_hot_max_action_hl, q_target_values_hl, name='mul_hot_target'), axis=1,
                                  name='q_prime_hl')
        self.q_prime_model_hl = Model(inputs=states_hl, outputs=q_prime_hl)

        # Define Bellman loss
        one_hot_rm_action_hl = tf.one_hot(action_input, depth=self.num_actions_hl, on_value=1.0, off_value=0.0,
                                          dtype=float)
        one_cold_rm_action_hl = tf.one_hot(action_input, depth=self.num_actions_hl, on_value=0.0, off_value=1.0,
                                           dtype=float)
        q_old_hl = tf.stop_gradient(tf.multiply(q_values_hl, one_cold_rm_action_hl))
        gamma_terminated_hl = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update_hl = tf.expand_dims(tf.add(reward_hl_input, tf.multiply(q_prime_hl_input, gamma_terminated_hl)), 1)
        q_update_hot_hl = tf.multiply(q_update_hl, one_hot_rm_action_hl)
        q_new_hl = tf.add(q_update_hot_hl, q_old_hl)
        q_loss_hl = tf.losses.MeanSquaredError()(q_new_hl, q_values_hl)
        self.q_loss_model_hl = Model(
            inputs=[local_map_input, global_map_input, scalars_input, action_input, reward_hl_input,
                    termination_input, q_prime_hl_input],
            outputs=[q_loss_hl, q_new_hl, q_update_hot_hl, q_old_hl])

        # Exploit act model
        self.exploit_model_hl = Model(inputs=states_hl, outputs=(max_action_hl, q_values_hl))
        self.exploit_model_target_hl = Model(inputs=states_hl, outputs=max_action_target_hl)

        # Softmax explore model
        tf.debugging.assert_all_finite(self.params.soft_max_scaling, message='Nan in softmax_scaling_factor')
        softmax_scaling_hl = tf.divide(q_values_hl, tf.constant(self.params.soft_max_scaling, dtype=float))
        tf.debugging.assert_all_finite(softmax_scaling_hl, message='Nan in softmax_scaling')
        softmax_action_hl = tf.math.softmax(softmax_scaling_hl, name='softmax_action')
        tf.debugging.assert_all_finite(softmax_action_hl, message='Nan in softmax_action')
        self.soft_explore_model_hl = Model(inputs=states_hl, outputs=(softmax_action_hl, q_values_hl, max_action_hl))

        self.q_optimizer_hl = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model_hl.summary()

        if stats:
            stats.set_model(self.target_network_hl)


    def build_dummy_model(self, states_in):
        local_map_in, global_map_in, states_proc_in = states_in
        states_proc = states_proc_in / self.initial_mb + 1e-6

        local_map_proc = Flatten()(local_map_in)
        local_map_proc = Dense(128)(local_map_proc)

        global_map_proc = Flatten()(global_map_in)
        global_map_proc = Dense(128)(global_map_proc)

        concat = Concatenate()([local_map_proc, global_map_proc, states_proc])
        out = Dense(self.num_actions_hl, activation='linear')(concat)
        model = Model(inputs=states_in, outputs=out)
        return model




    def build_hl_model(self, states_in, path_to_local_pretrained_weights, name=''):  # local:17,17,4; global:21:21,4
        '''
         usage: model = build_hl_model(lm[tf.newaxis, ...], gm[tf.newaxis, ...], states_proc[tf.newaxis, ...])
        '''

        local_map_in, global_map_in, states_proc_in = states_in
        local_map_in_sg = tf.stop_gradient(local_map_in)
        global_map_in_sg = tf.stop_gradient(global_map_in)
        states_proc_in_sg = tf.stop_gradient(states_proc_in)

        states_proc = states_proc_in_sg / self.initial_mb + 1e-6

        self.local_map_model = self.build_lm_preproc_model(local_map_in_sg, name)
        if self.params.use_pretrained_local_map_preproc:
            self.local_map_model.load_weights(path_to_weights=path_to_local_pretrained_weights)
        flatten_local, local_map_2, local_map_3, local_map_4 = self.local_map_model(local_map_in_sg)


        # global map processing layers

        global_map_1 = tf.keras.layers.Conv2D(4, 5, activation='swish',
                                              strides=(1, 1),
                                              name=name + 'global_conv_' + str(0 + 1))(global_map_in_sg)  # out:17
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

        flatten_map = tf.keras.layers.Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local, states_proc])

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

        if self.params.use_skip:
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

    def build_lm_preproc_model(self, local_map_in, name=''):
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

    def get_softmax_goal(self, state):
        goal = self.get_soft_max_exploration(state)
        # print(f'### soft goal:{goal}')
        return goal

    def get_random_goal(self):
        arr = np.zeros(self.params.local_map_size)
        arr[:1] = 1
        np.random.shuffle(arr)
        # print('rand goal:', arr.shape)

        arr = np.random.choice(range(self.num_actions_hl), size=1)
        return arr

    def get_exploitation_goal(self, state):
        local_map_in = state.get_local_map()[tf.newaxis, ...]
        global_map_in = state.get_global_map(self.params.global_map_scaling)[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        goal, q = self._get_exploitation_goal(local_map_in, global_map_in, scalars) #.numpy()[0]
        goal = goal.numpy()[0]
        # goal = tf.one_hot(goal, depth=self.num_actions_hl)

        # print(f'goal: {goal}')
        # print(f'qval: {q[0].numpy()[goal]}')
        # highest_qval = q[a]
        # print(f'highest qval: {highest_qval}')

        return goal

    @tf.function
    def _get_exploitation_goal(self, local_map_in, global_map_in, scalars):
        a, q = self.exploit_model_hl([local_map_in, global_map_in, scalars])
        # tf.debugging.assert_all_finite(a, message='Nan in exploitation goal')
        return a, q

    def get_eps_greedy_action(self, state):

        p = np.random.random()
        if p < self.epsilon:
            a = np.random.choice(range(self.num_actions_hl), size=1)
        else:
            a = self.get_exploitation_goal(state)

        if self.epsilon > self.params.final_epsilon:
            self.epsilon -= (self.params.initial_epsilon - self.params.final_epsilon) / self.params.eps_steps

        return a

    def get_soft_max_exploration(self, state):
        local_map_in = state.get_local_map()[tf.newaxis, ...]
        global_map_in = state.get_global_map(self.params.global_map_scaling)[tf.newaxis, ...]
        scalars_in = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        # if self.training:
        #     print(f'sum inputs: {tf.reduce_sum(local_map_in)} {tf.reduce_sum(global_map_in)} {tf.reduce_sum(scalars_in)/self.initial_mb}')
        if np.any(np.isnan(local_map_in)) or np.any(np.isnan(global_map_in)) or np.any(np.isnan(scalars_in)):
            print(f'###################### Nan in act input: {np.isnan(local_map_in)}')
        p = self._get_soft_max_exploration(local_map_in, global_map_in, scalars_in).numpy()[0]
        a = np.random.choice(range(self.num_actions_hl), size=1, p=p)
        # print(f'choosen act: {a}')
        return a

    @tf.function
    def _get_soft_max_exploration(self, local_map_in, global_map_in, scalars_in):
        p, q, max = self.soft_explore_model_hl([local_map_in, global_map_in, scalars_in])
        # print(f' sum Q vals: {tf.reduce_sum(q)}')
              #f'\nHighest IDX: {max}\nhighest prob idx: {np.argmax(p.numpy()[0])}')
        tf.debugging.assert_all_finite(p, message='Nan in soft explore output')
        return p

    def hard_update_hl(self):
        self.target_network_hl.set_weights(self.q_network_hl.get_weights())

    def soft_update_hl(self, alpha):
        weights = self.q_network_hl.get_weights()
        # if np.any(tf.math.is_nan(weights)) == True:
        #     print(f'###################### Nan in experiences: {np.isnan(weights)}')
        target_weights = self.target_network_hl.get_weights()
        self.target_network_hl.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def train_hl(self, experiences):
        # print(f'local map shape: {experiences[0].shape, experiences[0][0].shape, experiences[0]}')
        nan = [np.any(np.isnan(experiences[i])) for i in range(len(experiences))]
        if np.any(nan):
            print(f'###################### Nan in experiences: {np.isnan(experiences)}')
        local_map = tf.convert_to_tensor(experiences[0], dtype=tf.float64)  # np.asarray(experiences[0]).astype(np.float32))
        global_map = tf.convert_to_tensor(experiences[1], dtype=tf.float64)
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = tf.convert_to_tensor(experiences[4], dtype=tf.float64)
        next_local_map = tf.convert_to_tensor(experiences[5], dtype=tf.float64)
        next_global_map = tf.convert_to_tensor(experiences[6], dtype=tf.float64)
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float64)
        terminated = tf.convert_to_tensor(experiences[8])
        if self.var ==True:
            for i in range(len(experiences)):
                print(f'experiences {i}: {experiences[i]}')
            self.var = False

        self._train_hl(local_map, global_map, scalars, action, reward, terminated, next_local_map, next_global_map,
                       next_scalars)

        self.soft_update_hl(self.params.alpha)

        # if self.counter%500 == 0:
        #     print(f'###################### hard updating target #################################')
        #     self.hard_update_hl()
        #     self.counter = 1
        # self.counter += 1

    @tf.function
    def _train_hl(self, local_map, global_map, scalars, action, reward, terminated, next_local_map, next_global_map,
                  next_scalars):
        self.training=True
        q_prime = self.q_prime_model_hl(
            [next_local_map, next_global_map, next_scalars])
        tf.debugging.assert_all_finite(q_prime, message='Nan in qprime')
        # Train Value network
        with tf.GradientTape() as tape:
            q_loss, q_new_hl, q_update_hot_hl, q_old_hl = self.q_loss_model_hl(
                [local_map, global_map, scalars, action, reward,
                 terminated, tf.stop_gradient(q_prime)])
        tf.debugging.assert_all_finite(q_loss, message='Nan in q_loss')
        # print_node(f'q_loss: {q_loss}')
        # print_node(f'q_prime: {q_prime}\nq_loss: {q_loss},\n q_new: {q_new_hl},\n q_upd: {q_update_hot_hl},\n q_old: {q_old_hl}')

        q_grads = tape.gradient(q_loss, self.q_network_hl.trainable_variables)
        # grad_avg = tf.reduce_mean([tf.reduce_mean(tf.abs(grads)) for grads in q_grads])
        # tf.summary.scalar("grad/avg_actor", grad_avg, self.i)
        # self.i += 1
        [tf.debugging.assert_all_finite(grads, message='Nan in grads') for grads in q_grads]
        # tf.debugging.assert_all_finite(q_grads, message='Nan in qgrads')
        self.q_optimizer_hl.apply_gradients(zip(q_grads, self.q_network_hl.trainable_variables))

    def save_weights_hl(self, path_to_weights):
        self.target_network_hl.save_weights(path_to_weights)

    def save_model_hl(self, path_to_model):
        self.target_network_hl.save(path_to_model)

    def load_weights_hl(self, path_to_weights):
        self.q_network_hl.load_weights(path_to_weights)
        self.hard_update_hl()

    @tf.RegisterGradient("ZeroGrad")
    def _zero_grad(self, unused_op, grad):
        return tf.zeros_like(grad)

    # this is added for gradient check of NaNs
    def check_numerics_with_exception(self, grad, var):
        try:
            tf.debugging.check_numerics(grad, message='Gradient %s check failed, possible NaNs' % var.name)
        except:
            return tf.constant(False, shape=())
        else:
            return tf.constant(True, shape=())

    def conditional_node(self, grads):
        num_nans_grads = tf.Variable(1.0, name='num_nans_grads')
        check_all_numeric_op = tf.reduce_sum(
            tf.cast(tf.stack([tf.logical_not(self.check_numerics_with_exception(grad, var)) for grad, var in grads]),
                    dtype=tf.float32))

        with tf.control_dependencies([tf.assign(num_nans_grads, check_all_numeric_op)]):
            # Apply the gradients to adjust the shared variables.
            self.q_optimizer_hl.apply_gradients(zip(grads, self.q_network_hl.trainable_variables))

    def fn_true_apply_grad(self, grads, global_step):
        apply_gradients_true = self.q_optimizer_hl.apply_gradients(grads, global_step=global_step)
        return apply_gradients_true

    #
    def fn_false_ignore_grad(self, grads, global_step):
        # print('batch update ignored due to nans, fake update is applied')
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": "ZeroGrad"}):
            for (grad, var) in grads:
                tf.assign(var, tf.identity(var, name="Identity"))
                apply_gradients_false = self.q_optimizer_hl.apply_gradients(grads, global_step=global_step)
        return apply_gradients_false

    # def get_global_map(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.global_map_model([boolean_map_in, float_map_in]).numpy()
    #
    # def get_local_map_hl(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.local_map_hl_model([boolean_map_in, float_map_in]).numpy()
    #
    # def get_total_map_hl(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.total_map_model_hl([boolean_map_in, float_map_in]).numpy()
