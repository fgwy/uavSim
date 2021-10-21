import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, Reshape

import numpy as np


def print_node(x):
    print(x)
    return x


def identify_idx_highest_val_in_tensor(tensor):
    return tf.math.argmax(tensor)


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


class HL_DDQNAgent(object):

    def __init__(self, params: HL_DDQNAgentParams, example_state, example_action_hl, stats=None):

        self.params = params
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.float_map_shape = example_state.get_float_map_shape()
        self.scalars = example_state.get_num_scalars()
        self.goal_target_shape = example_state.get_goal_target_shape()
        self.num_actions_hl = self.params.local_map_size ** 2
        self.example_goal = example_state.get_example_goal()
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]

        # Create shared inputs
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_hl_input', dtype=tf.float32)
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_hl_input = Input(shape=(), name='reward_hl_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_star_hl_input = Input(shape=(), name='q_star_hl_input', dtype=tf.float32)

        states_hl = [boolean_map_input,
                     float_map_input,
                     scalars_input]

        map_cast_hl = tf.cast(boolean_map_input, dtype=tf.float32)
        padded_map_hl = tf.concat([map_cast_hl, float_map_input], axis=3)

        self.q_network_hl = self.build_model_hl(padded_map_hl, scalars_input, states_hl)
        self.target_network_hl = self.build_model_hl(padded_map_hl, scalars_input, states_hl, 'target_hl_')

        self.hard_update_hl()

        if self.params.use_global_local:
            self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                          outputs=self.global_map)
            self.local_map_hl_model = Model(inputs=[boolean_map_input, float_map_input],
                                            outputs=self.local_map)
            self.total_map_model_hl = Model(inputs=[boolean_map_input, float_map_input],
                                            outputs=self.total_map_hl)

        q_values_hl = self.q_network_hl.output
        q_target_values_hl = self.target_network_hl.output

        ########## HIGH-Level Agent ##############

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action_hl = tf.argmax(q_values_hl, axis=1, name='max_action',
                                  output_type=tf.int64)
        max_action_target_hl = tf.argmax(q_target_values_hl, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action_hl = tf.one_hot(max_action_hl, depth=self.num_actions_hl, dtype=float, on_value=0.0,
                                           off_value=1.0)
        one_hot_max_action_hl = tf.squeeze(one_hot_max_action_hl)
        q_star_hl = tf.reduce_sum(tf.multiply(one_hot_max_action_hl, q_target_values_hl, name='mul_hot_target'), axis=1,
                                  name='q_star_hl')
        self.q_star_model_hl = Model(inputs=states_hl, outputs=q_star_hl)

        # Define Bellman loss
        one_hot_rm_action_hl = tf.one_hot(action_input, depth=self.num_actions_hl, on_value=1.0, off_value=0.0,
                                          dtype=float)
        one_cold_rm_action_hl = tf.one_hot(action_input, depth=self.num_actions_hl, on_value=0.0, off_value=1.0,
                                           dtype=float)
        q_old_hl = tf.stop_gradient(tf.multiply(q_values_hl, one_cold_rm_action_hl))
        gamma_terminated_hl = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update_hl = tf.expand_dims(tf.add(reward_hl_input, tf.multiply(q_star_hl_input, gamma_terminated_hl)), 1)
        q_update_hot_hl = tf.multiply(q_update_hl, one_hot_rm_action_hl)
        q_new_hl = tf.add(q_update_hot_hl, q_old_hl)
        q_loss_hl = tf.losses.MeanSquaredError()(q_new_hl, q_values_hl)
        self.q_loss_model_hl = Model(
            inputs=[boolean_map_input, float_map_input, scalars_input, action_input, reward_hl_input,
                    termination_input, q_star_hl_input],
            outputs=q_loss_hl)

        # Exploit act model
        self.exploit_model_hl = Model(inputs=states_hl, outputs=max_action_hl)
        self.exploit_model_target_hl = Model(inputs=states_hl, outputs=max_action_target_hl)

        # Softmax explore model
        softmax_scaling_hl = tf.divide(q_values_hl, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action_hl = tf.math.softmax(softmax_scaling_hl, name='softmax_action')
        self.soft_explore_model_hl = Model(inputs=states_hl, outputs=softmax_action_hl)

        self.q_optimizer_hl = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model_hl.summary()

        if stats:
            stats.set_model(self.target_network_hl)

    def build_model_hl(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)

        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='elu', name=name + 'hidden_layer_all_hl_' + str(k))(
                layer)

        output = Dense(units=self.params.local_map_size ** 2, activation='linear', name=name + 'last_dense_layer_hl')(
            layer)
        # output = Reshape((self.params.local_map_size, self.params.local_map_size), name=name + 'output_layer_hl')(
        #     output)

        model = Model(inputs=inputs, outputs=output)

        return model

    def create_map_proc(self, conv_in, name):

        if self.params.use_global_local:
            # Forking for global and local map
            # Global Map
            global_map = tf.stop_gradient(
                AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

            self.global_map = global_map
            self.total_map_hl = conv_in

            # Todo: create discussed model

            for k in range(self.params.conv_layers):
                global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                    strides=(1, 1),
                                    name=name + 'global_conv_' + str(k + 1))(global_map)

            flatten_global = Flatten(name=name + 'global_flatten')(global_map)

            # Local Map
            flatten_local = self.create_local_map_proc_hl(conv_in, name)

            return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])
        else:
            conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu', strides=(1, 1),
                              name=name + 'map_conv_0')(conv_in)
            for k in range(self.params.conv_layers - 1):
                conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                  strides=(1, 1),
                                  name=name + 'map_conv_' + str(k + 1))(conv_map)

            flatten_map = Flatten(name=name + 'flatten')(conv_map)
            return flatten_map

    def create_local_map_proc_hl(self, conv_in, name):
        # Local Map_HL
        crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
        local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
        self.local_map = local_map

        self.total_map = conv_in

        for k in range(self.params.conv_layers):
            local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                               strides=(1, 1),
                               name=name + 'local_conv_' + str(k + 1))(local_map)

        return Flatten(name=name + 'local_flatten')(local_map)

    def get_goal(self, state):
        goal = self.get_soft_max_exploration(state)
        print('soft goal:', goal.shape, goal)
        return goal

    def get_random_goal(self):
        arr = np.zeros(self.params.local_map_size)
        arr[:1] = 1
        np.random.shuffle(arr)
        print('rand goal:', arr.shape)

        arr = np.random.choice(range(self.num_actions_hl), size=1)
        return arr

    def get_exploitation_goal(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        goal = self.exploit_model_hl([boolean_map_in, float_map_in, scalars]).numpy()[0]
        # goal = tf.one_hot(goal, depth=self.num_actions_hl)
        return goal

    def get_soft_max_exploration(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        p = self.soft_explore_model_hl([boolean_map_in, float_map_in, scalars]).numpy()[0]
        a = np.random.choice(range(self.num_actions_hl), size=1, p=p)
        # a = tf.one_hot(a, depth=self.num_actions_hl)
        return a

    def hard_update_hl(self):
        self.target_network_hl.set_weights(self.q_network_hl.get_weights())

    def soft_update_hl(self, alpha):
        weights = self.q_network_hl.get_weights()
        target_weights = self.target_network_hl.get_weights()
        self.target_network_hl.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def train_hl(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]

        q_star = self.q_star_model_hl(
            [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            q_loss = self.q_loss_model_hl(
                [boolean_map, float_map, scalars, action, reward,
                 terminated, q_star])
        q_grads = tape.gradient(q_loss, self.q_network_hl.trainable_variables)
        self.q_optimizer_hl.apply_gradients(zip(q_grads, self.q_network_hl.trainable_variables))

        self.soft_update_hl(self.params.alpha)

    def save_weights_hl(self, path_to_weights):
        self.target_network_hl.save_weights(path_to_weights)

    def save_model_hl(self, path_to_model):
        self.target_network_hl.save(path_to_model)

    def load_weights_hl(self, path_to_weights):
        self.q_network_hl.load_weights(path_to_weights)
        self.hard_update_hl()

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map_hl(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_hl_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map_hl(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model_hl([boolean_map_in, float_map_in]).numpy()
