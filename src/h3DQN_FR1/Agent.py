import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D

import numpy as np


def print_node(x):
    print(x)
    return x


class H_DDQNAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 3
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

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


class H_DDQNAgent(object):

    def __init__(self, params: H_DDQNAgentParams, example_state, example_action, stats=None):

        self.params = params
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.float_map_shape = example_state.get_float_map_shape()
        self.scalars = example_state.get_num_scalars()
        self.num_actions = len(type(example_action))
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]

        # Create shared inputs
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        mask_input = Input(shape=(), name='mask_input', dtype=tf.bool)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_star_input = Input(shape=(), name='q_star_input', dtype=tf.float32)
        states_h = [boolean_map_input,
                  float_map_input,
                  scalars_input]

        states_l = [boolean_map_input,
                    float_map_input,
                    mask_input]

        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)
        padded_map = tf.concat([map_cast, float_map_input], axis=3)

        self.q_network_h = self.build_model_h(padded_map, scalars_input, states_h)
        self.target_network_h = self.build_model_h(padded_map, scalars_input, states_h, 'target_')

        self.q_network_l = self.build_model_h(padded_map, scalars_input, states_l)
        self.target_network_l = self.build_model_h(padded_map, scalars_input, states_l, 'target_')

        self.hard_update_l()
        self.hard_update_h()

        if self.params.use_global_local:
            self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                          outputs=self.global_map)
            self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.local_map)
            self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.total_map)

        q_values_l = self.q_network_l.output
        q_target_values_l = self.target_network_l.output

        q_values_h = self.q_network_h.output
        q_target_values_h = self.target_network_h.output

        ########## LOW-level Agent ###########3#
        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action_l = tf.argmax(q_values_l, axis=1, name='max_action', output_type=tf.int64)
        max_action_target_l = tf.argmax(q_target_values_l, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action_l = tf.one_hot(max_action_l, depth=self.num_actions, dtype=float)
        q_star = tf.reduce_sum(tf.multiply(one_hot_max_action_l, q_target_values_l, name='mul_hot_target'), axis=1,
                               name='q_star')
        self.q_star_model_l = Model(inputs=states_l, outputs=q_star)

        # Define Bellman loss
        one_hot_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=1.0, off_value=0.0, dtype=float)
        one_cold_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0, dtype=float)
        q_old = tf.stop_gradient(tf.multiply(q_values_l, one_cold_rm_action))
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated)), 1)
        q_update_hot = tf.multiply(q_update, one_hot_rm_action)
        q_new = tf.add(q_update_hot, q_old)
        q_loss = tf.losses.MeanSquaredError()(q_new, q_values_l)
        self.q_loss_model_l = Model(
            inputs=[boolean_map_input, float_map_input, scalars_input, action_input, reward_input,
                    termination_input, q_star_input],
            outputs=q_loss)

        # Exploit act model
        self.exploit_model = Model(inputs=states_l, outputs=max_action_l)
        self.exploit_model_target = Model(inputs=states_l, outputs=max_action_target_l)

        # Softmax explore model
        softmax_scaling = tf.divide(q_values_l, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        self.soft_explore_model = Model(inputs=states_l, outputs=softmax_action)

        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model_l.summary()

        if stats:
            stats.set_model(self.target_network_l)



        ########## HIGH-Level Agent ##############

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action_h = tf.argmax(q_values_h, axis=1, name='max_action', output_type=tf.int64)
        max_action_target_h = tf.argmax(q_target_values_h, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action_h = tf.one_hot(max_action_h, depth=self.num_actions, dtype=float)
        q_star_h = tf.reduce_sum(tf.multiply(one_hot_max_action_h, q_target_values_h, name='mul_hot_target'), axis=1,
                               name='q_star')
        self.q_star_model_h = Model(inputs=states_h, outputs=q_star_h)

        # Define Bellman loss
        one_hot_rm_action_h = tf.one_hot(action_input, depth=self.num_actions, on_value=1.0, off_value=0.0,
                                       dtype=float)
        one_cold_rm_action_h = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0,
                                        dtype=float)
        q_old_h = tf.stop_gradient(tf.multiply(q_values_h, one_cold_rm_action_h))
        gamma_terminated_h = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update_h = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated_h)), 1)
        q_update_hot_h = tf.multiply(q_update_h, one_hot_rm_action_h)
        q_new_h = tf.add(q_update_hot_h, q_old_h)
        q_loss_h = tf.losses.MeanSquaredError()(q_new_h, q_values_h)
        self.q_loss_model_h = Model(
            inputs=[boolean_map_input, float_map_input, scalars_input, action_input, reward_input,
                    termination_input, q_star_input],
            outputs=q_loss_h)

        # Exploit act model
        self.exploit_model_h = Model(inputs=states_h, outputs=max_action_h)
        self.exploit_model_target_h = Model(inputs=states_h, outputs=max_action_target_h)

        # Softmax explore model
        softmax_scaling_h = tf.divide(q_values_h, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action_h = tf.math.softmax(softmax_scaling_h, name='softmax_action')
        self.soft_explore_model = Model(inputs=states_h, outputs=softmax_action_h)

        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model_h.summary()

        if stats:
            stats.set_model(self.target_network_h)

    def build_model_h(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)

        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_model_l(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)

        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model


    def create_map_proc(self, conv_in, name):

        if self.params.use_global_local:
            # Forking for global and local map
            # Global Map
            global_map = tf.stop_gradient(
                AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

            self.global_map = global_map
            self.total_map = conv_in

            for k in range(self.params.conv_layers):
                global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                    strides=(1, 1),
                                    name=name + 'global_conv_' + str(k + 1))(global_map)

            flatten_global = Flatten(name=name + 'global_flatten')(global_map)

            # Local Map
            crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
            local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
            self.local_map = local_map

            for k in range(self.params.conv_layers):
                local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                   strides=(1, 1),
                                   name=name + 'local_conv_' + str(k + 1))(local_map)

            flatten_local = Flatten(name=name + 'local_flatten')(local_map)

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

    def act(self, state):
        return self.get_soft_max_exploration(state)

    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_exploitation_action(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def get_soft_max_exploration(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        p = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        return np.random.choice(range(self.num_actions), size=1, p=p)

    def get_exploitation_action_target(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model_target([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def hard_update_h(self):
        self.target_network_h.set_weights(self.q_network_l.get_weights())

    def hard_update_l(self):
        self.target_network_l.set_weights(self.q_network_h.get_weights())

    def soft_update_h(self, alpha):
        weights = self.q_network_h.get_weights()
        target_weights = self.target_network_h.get_weights()
        self.target_network_h.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def soft_update_l(self, alpha):
        weights = self.q_network_l.get_weights()
        target_weights = self.target_network_l.get_weights()
        self.target_network_l.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def train_h(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]

        q_star = self.q_star_model_h(
            [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            q_loss = self.q_loss_model_h(
                [boolean_map, float_map, scalars, action, reward,
                 terminated, q_star])
        q_grads = tape.gradient(q_loss, self.q_network_h.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network_h.trainable_variables))

        self.soft_update_h(self.params.alpha)

    def train_l(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]

        q_star = self.q_star_model_l(
            [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            q_loss = self.q_loss_model_l(
                [boolean_map, float_map, scalars, action, reward,
                 terminated, q_star])
        q_grads = tape.gradient(q_loss, self.q_network_h.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network_h.trainable_variables))

        self.soft_update_l(self.params.alpha)

    def save_weights_l(self, path_to_weights):
        self.target_network_l.save_weights(path_to_weights)

    def save_weights_h(self, path_to_weights):
        self.target_network_h.save_weights(path_to_weights)

    def save_model_l(self, path_to_model):
        self.target_network_l.save(path_to_model)

    def save_model_h(self, path_to_model):
        self.target_network_h.save(path_to_model)

    def load_weights_l(self, path_to_weights):
        self.q_network_l.load_weights(path_to_weights)
        self.hard_update_l()

    def load_weights_h(self, path_to_weights):
        self.q_network_h.load_weights(path_to_weights)
        self.hard_update_h()

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()
