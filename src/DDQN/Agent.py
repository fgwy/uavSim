import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D

import numpy as np

from src.DDQN.models import build_flat_model_masked, build_flat_model_no_mask


def print_node(x):
    print(x)
    return x


class DDQNAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
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

        self.masked = True


class DDQNAgent(object):

    def __init__(self, params: DDQNAgentParams, example_state, example_action, stats=None):

        self.params = params
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.float_map_shape = example_state.get_float_map_shape()
        self.scalars = example_state.get_num_scalars()
        self.num_actions = len(type(example_action))
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]
        self.local_map_shape = example_state.get_local_map_shape()
        self.global_map_shape = example_state.get_global_map_shape(self.params.global_map_scaling)
        self.initial_mb = tf.convert_to_tensor(example_state.get_initial_mb(), dtype=tf.float32)

        self.local_map_shape = example_state.get_local_map_shape()
        self.global_map_shape = example_state.get_global_map_shape(self.params.global_map_scaling)

        # Create shared inputs
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_prime_input = Input(shape=(), name='q_star_input', dtype=tf.float32)
        local_map_input = Input(shape=self.local_map_shape, name='local_map_input', dtype=tf.float32)
        # local_map_input = Input(shape=self.local_map_shape, name='local_map_input')
        global_map_input = Input(shape=self.global_map_shape, name='global_map_input')
        states = [local_map_input,
                  global_map_input,
                  scalars_input]
        if self.params.masked:
            self.q_network = build_flat_model_masked(states, self.num_actions, self.initial_mb)
            self.target_network = build_flat_model_masked(states, self.num_actions, self.initial_mb, None, 'target_')
        else:
            self.q_network = build_flat_model_no_mask(states, self.num_actions, self.initial_mb)
            self.target_network = build_flat_model_no_mask(states, self.num_actions, self.initial_mb, None, 'target_')
        self.hard_update()

        # if self.params.use_global_local:
        #     self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
        #                                   outputs=self.global_map)
        #     self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
        #                                  outputs=self.local_map)
        #     self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
        #                                  outputs=self.total_map)

        q_values = self.q_network.output
        q_target_values = self.target_network.output

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action = tf.argmax(q_values, axis=1, name='max_action', output_type=tf.int64)
        max_action_target = tf.argmax(q_target_values, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=tf.bool, on_value=True,
                                           off_value=False)
        # one_hot_max_action = tf.squeeze(one_hot_max_action)
        q_prime = tf.reduce_sum(tf.where(one_hot_max_action, q_target_values, 0, name='where_hot_target'),
                                   axis=1,
                                   name='q_prime')
        self.q_prime_model = Model(inputs=states, outputs=q_prime)

        # Define Bellman loss
        one_hot_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=True, off_value=False,
                                          dtype=tf.bool)
        # one_cold_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0,
        #                                    dtype=float)
        # q_old = tf.stop_gradient(tf.multiply(q_values, one_cold_rm_action))
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update = tf.add(reward_input, tf.multiply(q_prime_input, gamma_terminated))

        q_pred = tf.reduce_sum(tf.where(one_hot_rm_action, q_values, 0), axis=1)

        # q_update_reduced = tf.reduce_sum(tf.multiply(q_update, one_hot_rm_action), axis=1)
        # q_new = tf.add(q_update_hot, q_old)
        q_loss = tf.losses.MeanSquaredError()(q_update, q_pred)
        self.q_loss_model = Model(
            inputs=[local_map_input, global_map_input, scalars_input, action_input, reward_input,
                    termination_input, q_prime_input],
            outputs=[q_loss])
        # outputs=[q_loss, q_new, q_update_hot, q_old])

        # Exploit act model
        self.exploit_model = Model(inputs=states, outputs=(max_action)) # , q_values))
        self.exploit_model_target = Model(inputs=states, outputs=max_action_target)

        # Softmax explore model
        tf.debugging.assert_all_finite(self.params.soft_max_scaling, message='Nan in softmax_scaling_factor')
        softmax_scaling = tf.divide(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        # tf.debugging.assert_all_finite(softmax_scaling, message='Nan in softmax_scaling')
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        # tf.debugging.assert_all_finite(softmax_action, message='Nan in softmax_action')
        self.soft_explore_model = Model(inputs=states, outputs=(softmax_action)) #, q_values, max_action))

        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model.summary()

        if stats:
            stats.set_model(self.target_network)

    def act(self, state):
        return self.get_soft_max_exploration(state)

    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_exploitation_action(self, state):

        local_map_in = state.get_local_map()[tf.newaxis, ...]
        global_map_in = state.get_global_map(self.params.global_map_scaling)[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self._get_exploitation_action(local_map_in, global_map_in, scalars).numpy()[0]

    @tf.function
    def _get_exploitation_action(self, local_map_in, global_map_in, scalars):
        a =  self.exploit_model([local_map_in, global_map_in, scalars])
        # tf.debugging.assert_all_finite(a, message='Nan in exploit_act')
        return a

    def get_soft_max_exploration(self, state):

        local_map_in = state.get_local_map()[tf.newaxis, ...]
        # tf.debugging.assert_all_finite(local_map_in, message='Nan in lm_in')
        global_map_in = state.get_global_map(self.params.global_map_scaling)[tf.newaxis, ...]
        # tf.debugging.assert_all_finite(global_map_in, message='Nan in gm_in')
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        # tf.debugging.assert_all_finite(scalars, message='Nan in scal')
        p = self._get_soft_max_exploration(local_map_in, global_map_in, scalars).numpy()[0]
        # tf.debugging.assert_all_finite(p, message='Nan in p')
        return np.random.choice(range(self.num_actions), size=1, p=p)

    @tf.function
    def _get_soft_max_exploration(self, local_map_in, global_map_in, scalars):
        return self.soft_explore_model([local_map_in, global_map_in, scalars])

    def get_exploitation_action_target(self, state):

        local_map_in = state.get_local_map()[tf.newaxis, ...]
        global_map_in = state.get_global_map(self.params.global_map_scaling)[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self._get_exploitation_action_target(local_map_in, global_map_in, scalars).numpy()[0]

    @tf.function
    def _get_exploitation_action_target(self, local_map_in, global_map_in, scalars):
        return self.exploit_model_target([local_map_in, global_map_in, scalars])


    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def soft_update(self, alpha):
        weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        self.target_network.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def train(self, experiences):
        # print('training')
        # for e in experiences:
        #     for val in e:
        #         if np.isnan(np.any(val)):
        #             print('NAN in exp')
        local_map = tf.convert_to_tensor(experiences[0])
        global_map = tf.convert_to_tensor(experiences[1])
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = tf.convert_to_tensor(experiences[4])
        next_local_map = tf.convert_to_tensor(experiences[5])
        next_global_map = tf.convert_to_tensor(experiences[6])
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = tf.convert_to_tensor(experiences[8])
        self._train(next_local_map, next_global_map, next_scalars, local_map, global_map, scalars, action, reward,
                 terminated)
        self.soft_update(self.params.alpha)

    @tf.function
    def _train(self, next_local_map, next_global_map, next_scalars, local_map, global_map, scalars, action, reward,
                 terminated ):

        q_star = self.q_prime_model(
            [next_local_map, next_global_map, next_scalars])
        # tf.debugging.assert_all_finite(q_star, message='Nan in qprime')
        # Train Value network
        with tf.GradientTape() as tape:
            q_loss = self.q_loss_model(
                [local_map, global_map, scalars, action, reward,
                 terminated, tf.stop_gradient(q_star)])
        q_grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        # [tf.debugging.assert_all_finite(grads, message='Nan in grads') for grads in q_grads]
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))


    def save_weights(self, path_to_weights):
        self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.target_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.q_network.load_weights(path_to_weights)
        self.hard_update()

    # def get_global_map(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.global_map_model([boolean_map_in, float_map_in]).numpy()
    #
    # def get_local_map(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.local_map_model([boolean_map_in, float_map_in]).numpy()
    #
    # def get_total_map(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.total_map_model([boolean_map_in, float_map_in]).numpy()
