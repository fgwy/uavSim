import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda
import tensorflow_probability as tf_p

import numpy as np


def print_node(x):
    print(x)
    return x


class PPOAgentParams:
    def __init__(self,
                 gamma=0.98,
                 lam=0.95,
                 entropy_coeff=0.004,
                 lr_a=0.0005,
                 lr_c=0.001,
                 clipping_range=0.2,
                 normalize=True,
                 tau=0.1, _1st_hidden_layer_size=258,
                 _2nd_hidden_layer_size=126,
                 _3rd_hidden_layer_size=64,
                 kernel_l2_norm=0.01):
        self.gamma = gamma
        self.lam = lam
        self.entropy_coeff = entropy_coeff
        self.kernel_l2_norm = kernel_l2_norm

        self.lr_a = lr_a  # Actor learning rate
        self.lr_c = lr_c  # Critic learning rate

        self.clipping_range = clipping_range
        self.normalize = normalize
        self.tau = tau

        self._1st_hidden_layer_size = _1st_hidden_layer_size
        self._2nd_hidden_layer_size = _2nd_hidden_layer_size
        self._3rd_hidden_layer_size = _3rd_hidden_layer_size


class PPOAgent:
    def __init__(self, env_params, params=PPOAgentParams(), state_size=5, action_size=1):
        self.state_size = state_size
        self.action_size = action_size
        self.params = params
        self.env_params = env_params
        self.x_threshold = env_params[0]

        self.actor_optimizer = tf.optimizers.Adadelta(learning_rate=params.lr_a)
        self.critic_optimizer = tf.optimizers.Adadelta(learning_rate=params.lr_c)

        gamma = tf.constant(self.params.gamma, dtype=float)

        # Shared Inputs
        state_input = Input(shape=(self.state_size,), name='state_input', dtype=tf.float64)
        next_state_input = Input(shape=(self.state_size,), name='next_state_input',
            dtype=tf.float64)
        action_input = Input(shape=(), name='action_input')
        reward_input = Input(shape=(), name='reward_input')
        termination_input = Input(shape=(), name='termination_input', dtype=bool)
        disc_sum_rewards = Input(shape=(), name='disc_sum_rewards')
        env_params_input = Input(shape=(), name='env_params_input', )
        old_logpi_input = Input(shape=(), name='old_logpi')
        gaes_input = Input(shape=(), name='gaes_input')

        states_concat = tf.concat([state_input, next_state_input], axis=0)

        # Build Critic
        self.critic_model = Model(inputs=(state_input),
            outputs=self.build_critic(state_input))
        self.target_critic_model = Model(inputs=([state_input, next_state_input]),
            outputs=self.build_critic((states_concat),
                name='target_'))
        target_critic_estimate = tf.squeeze(self.target_critic_model.output)

        # Build Actor
        self.actor_model = self.build_actor(state_input)
        mu = tf.squeeze(self.actor_model.outputs[0])
        sigma = tf.add(tf.squeeze(self.actor_model.outputs[1]), 0.05)
        distribution = tf_p.distributions.Normal(mu, sigma, name='action_distribution')
        self.actor_exploit_model = Model(inputs=[state_input], outputs=mu)

        # Policy
        sample = distribution.sample(name='action_sample')
        self.policy_model = Model(inputs=([state_input]), outputs=[sample, mu, sigma])

        # Critic Loss
        # critic_loss = tf.losses.MeanSquaredError()(disc_sum_rewards, self.critic_model.output)
        critic_loss = tf.keras.losses.MSE(disc_sum_rewards, self.critic_model.output)
        self.critic_loss_model = Model(inputs=[state_input, disc_sum_rewards], outputs=critic_loss)

        # Actor Loss

        # Advantage
        value_estimate, next_value_estimate = tf.split(target_critic_estimate, num_or_size_splits=2,
            axis=0)

        gamma_terminated = tf.multiply(
            tf.subtract(tf.constant(1.0, dtype=float), tf.cast(termination_input, dtype=float)),
            gamma)

        # A = (r + (1 - terminated) * V(s')) - V(s)
        bellman_residual = tf.subtract(
            tf.add(reward_input, tf.multiply(gamma_terminated, next_value_estimate)),
            value_estimate, name='advantage')

        # Entropy bonus
        entropy = distribution.entropy()
        entropy_term = tf.multiply(entropy, self.params.entropy_coeff)

        # Fraction Thingy

        log_prob_new = distribution.log_prob(action_input, name='log_prob_new')
        log_prob_old = tf.stop_gradient(tf.add(old_logpi_input, 1e-10), name='log_prob_old')

        ratio = tf.exp(tf.subtract(log_prob_new, log_prob_old))
        # ratio_times_adv = tf.multiply(ratio, bellman_residual)
        ratio_times_adv = tf.multiply(ratio, gaes_input)

        sign = tf.sign(gaes_input)

        # clipped ratio = 1 - or + clipping range depending on sign of adv
        clipped_ratio = tf.add(tf.cast(1., dtype=tf.float32),
            tf.multiply(sign, self.params.clipping_range))
        clipped_advantage = tf.multiply(clipped_ratio, gaes_input)

        min_ratio = tf.minimum(clipped_advantage, ratio_times_adv)
        # advantage_loss = tf.reduce_mean(tf.multiply(min_ratio, -1, name='advantage_loss'))
        advantage_loss = tf.multiply(min_ratio, -1, name='advantage_loss')

        total_loss = tf.add(tf.reduce_mean(advantage_loss),
            tf.multiply(tf.reduce_mean(entropy_term), -1))
        self.advantage_loss_model = Model(
            inputs=[state_input, action_input, reward_input, next_state_input, termination_input,
                    old_logpi_input, gaes_input],
            outputs=total_loss)


    def train_and_update_actor(self, states, actions, rewards, next_states, terminations,
                               old_logpis, gaes):
        with tf.GradientTape() as tape:
            advantage_loss = self.advantage_loss_model([states, actions, rewards, next_states,
                                                        terminations, old_logpis, gaes])
        grads = tape.gradient(advantage_loss, self.actor_model.trainable_variables)

        # grads = [tf.clip_by_norm(g, 2) for g in grads]
        if grads=='NaN':
            print('################### Nan in Actor!!! ############################')

        else:
            self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

        return advantage_loss, grads


    def train_and_update_critic(self, states, disc_sum_rewards):

        # self.critic_model.fit()
        # self.soft_update_target(self.params.tau)
        with tf.GradientTape() as tape:
            critic_loss = self.critic_loss_model([states, disc_sum_rewards])
        grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        # grads = [tf.clip_by_norm(g, 3) for g in grads]

        if grads=='NaN':
            print('################### Nan in Critic!!! ############################')

        else:
            self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

        # Soft update target critic
        self.soft_update_target(self.params.tau)

        return critic_loss, grads

    def hard_update_target_critic(self):
        self.target_critic_model.set_weights(self.critic_model.get_weights())

    def soft_update_target(self, tau):
        weights = self.critic_model.get_weights()
        target_weights = self.target_critic_model.get_weights()
        self.target_critic_model.set_weights(
            [w_new * tau + w_targ * (1. - tau) for w_new, w_targ in zip(weights, target_weights)])

    def build_critic(self, state_input, name=''):
        h1 = Dense(self.params._1st_hidden_layer_size, activation='relu', name=name +
                                                                               'hidden_layer_1_c',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(state_input)
        h2 = Dense(self.params._2nd_hidden_layer_size, activation='relu', name=name +
                                                                               'hidden_layer_2_c',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(h1)
        h3 = Dense(self.params._3rd_hidden_layer_size, activation='relu', name=name +
                                                                               'hidden_layer_3_c',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(h2)
        output_layer_c = Dense(1, name=name + 'output_layer_c', activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(h3)

        return output_layer_c

    def build_actor(self, state_input):
        h1 = Dense(self.params._1st_hidden_layer_size, activation='relu', name='hidden_layer_1_a',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(state_input)
        h2 = Dense(self.params._2nd_hidden_layer_size, activation='relu', name='hidden_layer_2_a',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(h1)
        h3 = Dense(self.params._3rd_hidden_layer_size, activation='relu', name='hidden_layer_3_a',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(h2)
        output_mean = Dense(1, name='output_mean', activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(h3)
        output_std = Dense(1, name='output_std', activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(self.params.kernel_l2_norm))(h3)

        return Model(inputs=state_input, outputs=[output_mean, output_std])

    def getr_action(self, state):
        action, mu, sigma = self.policy_model(tf.convert_to_tensor([state]))

        # TODO sample logpi in model
        dist = tf_p.distributions.Normal(mu, sigma)

        logpi = dist.log_prob(action)

        return action, mu, sigma, logpi

    def get_exploit_action(self, state):
        return self.actor_exploit_model(tf.convert_to_tensor([state])).numpy()

    def get_value(self, state):
        return self.critic_model([state])

    def get_value_target(self, state):
        return self.target_critic_model([tf.convert_to_tensor(state), tf.convert_to_tensor(
            state)])  # [0]  # WTF is happening here?

    def get_value_and_nextvalue_target(self, state, next_state):
        return self.target_critic_model([tf.convert_to_tensor(state), tf.convert_to_tensor(
            next_state)])

    def get_action(self, state):
        local_map_in = state.get_local_map()[tf.newaxis, ...]
        global_map_in = state.get_global_map(self.params.global_map_scaling)[tf.newaxis, ...]
        scalars_in = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        goal_probs = self._get_goal(local_map_in, global_map_in, scalars_in).numpy()[0]

    @tf.function
    def _get_action(self, local_map_in, global_map_in, scalars_in):

        goal_array,  = self.policy_model([local_map_in, global_map_in, scalars_in])
        tf.debugging.assert_all_finite(goal_array, message='Nan in soft explore output')