import wild_fire_mlc_gym as wfg
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.python.keras.layers import AvgPool2D, Input

from Agent import Agent
from DebugAgent import DebugAgent
from DDPGTrainer import DDPGTrainer
from MapParams import MapParams
from TrainerBase import Trainer


class AgentManager:

    def __init__(self, trainer: Trainer, stats, debug=False):

        self.trainer = trainer
        self.gym = None
        self.stats = stats
        self.deterministic_actor_prob = 0
        self.deterministic_actor_prob_decay = 0.98
        self.max_num_agents = 1
        self.actors = [self.trainer.get_latest_actor(agent_id) for agent_id in range(self.max_num_agents)]
        self.agents = [None] * self.max_num_agents
        self.debug = debug
        self.max_life = 100.0 * 1000
        self.max_age = 700.0 * 1000

        self.map_params = MapParams()

        self.map_input_dim = 8

        x_width = (self.map_params.shape[0] - 1) * self.map_params.cell_size
        centered_size = self.map_params.shape[0] * 2 - 1

        lin = np.linspace(-x_width, x_width, num=centered_size)
        lin = np.expand_dims(lin, -1)
        y_dist = np.repeat(lin, repeats=centered_size, axis=1)
        x_dist = y_dist.transpose()
        self.constant_input = np.stack([x_dist, y_dist], axis=-1)

        with tf.device('/cpu:0'):
            map_in = Input(
                shape=(centered_size, centered_size, self.map_input_dim),
                dtype=float)
            scalar_in = Input(shape=(1,), dtype=float)
            life_normalize = scalar_in / self.max_life
            global_map = AvgPool2D([self.map_params.global_map_scaling] * 2)(map_in)
            crop_frac = float(self.map_params.local_map_size) / float(centered_size)
            local_map = tf.image.central_crop(map_in, crop_frac)
            self.preproc_model = Model(inputs=[map_in, scalar_in], outputs=[global_map, local_map, life_normalize])

    def observation_processing(self, obs: wfg.observation):
        age_proc = obs.age / self.max_age
        mat = np.stack([obs.resolution, age_proc, obs.fuel, obs.fire, obs.reachable, obs.agents], axis=-1)
        x, y = obs.position
        padded_mat = np.pad(mat, ((self.map_params.shape[1] - 1 - y, y), (self.map_params.shape[0] - 1 - x, x), (0, 0)))

        dx_dy = (self.constant_input - np.reshape(obs.position_in_cell, [1, 1, -1])) / (
                    self.map_params.shape[0] * self.map_params.cell_size)
        padded_mat = np.concatenate([padded_mat, dx_dy], axis=2)

        scalars = np.array([obs.remaining_flying_time], dtype=float)

        return self.preproc_model([tf.expand_dims(padded_mat, 0), tf.expand_dims(scalars, 0)])

    def deterministic_action(self, agent_id):
        action = self.gym.get_potential_field_motion(agent_id)
        a0 = np.linalg.norm(action)
        return tf.convert_to_tensor([action[0] / a0, action[1] / a0], dtype=float)

    def act(self, obs: wfg.observation):
        proc = self.observation_processing(obs)

        agent_id = obs.agent_id % self.max_num_agents
        agent = self.agents[agent_id]
        if not agent:

            if self.gym and np.random.uniform() < self.deterministic_actor_prob:
                agent = Agent(actor=lambda _: self.deterministic_action(obs.agent_id),
                              action=self.trainer.action, action_noise=self.trainer.get_no_noise(),
                              rewards=self.trainer.rewards)
                self.deterministic_actor_prob *= self.deterministic_actor_prob_decay
                self.gym.add_highlighted_agent(obs.agent_id)
            else:
                actor = self.actors[agent_id]
                self.trainer.update_weights(actor)
                if agent_id == 0 and self.debug:
                    agent = DebugAgent(actor=actor, action=self.trainer.action,
                                       action_noise=self.trainer.get_exploration_noise(),
                                       rewards=self.trainer.rewards, trainer=self.trainer)
                else:
                    agent = Agent(actor=actor, action=self.trainer.action,
                                  action_noise=self.trainer.get_exploration_noise(),
                                  rewards=self.trainer.rewards)
            self.agents[agent_id] = agent

        action = agent.get_action(obs, proc)

        if obs.terminal:
            if self.trainer.training_thread.isAlive():
                self.trainer.training_thread.join()
            self.trainer.add_trajectory(agent.trajectory)
            self.trainer.start_training_thread()
            agent = self.agents[agent_id]
            self.stats.despawning_agent(agent)
            agent.clear()
            self.agents[agent_id] = None

            return np.array([0, 0])

        return action

    def set_deterministic_actor(self, det_actor):
        self.deterministic_actor = det_actor

    def set_gym(self, gym):
        self.gym = gym

    def act_centralized(self, obs):
        proc = self.observation_processing(obs)

        agent = self.agents[0]
        if not agent:
            if self.debug:
                agent = DebugAgent(actor=self.trainer.get_actor(), action=self.trainer.action,
                                   action_noise=self.trainer.get_exploration_noise(),
                                   rewards=self.trainer.rewards, trainer=self.trainer)
            else:
                agent = Agent(actor=self.trainer.get_actor(), action=self.trainer.action,
                              action_noise=self.trainer.get_exploration_noise(),
                              rewards=self.trainer.rewards)

            self.agents[0] = agent

        action = agent.get_action(obs, proc)

        if len(agent.trajectory) > 1:
            self.trainer.replay_memory.store(agent.get_last_experience())
            self.trainer.train_on_batch()

        if obs.terminal:
            self.stats.despawning_agent(self.agents[0])
            self.agents[0] = None

            return np.array([0, 0])

        return action

    def act_eval(self, obs):
        proc = self.observation_processing(obs)

        agent = self.agents[0]
        if not agent:
            if self.debug:
                agent = DebugAgent(actor=self.trainer.get_actor(), action=self.trainer.action,
                                   action_noise=self.trainer.get_no_noise(),
                                   rewards=self.trainer.rewards, trainer=self.trainer)
            else:
                agent = Agent(actor=self.trainer.get_actor(), action=self.trainer.action,
                              action_noise=self.trainer.get_no_noise(),
                              rewards=self.trainer.rewards)

            self.agents[0] = agent

        action = agent.get_action(obs, proc)

        if obs.terminal:
            self.stats.despawning_agent(self.agents[0])
            self.agents[0] = None

            return np.array([0, 0])

        return action
