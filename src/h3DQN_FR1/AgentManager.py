import numpy as np
import copy
from src.h3DQN_FR1.Agent import H_DDQNAgent, H_DDQNAgentParams
from src.h3DQN_FR1.PPO_Agent import PPOAgent, PPOAgentParams
from src.h3DQN_FR1.Agent_HL import HL_DDQNAgent, HL_DDQNAgentParams
from src.h3DQN_FR1.Agent_LL import LL_DDQNAgent, LL_DDQNAgentParams
from src.h3DQN_FR1.Trainer import H_DDQNTrainer, H_DDQNTrainerParams
from src.h_CPP.Rewards import H_CPPRewardParams, H_CPPRewards
from src.a_star.A_star import A_star

import tensorflow as tf


class AgentManager_Params():
    def __init__(self):
        self.hierarchical = True
        self.use_soft_max = True
        self.pretrain_ll = False
        self.use_ddqn = True
        self.use_ddpg = False
        self.use_ppo = False
        self.h_trainer = H_DDQNTrainerParams()
        self.ll_agent = LL_DDQNAgentParams()
        self.hl_agent = HL_DDQNAgentParams()


class AgentManager():
    def __init__(self, params: AgentManager_Params, camera, example_state, example_action, stats):
        self.params = params
        self.stats = stats
        self.trainer_params = params.h_trainer
        self.ll_agent_params = params.ll_agent
        self.hl_agent_params = params.hl_agent
        self.agent_ll = LL_DDQNAgent(self.ll_agent_params, example_state, example_action[1], stats)
        if self.params.use_ppo:
            self.agent_hl = PPOAgent(self.hl_agent_params, example_state, example_action[0], stats)
        elif self.params.use_ddpg:
            pass
        elif self.params.use_ddqn:
            self.agent_hl = HL_DDQNAgent(self.hl_agent_params, example_state, example_action[0], stats)

        self.trainer = H_DDQNTrainer(self.trainer_params, self.agent_ll, self.agent_hl)
        # self.rewards = H_CPPRewards(H_CPPRewardParams(), self.stats)
        self.astar = A_star()

        if self.params.use_ppo or self.params.use_ddpg:
            self.multigoal = True
        else:
            self.multigoal = False

        self.next_state_hl = None
        self.state_hl = None
        self.current_goal_idx = None
        self.current_goal = None
        self.camera = camera

        if self.trainer.params.load_model != "":
            print("Loading model", self.trainer.params.load_model, "for agent")
            self.agent_ll.load_weights_ll(self.trainer.params.load_model)
            self.agent_hl.load_weights_hl(self.trainer.params.load_model)

    # def step(self, state=None, exploit=False, random=False):
    #     action_l = None
    #     reward_h = 0
    #     if self.current_goal is not None:
    #         self.old_goal = copy.deepcopy(self.current_goal)
    #     if state.is_terminal_h:
    #         pass
    #     if state.goal_not_active():
    #         new_goal = self.generate_goal(state, exploit, random)
    #         # self.current_goal = goal_array
    #         state.goal_active = True
    #         state.reset_h_target(new_goal)
    #         state.reset_ll_mb()
    #         if not self.check_valid_target(new_goal, state):
    #             # Set ll action to Hover if goal is invalid
    #             action_l = [5]
    #         else:
    #             action_l = self.agent_ll.act(state)
    #
    #     return action_l

    def should_fill_replay_memory(self):
        return self.trainer.should_fill_replay_memory()

    # def act(self, state, random=False):
    #     if random:
    #         return self.step(state=state, random=random)
    #     else:
    #         return self.step(state=state)

    # def calculate_reward(self, state, action, next_state):
    #     reward_h = self.rewards.calculate_reward_h(state, action, next_state)
    #     reward_l = self.rewards.calculate_reward_l(state, action, next_state)
    #     return

    # def add_experience(self, state, action, reward, next_state):
    #     valid = self.check_valid_target(self.current_goal,
    #                                     state)  # TODO: Check validity of goal and adjust goal_active variable
    #     if not valid:
    #         reward_h = self.rewards.invalid_goal_penalty()
    #         # TODO: generate experience for HLagent
    #         # TODO: Flag goal as inactive
    #     else:
    #         self.trainer.add_experience_ll(state, action, reward, next_state)
    #
    #     # TODO: Make pretty and also working
    #     if self.old_goal is not None:
    #         # Add experience here because it's better
    #         self.old_state_hl = copy.deepcopy(state)
    #         self.last_hl_action = copy.deepcopy(self.current_goal_idx)
    #         self.last_hl_reward = reward_h
    #     self.high_level_old_state = copy.deepcopy(state)
    #     self.last_hl_action = copy.deepcopy(self.current_goal_idx)
    #     self.last_hl_reward = reward_h

    def add_experience_hl(self, state, action, reward, next_state):
        self.trainer.add_experience_hl(state, action, reward, next_state)

    def train_l(self):
        self.trainer.train_l()

    # def get_exploitation_action_target(self, state):
    #     return self.step(state, exploit=True)

    def generate_goal(self, state=None, exploit=False, random=False):
        if self.params.use_ddqn:
            return self.generate_goal_ddqn(state, exploit, random)
        elif self.params.use_ddpg:
            return self.generate_goal_ddpg(state, exploit, random)
        elif self.params.use_ppo:
            return self.generate_goal_ppo(state, exploit, random)

    def generate_goal_ppo(self, state, exploit, random):
        pass

    def generate_goal_ddpg(self, state, exploit, random):
        pass

    def generate_goal_ddqn(self, state, exploit, random):
        try_landing = False
        q = 0
        if random:
            self.current_goal_idx = self.agent_hl.get_random_goal()
        elif exploit:
            self.current_goal_idx, q = self.agent_hl.get_exploitation_goal(state)

        else:
            if self.params.use_soft_max:
                self.current_goal_idx, q = self.agent_hl.get_softmax_goal(state)
            else:
                self.current_goal_idx = self.agent_hl.get_eps_greedy_action(state)

        # generate goal out of idx
        if self.current_goal_idx == self.agent_hl.num_actions_hl - 1:
            try_landing = True
            self.current_goal = np.zeros((self.hl_agent_params.goal_size, self.hl_agent_params.goal_size))  # todo please
        else:
            self.current_goal = tf.one_hot(self.current_goal_idx,
                                           depth=self.agent_hl.num_actions_hl - 1).numpy().reshape(self.agent_hl.params.goal_size, self.agent_hl.params.goal_size)
        # (self.agent_ll.local_map_size, self.agent_ll.local_map_size))
        # print(f'Goal idx and shape: {self.current_goal_idx} / {self.current_goal.shape} # Try landing: {try_landing}')

        return self.current_goal, self.current_goal_idx, try_landing, q

    # def preproc_goal(self, goal, state):
    #     """
    #
    #     :param state: goal already padded to total size (to enable comparison to view and
    #     :return: PROCESSED GOAL
    #     """
    #     goal = goal*1
    #     # print(f'goal before preproc: {np.sum(goal*1)}')
    #     view = self.camera.computeView(state.position, 0)
    #     # print(f'VIEW: {np.sum(view * 1)} {view.shape} {np.sum((~view) * 1)}')
    #     # print(f'VIEW: {view}')
    #     # print(f'VIEW: {~view}')
    #     # goal = goal * ((~view) * 1)
    #     # print(f'goal after VIEW: {np.sum(goal)}')
    #     # goal *= ~state.no_fly_zone * 1
    #     # goal *= (~state.obstacles) * 1
    #     # print(f'goal after preproc: {np.sum(goal*1)}')
    #     return goal.astype(bool)

    def act_l(self, state, steps_in_smdp, exploit=False, random=False, use_astar=False):
        # print('########### act_l ################')
        if random:
            return self.agent_ll.get_random_action()
        if use_astar:
            # print(f'Using A_star!')
            return self.astar.get_A_star_action(copy.deepcopy(state), steps_in_smdp)
            # return self.agent_ll.get_random_action()
        if exploit:
            return self.agent_ll.get_exploitation_action(state)
        else:
            return self.agent_ll.get_soft_max_exploration(state)

    def check_valid_target(self, state, test):
        total_goal = state.h_target * 1
        # print(f'htarget:{total_goal}')
        # if bool(np.sum(total_goal)):
        # print(f'at least one goal in total goal: {bool(np.sum(total_goal))}')
        if self.multigoal:
            inside_bounds = bool(np.sum(total_goal))
            valid = inside_bounds
        else:
            # total_goal = state.pad_lm_to_total_size(target_lm)

            # if np.sum(total_goal) == 0:
            #     print('######################## Goal Not Valid ####################################')
            #     return False
            # nfz = np.logical_not(state.no_fly_zone)
            # obs = np.logical_not(state.obstacles)

            view = self.camera.computeView(state.position, 0)
            nfz = state.no_fly_zone * 1
            obs = state.obstacles * 1
            # on_nfz = np.any(total_goal * nfz) == 1
            on_nfz = total_goal * nfz
            on_nfz = bool(np.sum(on_nfz))
            # on_obs = np.any(total_goal * obs) == 1
            on_obs = bool(np.sum(total_goal * obs))
            inside_bounds = bool(np.sum(total_goal))
            on_position = np.any(total_goal * view*1) == 1
            if on_nfz:
                print(f'on-nfz: {np.sum()} test:{test}')
            elif on_obs:
                print(f'on-obs')
            elif not inside_bounds:
                print(f'not inside bounds {np.sum(total_goal)}')
            elif on_position:
                print(f'on-position')

            valid = not on_obs and inside_bounds and not on_position and not on_nfz
            # valid = not np.all(valid1 == 0) or not np.all(valid2 == 0)
            # print(f'Goal on obs: {on_nfz} # On nfz: {on_obs} # Outside Bounds: {not inside_bounds} # Goal valid: {valid}')

            if not valid:
                # print(f'######################## Goal Not Valid ####################################')
                pass
        return valid

    def find_h_target_idx(self, state):
        end = np.where(state.h_target == 1)
        return end

    def save_models(self, path):
        if not self.trainer.params.use_astar:
            self.agent_ll.save_model_ll(path)
        if not self.params.pretrain_ll:
            self.agent_hl.save_model_hl(path)

    def save_weights(self, path):
        if not self.trainer.params.use_astar:
            self.agent_ll.save_weights_ll(path)
        if not self.params.pretrain_ll:
            self.agent_hl.save_weights_hl(path)
