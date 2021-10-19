import numpy as np
import copy
from src.h3DQN_FR1.Agent import H_DDQNAgent, H_DDQNAgentParams
from src.h3DQN_FR1.Agent_HL import HL_DDQNAgent, HL_DDQNAgentParams
from src.h3DQN_FR1.Agent_LL import LL_DDQNAgent, LL_DDQNAgentParams
from src.h3DQN_FR1.Trainer import H_DDQNTrainer, H_DDQNTrainerParams
from src.h_CPP.Rewards import H_CPPRewardParams, H_CPPRewards

import tensorflow as tf

class AgentManager_Params():
    def __init__(self):
        self.hierarchical = False

class AgentManager():
    def __init__(self, params: AgentManager_Params, example_state, example_action, stats):
        self.params = params
        self.stats = stats
        self.agent_ll = LL_DDQNAgent(LL_DDQNAgentParams(), example_state, example_action[1], stats)
        self.agent_hl = HL_DDQNAgent(HL_DDQNAgentParams(), example_state, example_action[0], stats)
        self.trainer = H_DDQNTrainer(H_DDQNTrainerParams(), self.agent_ll, self.agent_hl)
        self.rewards = H_CPPRewards(H_CPPRewardParams(), self.stats)

        if self.trainer.params.load_model != "":
            print("Loading model", self.trainer.params.load_model, "for agent")
            self.agent_ll.load_weights_ll(self.trainer.params.load_model)
            self.agent_hl.load_weights_hl(self.trainer.params.load_model)


    def step(self, state=None, exploit=False, random=False):
        action_l = None
        reward_h = 0

        # print("reached step_h")
        if exploit:
            while state.goal_not_present():
                # TODO: reshape the goal in good size
                goal = self.agent_hl.get_goal(state).reshape((self.agent_ll.params.local_map_size, self.agent_ll.params.local_map_size))
                valid = self.check_valid_target(goal, state)
                if not valid:
                    reward_h = self.rewards.invalid_goal_penalty()
                else:
                    state.goal_active = True
                    state.reset_h_target(goal)
                    state.reset_ll_mb()
                    action_l = self.agent_ll.get_exploitation_action_target(state)

        else:
            if random:
                action_l = self.agent_ll.get_random_action()
            else:
                while state.goal_not_present():
                    # TODO: reshape the goal in useful size
                    print("reached new goal")
                    if random:
                        goal = self.agent_hl.get_random_goal().reshape((self.agent_ll.params.local_map_size, self.agent_ll.params.local_map_size))
                        valid = self.check_valid_target(goal, state)
                        if not valid:
                            reward_h = self.rewards.invalid_goal_penalty()
                    else:
                        goal = self.agent_hl.get_goal(state).reshape(
                            (self.agent_ll.params.local_map_size,
                             self.agent_ll.params.local_map_size)).reshape((self.agent_ll.params.local_map_size, self.agent_ll.params.local_map_size)) # TODO: reshape soon not needed
                        valid = self.check_valid_target(goal, state)
                        if not valid:
                            reward_h = self.rewards.invalid_goal_penalty()
                        else:
                            # print('##### htarget reset #############')
                            state.goal_active = True
                            state.reset_h_target(goal)
                            state.reset_ll_mb()
                            action_l = self.agent_ll.act(state)

        return action_l

    def should_fill_replay_memory(self):
        return self.trainer.should_fill_replay_memory()

    def get_random_action(self):
        return self.step(random=True)

    def act(self, state):
        return self.step(state=state)

    def calculate_reward(self, state, action, next_state):
        return self.rewards.calculate_reward_h(state, action, next_state)

    def add_experience(self, state, action, reward, next_state):
        self.trainer.add_experience_ll(state, action, reward,next_state)

    def train_agent(self):
        self.trainer.train_agent()

    def get_exploitation_action_target(self, state):
        return self.step(state, exploit=True)

    def check_valid_target(self, target_lm, state):
        # local_map = np.logical_not(local_map).astype(int)
        # print('one hot', tf.math.argmax(target_lm))
        # print(target_lm)
        total_goal = state.pad_lm_to_total_size(target_lm)
        # tt_m = total_goal
        # print(total_goal.shape)
        nfz = np.logical_not(state.no_fly_zone)
        obs = np.logical_not(state.obstacles)
        # nfz = np.logical_not(tt_m)
        # print(nfz.shape)
        valid1 = total_goal*nfz
        valid2 = total_goal * obs
        valid = not np.all(valid1==0) or not np.all(valid2==0)
        print('on obs: ', not np.all(valid2==0), 'on nfz: ', not np.all(valid1==0), 'Goal valid: ', valid, )

        if not valid:
            print('######################## Goal Not Valid ####################################')
        return valid





