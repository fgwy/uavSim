import copy
import tqdm
import numpy as np

from src.h3DQN_FR1.AgentManager import AgentManager_Params, AgentManager
from src.h_CPP.Display import h_CPPDisplay
from src.h_CPP.Grid import H_CPPGrid, H_CPPGridParams
from src.h_CPP.Physics import H_CPPPhysics, H_CPPPhysicsParams
from src.h_CPP.State import H_CPPState
from src.h_CPP.Rewards import H_CPPRewardParams, H_CPPRewards

from src.h3DQN_FR1.Trainer import H_DDQNTrainerParams, H_DDQNTrainer
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class H_CPPEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = H_CPPGridParams()
        self.reward_params = H_CPPRewardParams()
        # self.trainer_params = H_DDQNTrainerParams()
        self.agent_params = AgentManager_Params()
        self.physics_params = H_CPPPhysicsParams()




class H_CPPEnvironment(BaseEnvironment):

    def __init__(self, params: H_CPPEnvironmentParams):
        self.display = h_CPPDisplay()
        super().__init__(params, self.display)
        self.params = params
        self.grid = H_CPPGrid(params.grid_params, self.stats)
        self.rewards = H_CPPRewards(params.reward_params, stats=self.stats)
        self.physics = H_CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent = AgentManager(params.agent_params, example_state=self.grid.get_example_state(),
                                  example_action=self.physics.get_example_action(),
                                  stats=self.stats)
        self.draw = True
        # self.random = False

    def run(self):
        # self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)
        bar = tqdm.tqdm(total=int(self.agent.trainer.params.num_steps))
        last_step = 0

        while self.step_count < self.agent.trainer.params.num_steps:
            # print('########### starting new episode ##########')
            print(f'episode count: {self.episode_count}, eval period: {self.agent.trainer.params.eval_period}, draw: {self.draw}')
            state = copy.deepcopy(self.init_episode())
            self.stats.on_episode_begin(self.episode_count)

            ## run_MDP
            # test = True if self.episode_count % self.agent.trainer.params.eval_period == 0 else False #  and self.episode_count != 0
            test = True if self.episode_count % self.agent.trainer.params.eval_period == 0 and self.episode_count != 0 else False
            self.run_MDP(state, last_step, bar, test=test)

            self.stats.on_episode_end(self.episode_count)
            self.stats.log_training_data(step=self.step_count)

            self.episode_count += 1
        self.stats.training_ended()

    def run_MDP(self, state, last_step, bar, test=False, prefill=False, random=False):
        """
        Runs MDP: High level interaction loop
        """

        while not state.is_terminal():
            goal, goal_idx, try_landing = self.agent.generate_goal(state)
            # print(f'fresh goal count: {np.sum(goal)}')
            state = copy.deepcopy(self.physics.reset_h_target(goal))
            # print(f'fresh goal count: {state.get_remaining_h_target_cells()}')
            valid = self.agent.check_valid_target(copy.deepcopy(state))

            state_h = copy.deepcopy(state)
            # print('check same shape:', state.target.shape, state.h_target.shape)

            ## sub MDP
            reward_h, next_state, last_step = self.run_subMDP(state, valid, bar, last_step, try_landing, test=test, random=random, prefill=prefill)

            if not test:
                reward_h += self.agent.rewards.calculate_reward_h(state_h, goal_idx, next_state, valid, reward_h)

                self.agent.trainer.add_experience_hl(state_h, goal_idx, reward_h, next_state)
                self.agent.trainer.train_h()

            self.step_count += 1

            state = copy.deepcopy(next_state)

    def run_subMDP(self, state, valid, bar, last_step, try_landing, test=False, random=False, prefill=False):
        """
        Runs subMDP: low-level interaction loop
        """

        i = 0
        reward_h = 0
        # print(f'######### hterminal? {state.is_terminal_h()}')
        while not state.is_terminal_h() and not state.is_terminal():
            i += 1
            if test and self.draw:
                self.display.plot_map(copy.deepcopy(state))

            if try_landing:
                # print('########## tried landing!')
                action = 4  # Landing action
                next_state = copy.deepcopy(self.physics.step(GridActions(action)))
                bar.update(self.step_count - last_step)
                last_step = self.step_count
            elif not valid:
                # print('###### invalid')
                action = 5  # Hover such that state changes (mb is decreased and different goal generated)
                self.physics.step(GridActions(action))
                next_state = copy.deepcopy(self.physics.set_terminal_h(True))
                bar.update(self.step_count - last_step)
                last_step = self.step_count

            else:
                bar.update(self.step_count - last_step)
                last_step = self.step_count

                action = self.agent.act_l(state, i, exploit=test, use_astar=self.agent.trainer.params.use_astar, random=random)
                next_state = self.physics.step(GridActions(action))
                # if next_state.get_remaining_h_target_cells() == 0:
                    # print(f'Subgoal reached!!!!  {next_state.get_remaining_h_target_cells()}')
                # print(f"step {i} in Sub MDP, \n ############ current ll_mb:{next_state.current_ll_mb} \n ########### current mb: {next_state.movement_budget}")
                reward = self.agent.rewards.calculate_reward_l(state, GridActions(action), next_state)
                if not test and not self.agent.trainer.params.use_astar:
                    # print('training llag')
                    self.agent.trainer.add_experience_ll(state, action, reward, next_state)
                    self.agent.trainer.train_l()

                self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))  # TODO Check
            reward_h += self.agent.rewards.calculate_reward_h_per_step(state, GridActions(action), next_state,
                                                                       valid)

            if test and self.draw:
                    self.display.plot_map(copy.deepcopy(next_state), next_state.is_terminal())
            # print(
                # f"step {i} in Sub MDP, \n ############ current ll_mb:{next_state.current_ll_mb} \n ########### current mb: {next_state.movement_budget}")

            state = copy.deepcopy(next_state)

        return reward_h, state, last_step


    def fill_replay_memory(self):

        while self.agent.should_fill_replay_memory():

            state = copy.deepcopy(self.init_episode())
            while not state.terminal:
                next_state = self.step(state, random=self.agent.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)



    # def init_episode(self, init_state=None):
    #     if init_state:
    #         state = copy.deepcopy(self.grid.init_scenario(init_state))
    #     else:
    #         state = copy.deepcopy(self.grid.init_episode())
    #
    #     self.rewards.reset()
    #     self.physics.reset(state)
    #     return state
