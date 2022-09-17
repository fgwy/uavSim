import copy
import tqdm
import numpy as np

from src.H2D2.AgentManager import AgentManager_Params, AgentManager
from src.h_CPP.Display import h_CPPDisplay
from src.h_CPP.Grid import H_CPPGrid, H_CPPGridParams
from src.h_CPP.Physics import H_CPPPhysics, H_CPPPhysicsParams
from src.h_CPP.State import H_CPPState
from src.h_CPP.Rewards import H_CPPRewardParams, H_CPPRewards

from src.H2D2.Trainer import H_DDQNTrainerParams, H_DDQNTrainer
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions, GridActionsDiagonal


class H_CPPEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = H_CPPGridParams()
        self.reward_params = H_CPPRewardParams()
        # self.trainer_params = H_DDQNTrainerParams()
        self.agent_params = AgentManager_Params()
        self.physics_params = H_CPPPhysicsParams()


class H_CPPEnvironment(BaseEnvironment):

    def __init__(self, params: H_CPPEnvironmentParams, eval=False):
        self.display = h_CPPDisplay()
        super().__init__(params, self.display)
        self.params = params

        self.grid = H_CPPGrid(params.grid_params, self.stats)
        self.example_state = self.grid.get_example_state()
        self.rewards = H_CPPRewards(params.reward_params, stats=self.stats)
        self.physics = H_CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent_manager = AgentManager(params.agent_params, camera=self.physics.camera,
                                          example_state=self.example_state,
                                          example_action=self.physics.get_example_action(),
                                          stats=self.stats)
        self.eval = self.agent_manager.params.eval
        self.actions = GridActions if not self.params.agent_params.diagonal else GridActionsDiagonal
        # self.display.init_plt(self.example_state)
        # super().__init__(params, self.display)
        self.hl_steps = 0

        # self.random = False

    def run(self):
        # self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)
        if self.eval:
            max_steps = 400000
        else:
            max_steps = self.agent_manager.trainer.params.num_steps
        bar = tqdm.tqdm(total=int(max_steps))
        last_step = 0
        w = 0
        steps_ins_mdp = 0

        # while self.step_count < self.agent_manager.trainer.params.num_steps:
        while self.hl_steps < max_steps:
        # while True:
            # print(w)
            w+=1
            if self.episode_count%10==0:
                print(
                    f'\nepisode count: {self.episode_count}, steps in mdp: {steps_ins_mdp} eval period: {self.agent_manager.trainer.params.eval_period}, draw: {self.stats.params.draw}')
            state = copy.deepcopy(self.init_episode())
            self.stats.on_episode_begin(self.episode_count)

            # print(f'initial movement budget: {state.initial_movement_budget}')

            ## run_MDP
            # test = True if self.episode_count % self.agent.trainer.params.eval_period == 0 else False #  and self.episode_count != 0
            test = True if self.episode_count % self.agent_manager.trainer.params.eval_period == 0 and self.episode_count != 0 else False
            # test = False
            steps_ins_mdp = self.run_MDP(last_step, bar, episode_num=self.episode_count, test=test, random_h=self.agent_manager.params.pretrain_ll)

            self.stats.on_episode_end(self.episode_count)
            self.stats.log_training_data(step=self.hl_steps)
            self.stats.save_if_best()

            self.episode_count += 1

        self.agent_manager.save_weights(self.stats.params.save_model + 'end')
        self.agent_manager.save_models(self.stats.params.save_model + 'end')
        self.stats.training_ended()

    def run_MDP(self, last_step, bar, episode_num, test=False, prefill=False, random_h=False, random_l=False):
        """
        Runs MDP: High level interaction loop
        """
        display_trajectory = []
        display_trajectory.append(copy.deepcopy(self.physics.state))
        
        cumulative_reward_h = 0
        tried_landing_and_succeeded = False
        i = 0
        while not self.physics.state.is_terminal():
            # terminal = self.physics.state.is_terminal()
            # terminal_h = self.physics.state.is_terminal_h()
            i+=1
            # bar.update(self.hl_steps - last_step)  # todo check this insanity
            bar.update(1)
            last_step = self.hl_steps
            goal, goal_idx, try_landing, q = self.agent_manager.generate_goal(self.physics.state, random=random_h,
                                                                           exploit=(test or self.agent_manager.params.eval_exploit))
            # if self.hl_steps %1000 == 0:
            #     print(f'sum qval = {sum(q[0])}')

            # print(f'goal before padding: {np.sum(goal * 1)}')
            goal = self.physics.state.pad_lm_to_total_size(goal)
            # print(f'goal before preproc env: {np.sum(goal * 1)}')
            # goal = self.agent_manager.preproc_goal(goal, copy.deepcopy(self.physics.state))
            # print(f'goal before resetting: {np.sum(goal * 1)}')
            state = self.physics.reset_h_target(goal)
            # print(f'goal before chack validity: {np.sum(state.h_target * 1)}')
            if try_landing:
                valid = True
            else:
                valid = self.agent_manager.check_valid_target(self.physics.state, test) or try_landing

            state_h = copy.deepcopy(self.physics.state)

            ## sub MDP
            tried_landing_and_succeeded, last_step, display_trajectory = self.run_subMDP(valid, bar,
                                                                                         last_step,
                                                                                         try_landing,
                                                                                         display_trajectory,
                                                                                         test=test,
                                                                                         random_l=random_l,
                                                                                         prefill=prefill)

            reward_h = self.rewards.calculate_reward_h(state_h, goal_idx, self.physics.state, valid,
                                                       tried_landing_and_succeeded)

            cumulative_reward_h += reward_h

            if not test:
                self.hl_steps += 1

            # print(f'reward_h: {reward_h}')

            if not test and not self.agent_manager.params.pretrain_ll and not self.eval:

                self.agent_manager.trainer.add_experience_hl(state_h, goal_idx, reward_h,
                                                             copy.deepcopy(self.physics.state))
                self.agent_manager.trainer.train_h()
            self.stats.add_experience((state, goal_idx, reward_h, copy.deepcopy(self.physics.state)))

        # self.stats.on_episode_end(self.episode_count)
        # self.stats.log_training_data(step=self.step_count)

        # print(f'H-Steps in episode: {i}')
        # if test or episode_num % (self.agent_manager.trainer.params.eval_period - 1) == 0 or tried_landing_and_succeeded and not self.eval:
        # if True:
        #     self.display.save_plot_map(trajectory=display_trajectory, episode_num=episode_num, testing=test,
        #                                name=self.stats.params.log_file_name, las=tried_landing_and_succeeded, cum_rew=cumulative_reward_h, hl_steps=self.hl_steps)
            # self.display.save_state_and_hm(state=state_h, q_vals=q, name=self.stats.params.log_file_name, episode_num=episode_num)
            # pass

        if test and not self.eval:
            self.agent_manager.save_weights(self.stats.params.save_model + f'/{self.stats.params.log_file_name}/{episode_num}/')
            self.agent_manager.save_models(self.stats.params.save_model + f'/{self.stats.params.log_file_name}/{episode_num}/')

        return i

    def run_subMDP(self, valid, bar, last_step, try_landing, display_trajectory, test=False, random_l=False,
                   prefill=False):
        """
        Runs subMDP: low-level interaction loop
        """
        i = 0
        tried_landing_and_succeeded = None
        while not self.physics.state.is_terminal_h() and not self.physics.state.is_terminal():
            i += 1
            state = copy.deepcopy(self.physics.state)
            # bar.update(self.step_count - last_step)  # todo check this insanity
            # last_step = self.step_count
            if try_landing:
                # todo: check!! enters twice before landing
                print('tried landing')
                action = self.actions.LAND  # Landing action
                next_state = self.physics.step(self.actions(action))
                # next_state = self.physics.set_terminal_h(True)
                tried_landing_and_succeeded = next_state.landed
                if tried_landing_and_succeeded:
                    print(f'########## tried landing and succeded {tried_landing_and_succeeded}! action: {self.actions.LAND}')
            elif not valid:
                # print('invalid!')
                action = 5  # Hover such that state changes (mb is decreased and different goal generated)
                self.physics.step(self.actions(action))
                next_state = self.physics.set_terminal_h(True)
            else:

                action = self.agent_manager.act_l(self.physics.state, i, exploit=test,
                                                  use_astar=self.agent_manager.trainer.params.use_astar,
                                                  random=random_l)
                next_state = self.physics.step(self.actions(action))
                reward = self.rewards.calculate_reward_l(state, self.actions(action), next_state)
                # print(f'reward_l: {reward}')
                if not test and not self.agent_manager.trainer.params.use_astar and not self.eval:
                    self.agent_manager.trainer.add_experience_ll(state, action, reward, next_state)
                    self.agent_manager.trainer.train_l()

            print(self.physics.state.is_terminal, )

                # self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))  # TODO Check
            if test and self.stats.params.draw:
                self.display.plot_map(copy.deepcopy(next_state), next_state.is_terminal())
            display_trajectory.append(copy.deepcopy(self.physics.state))
            # if next_state.h_terminal or next_state.goal_covered:
            #     print(f'next state term?terminal: {next_state.is_terminal()} terminal_h: {next_state.h_terminal} goal active: {next_state.goal_active} goal_covered: {next_state.goal_covered}')
            # self.step_count += 1
        # print(f'how many setps in smdp: {i}')
        return tried_landing_and_succeeded, last_step, display_trajectory

    def fill_replay_memory(self):

        while self.agent_manager.should_fill_replay_memory():

            state = copy.deepcopy(self.init_episode())
            while not state.terminal:
                next_state = self.step(state, random=self.agent_manager.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)

    def init_episode(self, init_state=None, test=False):
        if self.params.agent_params.hl_agent.multimap:
            if test:
                state, path = copy.deepcopy(self.grid.random_new_map_image_test())
                self.physics.reset_camera(path)
            else:
                state, path = copy.deepcopy(self.grid.random_new_map_image_train())
                self.physics.reset_camera(path)
        else:
            if init_state:
                state = copy.deepcopy(self.grid.init_scenario(init_state))
            else:
                state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state
