import copy
import tqdm

from src.h3DQN_FR1.AgentManager import AgentManager_Params, AgentManager
from src.CPP.Display import CPPDisplay
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
        self.display = CPPDisplay()
        super().__init__(params, self.display)

        self.grid = H_CPPGrid(params.grid_params, self.stats)
        self.rewards = H_CPPRewards(params.reward_params, stats=self.stats)
        self.physics = H_CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent = AgentManager(params.agent_params, example_state=self.grid.get_example_state(),
                                  example_action=self.physics.get_example_action(),
                                  stats=self.stats)

    def train_episode(self):
        '''
        function trains one episode for given agent
        '''
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.is_terminal():
            goal = self.agent.generate_goal(state)
            state.reset_h_target(goal)
            while not state.is_terminal_h():
                state = self.agent.act(state)
                self.agent.train_agent()

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    # def run(self):
    #
    #     self.fill_replay_memory()
    #
    #     print('Running ', self.stats.params.log_file_name)
    #
    #     bar = tqdm.tqdm(total=int(self.agent.trainer.params.num_steps))
    #     last_step = 0
    #     while self.step_count < self.agent.trainer.params.num_steps:
    #         bar.update(self.step_count - last_step)
    #         last_step = self.step_count
    #         self.train_episode()
    #
    #         if self.episode_count % self.agent.trainer.params.eval_period == 0:
    #             self.test_episode()
    #
    #         self.stats.save_if_best()
    #
    #     self.stats.training_ended()

    def run(self):
        # self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.agent.trainer.params.num_steps))
        last_step = 0
        goal = None
        while self.step_count < self.agent.trainer.params.num_steps:
            print('########### starting new episode ##########')

            state = copy.deepcopy(self.init_episode())
            self.stats.on_episode_begin(self.episode_count)
            current_llmb = state.current_ll_mb
            while not state.is_terminal():

                reward_h = 0
                if goal is not None:
                    old_goal = goal
                goal, goal_idx = self.agent.generate_goal(state)
                state.reset_h_target(goal)
                valid = self.agent.check_valid_target(goal, state)
                if not valid:
                    print('###### invalid')
                    action = 5
                    next_state = self.physics.step(GridActions(action))
                    state.h_terminal = True
                    # continue
                else:
                    state.h_terminal = False
                state_h = copy.deepcopy(state)
                i = 0
                print(
                    f"############ current ll_mb:{state.current_ll_mb} \n ########### current mb: {state.movement_budget}")
                while not state.is_terminal_h():
                    print(f'################ terminal h? :{state.is_terminal_h}')
                    i += 1
                    print(f"step {i} in Sub MDP, ############ current ll_mb:{state.current_ll_mb}")
                    bar.update(self.step_count - last_step)
                    last_step = self.step_count
                    action = self.agent.act_l(state)
                    next_state = self.physics.step(GridActions(action))

                    reward = self.agent.rewards.calculate_reward_l(state, GridActions(action), next_state)
                    reward_h += self.agent.rewards.calculate_reward_h_per_step(state, GridActions(action), next_state,
                                                                               valid)
                    self.agent.trainer.add_experience_ll(state, action, reward, next_state)

                    self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))  # TODO Check
                    self.agent.train_l()
                    # if self.agent.trainer.is_h_terminated(state, next_state):
                    if state.is_terminal_h:
                        print('############# sub mdp terminated', self.agent.trainer.is_h_terminated(state, next_state))
                        next_state.goal_terminated()
                    self.agent.trainer.train_l()
                    state = copy.deepcopy(next_state)
                    # state = next_state
                reward_h += self.agent.rewards.calculate_reward_h(state_h, goal_idx, next_state, valid)

                self.agent.trainer.add_experience_hl(state_h, goal_idx, reward_h, next_state)
                self.agent.trainer.train_h()
                state.set_terminal_h(False)
                self.step_count += 1

            self.stats.on_episode_end(self.episode_count)
            self.stats.log_training_data(step=self.step_count)

            self.episode_count += 1

    # def step(self, state, random=False, test=False):
    #     action = self.agent.act_l(state, random)  # TODO: return both actions, hover ll if goal invalid
    #     next_state = self.physics.step(GridActions(action))
    #     reward = self.agent.calculate_reward(state, GridActions(action),
    #                                          next_state)  # TODO: calculate both rewards, check coverage and set goal_active in state
    #     self.agent.add_experience(state, action, reward, next_state)  # TODO: Check which experience to add
    #     self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))
    #     self.step_count += 1
    #     return copy.deepcopy(next_state)

    def test_episode(self, scenario=None):
        state = copy.deepcopy(self.init_episode(scenario))
        self.stats.on_episode_begin(self.episode_count)
        while not state.terminal:
            action = self.agent.get_exploitation_action_target(state)
            next_state = self.physics.step(GridActions(action))
            reward = self.agent.calculate_reward(state, GridActions(action), next_state)
            self.stats.add_experience((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
            state = copy.deepcopy(next_state)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def fill_replay_memory(self):

        while self.agent.should_fill_replay_memory():

            state = copy.deepcopy(self.init_episode())
            while not state.terminal:
                next_state = self.step(state, random=self.agent.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)
