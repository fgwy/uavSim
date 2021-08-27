import copy
import tqdm
import distutils.util

from src.h3DQN_FR1.Agent import H_DDQNAgentParams, H_DDQNAgent
from src.CPP.Display import CPPDisplay
from src.h_CPP.Grid import H_CPPGrid, H_CPPGridParams
from src.h_CPP.Physics import H_CPPPhysics, H_CPPPhysicsParams
from src.h_CPP.State import CPPState
from src.h_CPP.Rewards import CPPRewardParams, CPPRewards

from src.h3DQN_FR1.Trainer import H_DDQNTrainerParams, H_DDQNTrainer
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class H_CPPEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = H_CPPGridParams()
        self.reward_params = CPPRewardParams()
        self.trainer_params = H_DDQNTrainerParams()
        self.agent_params = H_DDQNAgentParams()
        self.physics_params = H_CPPPhysicsParams()


class H_CPPEnvironment(BaseEnvironment):

    def __init__(self, params: H_CPPEnvironmentParams):
        self.display = CPPDisplay()
        super().__init__(params, self.display)

        self.grid = H_CPPGrid(params.grid_params, self.stats)
        self.rewards = CPPRewards(params.reward_params, stats=self.stats)
        self.physics = H_CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent = H_DDQNAgent(params.agent_params, self.grid.get_example_state(),
                                 self.physics.get_example_action_l(), self.physics.get_example_action_h(),
                                 stats=self.stats)
        self.trainer = H_DDQNTrainer(params=params.trainer_params, agent=self.agent)

    def run_h(self):

        self.fill_replay_memory_h()

        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode_h()

            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode_h()

            self.stats.save_if_best()

        self.stats.training_ended()

    def fill_replay_memory_h(self):

        while self.trainer.should_fill_replay_memory():

            state = copy.deepcopy(self.init_episode())
            while not state.terminal:
                next_state = self.step_h(state, random=self.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)

    def train_episode_h(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.is_terminal():
            state = self.step_h(state)
            self.trainer.train_agent()

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    def step_h(self, state, random=False, exploit=False):
        if exploit:
            goal = self.agent.get_goal(state)
            state.reset_h_target(goal)
            state.reset_ll_mb()
            action = self.agent.get_exploitation_action_target(state)
            next_state = self.physics.step(GridActions(action))
            reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
            self.stats.add_experience((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))

        else:
            if state.goal_not_present():
                if random:
                    goal = self.agent.get_random_goal()
                else:
                    goal = self.agent.get_goal(state)
                state.reset_h_target(goal)
                state.reset_ll_mb()
            if random:
                action = self.agent.get_random_action()
            else:
                action = self.agent.act(state)
            next_state = self.physics.step(GridActions(action))
            reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
            self.trainer.add_experience_ll(state, action, reward, next_state)

            if state.goal_ultimated():
                self.trainer.add_experience_hl(state, action, reward, next_state)

            self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))
        self.step_count += 1
        return copy.deepcopy(next_state)

    def test_episode_h(self, scenario=None):
        state = copy.deepcopy(self.init_episode_h(scenario))
        self.stats.on_episode_begin(self.episode_count)
        while not state.terminal:
            state = self.step_h(state, exploit=True)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def test_scenario_h(self, scenario):
        state = copy.deepcopy(self.init_episode_h(scenario))
        while not state.terminal:
            state = self.step_h(state, exploit=True)

    def init_episode_h(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
        else:
            state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state

    def eval_h(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.test_episode()
            self.step_count += 1  # Increase step count so that logging works properly

            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    def eval_scenario_h(self, init_state):
        self.test_scenario_h(init_state)

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass
