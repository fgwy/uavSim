import copy
import tqdm

from src.DDQN.Agent import DDQNAgentParams, DDQNAgent
from src.CPP.Display import CPPDisplay
from src.CPP.Grid import CPPGrid, CPPGridParams
from src.CPP.Physics import CPPPhysics, CPPPhysicsParams
from src.CPP.State import CPPState
from src.CPP.Rewards import CPPRewardParams, CPPRewards

from src.DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class CPPEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = CPPGridParams()
        self.reward_params = CPPRewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = CPPPhysicsParams()


class CPPEnvironment(BaseEnvironment):

    def __init__(self, params: CPPEnvironmentParams):
        self.display = CPPDisplay()
        super().__init__(params, self.display)
        self.params = params

        self.grid = CPPGrid(params.grid_params, self.stats)
        self.rewards = CPPRewards(params.reward_params, stats=self.stats)
        self.physics = CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent = DDQNAgent(params.agent_params, self.grid.get_example_state(), self.physics.get_example_action(),
                               stats=self.stats)
        self.trainer = DDQNTrainer(params=params.trainer_params, agent=self.agent)

    def init_episode(self, init_state=None, test=False):
        if self.params.agent_params.multimap:
            # print("random map!")
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

    def run(self):

        # self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode()

            self.stats.save_if_best()

        self.stats.training_ended()

    def test_episode(self, scenario=None):
        state = copy.deepcopy(self.init_episode(scenario, test=True))
        self.stats.on_episode_begin(self.episode_count)
        while not state.terminal:
            action = self.agent.get_exploitation_action_target(state)
            next_state = self.physics.step(GridActions(action))
            reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
            self.stats.add_experience((copy.deepcopy(state), action, reward, copy.deepcopy(next_state)))
            state = copy.deepcopy(next_state)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def train_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.is_terminal():
            state = self.step(state)
            self.trainer.train_agent()

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    def fill_replay_memory(self):

        while self.trainer.should_fill_replay_memory():

            state = copy.deepcopy(self.init_episode(test=True))
            while not state.terminal:
                next_state = self.step(state, random=self.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)


