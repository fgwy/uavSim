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
            state = self.step(state)
            self.agent.train_agent()

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    def run(self):

        self.fill_replay_memory()

        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.agent.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.agent.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            if self.episode_count % self.agent.trainer.params.eval_period == 0:
                self.test_episode()

            self.stats.save_if_best()

        self.stats.training_ended()

    def step(self, state, random=False, test=False):
        if random:
            action = self.agent.get_random_action()
        else:
            action = self.agent.act(state)
        next_state = self.physics.step(GridActions(action))
        reward = self.agent.calculate_reward(state, GridActions(action), next_state)
        self.agent.add_experience(state, action, reward, next_state)
        self.stats.add_experience((state, action, reward, copy.deepcopy(next_state)))
        self.step_count += 1
        return copy.deepcopy(next_state)

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
