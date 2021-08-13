import copy

from src.h3DQN_FR1.Agent import H_DDQNAgentParams, H_DDQNAgent
from src.CPP.Display import CPPDisplay
from src.h_CPP.Grid import CPPGrid, CPPGridParams
from src.h_CPP.Physics import CPPPhysics, CPPPhysicsParams
from src.h_CPP.State import CPPState
from src.h_CPP.Rewards import CPPRewardParams, CPPRewards

from src.h3DQN_FR1.Trainer import H_DDQNTrainerParams, H_DDQNTrainer
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class h_CPPEnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = CPPGridParams()
        self.reward_params = CPPRewardParams()
        self.trainer_params = H_DDQNTrainerParams()
        self.agent_params = H_DDQNAgentParams()
        self.physics_params = CPPPhysicsParams()


class h_CPPEnvironment(BaseEnvironment):

    def __init__(self, params: h_CPPEnvironmentParams):
        self.display = CPPDisplay()
        super().__init__(params, self.display)

        self.grid = CPPGrid(params.grid_params, self.stats)
        self.rewards = CPPRewards(params.reward_params, stats=self.stats)
        self.physics = CPPPhysics(params=params.physics_params, stats=self.stats)
        self.agent = H_DDQNAgent(params.agent_params, self.grid.get_example_state(), self.physics.get_example_action(),
                               stats=self.stats)
        self.trainer = H_DDQNTrainer(params=params.trainer_params, agent=self.agent)