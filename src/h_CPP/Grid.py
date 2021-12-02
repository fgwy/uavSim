import numpy as np

import src.Map.Map as ImageLoader
from src.h_CPP.State import H_CPPScenario
from src.h_CPP.State import H_CPPState
from src.CPP.RandomTargetGenerator import RandomTargetGenerator, RandomTargetGeneratorParams
from src.base.BaseGrid import BaseGrid, BaseGridParams


class H_CPPGridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.generator_params = RandomTargetGeneratorParams()
        self.local_map_size = 17


class H_CPPGrid(BaseGrid):

    def __init__(self, params: H_CPPGridParams, stats):
        super().__init__(params, stats)
        self.params = params

        self.generator = RandomTargetGenerator(params.generator_params, self.map_image.get_size())
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

    def init_episode(self):
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

        state = H_CPPState(self.map_image)
        state.reset_target(self.target_zone)

        idx = np.random.randint(0, len(self.starting_vector))
        state.position = self.starting_vector[idx]

        state.movement_budget = np.random.randint(low=self.params.movement_range[0],
                                                  high=self.params.movement_range[1] + 1)

        # state.current_ll_mb = 50 #np.random.randint(low=self.params.movement_range_ll[0],
                                                  #high=self.params.movement_range_ll[1] + 1)

        state.initial_movement_budget = state.movement_budget
        # state.reset_h_target(np.zeros((self.params.local_map_size, self.params.local_map_size)))
        # state.initial_ll_movement_budget = state.current_ll_mb
        state.reset_h_target(np.zeros_like(self.target_zone))
        state.landed = False
        state.terminal = False

        return state

    def create_scenario(self, scenario: H_CPPScenario):
        state = H_CPPState(self.map_image)
        target = ImageLoader.load_target(scenario.target_path, self.map_image.obstacles)
        state.reset_target(target)
        state.position = self.starting_vector[scenario.position_idx]
        state.movement_budget = scenario.movement_budget
        state.initial_movement_budget = scenario.movement_budget
        state.current_ll_mb = scenario.ll_movement_budget
        return state

    def init_scenario(self, state: H_CPPState):
        self.target_zone = state.target

        return state

    def get_example_state(self):
        state = H_CPPState(self.map_image)
        state.reset_target(self.target_zone)
        # state.reset_h_target(np.zeros((self.params.local_map_size, self.params.local_map_size)))
        state.reset_h_target(np.zeros_like(self.target_zone))
        state.position = [0, 0]
        state.movement_budget = self.params.movement_range[1]
        state.initial_movement_budget = self.params.movement_range[1]
        return state

    def get_target_zone(self):
        return self.target_zone
