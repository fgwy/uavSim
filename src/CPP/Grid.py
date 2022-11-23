import numpy as np

import src.Map.Map as ImageLoader
from src.CPP.State import CPPState, CPPScenario
from src.CPP.RandomTargetGenerator import RandomTargetGenerator, RandomTargetGeneratorParams
from src.base.BaseGrid import BaseGrid, BaseGridParams

import src.Map.Map as Map


class CPPGridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.generator_params = RandomTargetGeneratorParams()
        self.train_map_set = ['res/manhattan32.png',
                              'res/urban50.png',
                              'res/easy50.png',
                              'res/barrier50.png']
        self.test_map_set = ['']


class CPPGrid(BaseGrid):

    def __init__(self, params: CPPGridParams, stats, diagonal=False):
        super().__init__(params, stats)
        self.diagonal = diagonal
        self.params = params

        self.generator = RandomTargetGenerator(params.generator_params, self.map_image.get_size())
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

    def init_episode(self):
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

        state = CPPState(self.map_image, self.diagonal)
        state.reset_target(self.target_zone)

        idx = np.random.randint(0, len(self.starting_vector))
        state.position = self.starting_vector[idx]

        state.movement_budget = np.random.randint(low=self.params.movement_range[0],
                                                  high=self.params.movement_range[1] + 1)

        state.initial_movement_budget = state.movement_budget
        state.landed = False
        state.terminal = False

        return state

    def create_scenario(self, scenario: CPPScenario):
        state = CPPState(self.map_image)
        target = ImageLoader.load_target(scenario.target_path, self.map_image.obstacles)
        state.reset_target(target)
        state.position = self.starting_vector[scenario.position_idx]
        state.movement_budget = scenario.movement_budget
        state.initial_movement_budget = scenario.movement_budget
        return state

    def init_scenario(self, state: CPPState):
        self.target_zone = state.target

        return state

    def get_example_state(self):
        state = CPPState(self.map_image, False)
        state.reset_target(self.target_zone)
        state.position = [0, 0]
        state.movement_budget = 0
        state.initial_movement_budget = 0
        return state

    def get_target_zone(self):
        return self.target_zone

    def random_new_map_image_train(self, test):
        a = np.random.randint(0, len(self.params.train_map_set)) if not test else np.random.randint(0, len(self.params.test_map_set))
        self.map_image = Map.load_map(self.params.train_map_set[np.random.randint(0, len(self.params.train_map_set))]) if not test else Map.load_map(self.params.test_map_set[np.random.randint(0, len(self.params.test_map_set))])
        self.shape = self.map_image.start_land_zone.shape
        self.generator.update_shape(self.shape)
        self.starting_vector = self.map_image.get_starting_vector()
        self.stats.set_env_map_callback(self.get_map_image)
        state = self.init_episode()
        return state, self.params.train_map_set[a]
    #
    # def random_new_map_image_test(self):
    #     a = np.random.randint(0, len(self.params.test_map_set))
    #     self.map_image = Map.load_map(self.params.test_map_set[a])
    #     self.shape = self.map_image.start_land_zone.shape
    #     self.generator.update_shape(self.shape)
    #     self.starting_vector = self.map_image.get_starting_vector()
    #     self.stats.set_env_map_callback(self.get_map_image)
    #     state = self.init_episode()
    #     return state, self.params.test_map_set[a]
