import numpy as np

import src.Map.Map as ImageLoader
from src.h_CPP.State import H_CPPScenario
from src.h_CPP.State import H_CPPState
from src.CPP.RandomTargetGenerator import RandomTargetGenerator, RandomTargetGeneratorParams
from src.base.BaseGrid import BaseGrid, BaseGridParams

import src.Map.Map as Map


class H_CPPGridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.generator_params = RandomTargetGeneratorParams()
        self.local_map_size = 17
        self.train_map_set = ['res/manhattan32.png',
                              'res/urban50.png',
                              'res/easy50.png',
                              'res/barrier50.png',
                              'res/center60.png',
                              'res/maze60.png',
                              'res/smiley60.png'
                              ]
        self.test_map_set = ['']


class H_CPPGrid(BaseGrid):

    def __init__(self, params: H_CPPGridParams, stats, diagonal=False):
        super().__init__(params, stats)
        self.diagonal = diagonal
        self.params = params

        self.generator = RandomTargetGenerator(params.generator_params, self.map_image.get_size())
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

    def init_episode(self):
        self.generator.update_shape(self.map_image.get_size())
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

        state = H_CPPState(self.map_image, self.diagonal)
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
        state.reset_target_h(np.zeros_like(self.target_zone))
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
        state = H_CPPState(self.map_image, True)
        state.reset_target(self.target_zone)
        # state.reset_h_target(np.zeros((self.params.local_map_size, self.params.local_map_size)))
        state.reset_target_h(np.zeros_like(self.target_zone))
        state.position = [0, 0]
        state.movement_budget = self.params.movement_range[1]
        state.initial_movement_budget = self.params.movement_range[1]
        return state

    def get_target_zone(self):
        return self.target_zone


    def random_new_map_image_train(self):
        a = np.random.randint(0, len(self.params.train_map_set))
        self.map_image = Map.load_map(self.params.train_map_set[a])
        self.shape = self.map_image.start_land_zone.shape
        self.generator.update_shape(self.shape)
        self.starting_vector = self.map_image.get_starting_vector()
        self.stats.set_env_map_callback(self.get_map_image)
        state = self.init_episode()
        return state, self.params.train_map_set[a]

    def random_new_map_image_test(self):
        a = np.random.randint(0, len(self.params.test_map_set))
        self.map_image = Map.load_map(self.params.test_map_set[a])
        self.shape = self.map_image.start_land_zone.shape
        self.generator.update_shape(self.shape)
        self.starting_vector = self.map_image.get_starting_vector()
        self.stats.set_env_map_callback(self.get_map_image)
        state = self.init_episode()
        return state, self.params.test_map_set[a]
