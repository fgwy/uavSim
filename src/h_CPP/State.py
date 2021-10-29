import numpy as np

from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.CPP.State import CPPState


class h_CPPScenario:
    def __init__(self):
        self.target_path = ""
        self.position_idx = 0
        self.movement_budget = 100


class h_CPPState(CPPState):
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.h_target = None
        self.initial_h_target_cell_count = 0
        self.h_coverage = 0
        self.initial_ll_movement_budget = 50
        self.current_ll_mb = 0
        self.h_terminal = False

    def reset_h_target(self, h_target):
        # TODO: create padded target of size total-map
        shape = self.get_boolean_map_shape()
        shape =
        padded_h_target = np.pad(h_target, )
        self.h_target = h_target
        self.initial_h_target_cell_count = np.sum(padded_h_target)
        self.h_coverage = np.zeros(self.h_target.shape, dtype=bool)

    def get_remaining_h_target_cells(self):
        return np.sum(self.h_target)

    def add_explored_h_target(self, view):
        self.h_target &= ~view
        self.h_coverage |= view

    def goal_ultimated(self):

        return not bool(self.h_target is not None or self.get_remaining_h_target_cells() or self.current_ll_mb >= 0)

    def reset_ll_mb(self):
        self.current_ll_mb = self.initial_ll_movement_budget

    def get_boolean_map_ll(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        ###### Pad h-target to total map #########

        padded_rest = pad_centered(self, np.concatenate([np.expand_dims(self.landing_zone, -1),
                                                         np.expand_dims(self.h_target, -1)], axis=-1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_float_map_ll(self):
        shape = list(self.get_boolean_map_ll().shape)
        shape[2] = 0
        float_map = np.zeros(tuple(shape), dtype=float)
        return float_map

    def get_float_map_ll_shape(self):
        return self.get_float_map_ll().shape

    def get_boolean_map_shape(self):
        return self.get_boolean_map_ll().shape

    def get_scalars_hl(self):
        return np.array([self.movement_budget])

    def get_scalars_ll(self):
        return np.array([self.current_ll_mb])

    def decrement_ll_mb(self):
        self.current_ll_mb -= 1

    def set_terminal_h(self, terminal):
        self.h_terminal = terminal

    def is_terminal_h(self):
        return self.h_terminal
