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

    def reset_h_target(self, h_target):
        self.h_target = h_target
        self.initial_h_target_cell_count = np.sum(h_target)
        self.h_coverage = np.zeros(self.h_target.shape, dtype=bool)

    def get_remaining_h_target_cells(self):
        return np.sum(self.h_target)

    def add_explored_h_target(self, view):
        self.h_target &= ~view
        self.h_coverage |= view

    def goal_not_present(self):

        return not bool(self.h_target or self.get_remaining_h_target_cells())
