import numpy as np

from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.CPP.State import CPPState


class H_CPPScenario:
    def __init__(self):
        self.target_path = ""
        self.position_idx = 0
        self.movement_budget = 100
        self.ll_movement_budget = 30


class H_CPPState(CPPState):
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.h_target = np.zeros((17, 17))
        self.initial_h_target_cell_count = 0
        self.h_coverage = 0
        self.initial_ll_movement_budget = 30
        self.current_ll_mb = None
        self.h_terminal = False
        self.goal_active = False
        self.movement_budget = 100

    def reset_h_target(self, h_target):

        # print(h_target)

        self.h_target = self.pad_lm_to_total_size(h_target)
        self.initial_h_target_cell_count = np.sum(h_target)
        print(self.initial_h_target_cell_count)
        self.h_coverage = np.zeros(self.h_target.shape, dtype=bool)
        self.reset_ll_mb()
        self.set_terminal_h(False)

    def pad_lm_to_total_size(self, h_target):
        """
        pads input of shape local_map to output of total_map_size
        """

        shape_map = self.landing_zone.shape[:2]
        shape_htarget = h_target.shape
        if self.position == None:
            x, y = 0, 0
        else:
            x, y = self.position

        # outline = (shape_htarget[0]-1)/2

        # try:
        #     no_fly_border = int((shape_htarget[0]-1)/2)
        # except ValueError:
        #     print("Division does not yield a valid integer!")

        # padding htarget to total-map shape: x and y are 0 at bottom left corner
        # TODO: check how x and y are initialized (0 at bottom left?)
        pad_left = x
        pad_right = shape_map[0] - x - 1
        pad_up = shape_map[1] - y - 1
        pad_down = y

        padded = np.pad(h_target, ((pad_up, pad_down), (pad_left, pad_right)))
        # print(int((shape_htarget[0]-1)/2), int((padded.shape[0]-(shape_htarget[0]-1)/2)))

        lm_as_tm_size = padded[int((shape_htarget[0] - 1) / 2):int(padded.shape[0] - (shape_htarget[0] - 1) / 2),
                        int((shape_htarget[1] - 1) / 2):int(padded.shape[1] - (shape_htarget[1] - 1) / 2)]

        return lm_as_tm_size.astype(bool)

    def goal_not_active(self):
        return not self.goal_active

    def get_remaining_h_target_cells(self):
        return np.sum(self.h_target)

    def add_explored_h_target(self, view):
        # print("reached")
        self.h_target &= ~view
        self.h_coverage |= view

    def goal_ultimated(self):

        return not bool(self.h_target is not None or self.get_remaining_h_target_cells() or self.current_ll_mb >= 0)

    def reset_ll_mb(self):
        # TODO: set to min(mb, ll_mb)
        self.current_ll_mb = self.initial_ll_movement_budget

    def get_boolean_map_ll(self):
        # print("reached bool map ll")
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)


        ##### use this if h_target is of shape map (h_targets outside of map are not seen by ll-ag)####
        padded_rest = pad_centered(self, np.concatenate([np.expand_dims(self.landing_zone, -1),
                                                         np.expand_dims(self.h_target, -1)], axis=-1), 0)

        ##### use this if h_target is of shape after pad_centered ###
        # padded_rest = pad_centered(self, np.expand_dims(self.landing_zone, -1), 0)
        #
        # print("\n", padded_rest.shape, "\n", padded_red.shape, "\n", self.no_fly_zone.shape)
        # padded_rest = np.concatenate([np.expand_dims(padded_rest, -1), np.expand_dims(self.h_target, -1)], axis=-1)

        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_float_map_ll(self):
        shape = list(self.get_boolean_map_ll().shape)
        shape[2] = 0
        float_map = np.zeros(tuple(shape), dtype=float)
        return float_map

    def get_float_map_ll_shape(self):
        return self.get_float_map_ll().shape

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_goal_target_shape(self):
        return self.h_target.shape

    def get_boolean_map_ll_shape(self):
        return self.get_boolean_map_ll().shape

    def get_example_goal(self):
        return self.h_target

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

    def goal_terminated(self):
        self.set_terminal_h(True)
        self.goal_active = False
