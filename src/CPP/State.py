import numpy as np

from src.Map.Map import Map
from src.base.BaseState import BaseState

import tensorflow as tf

from src.StateUtils import pad_centered, pad_with_nfz_gm

from tensorflow.image import central_crop
from tensorflow.keras.layers import AvgPool2D


class CPPScenario:
    def __init__(self):
        self.target_path = ""
        self.position_idx = 0
        self.movement_budget = 30


class CPPState(BaseState):
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.target = None
        self.position = [17, 17]
        self.movement_budget = None
        self.landed = False
        self.terminal = False

        self.initial_movement_budget = 180
        self.initial_target_cell_count = 0
        self.coverage = None
        self.local_map_size = 17
        self.hierarchical = False

    def reset_target(self, target):
        self.target = target
        self.initial_target_cell_count = np.sum(target)
        self.coverage = np.zeros(self.target.shape, dtype=bool)

    def get_remaining_cells(self):
        return np.sum(self.target)

    def get_total_cells(self):
        return self.initial_target_cell_count

    def get_local_map_shape(self):
        return self.get_local_map().shape

    def get_padded_map(self):
        bm = self.get_boolean_map()[tf.newaxis, ...]
        fm = self.get_float_map()[tf.newaxis, ...]
        map_cast_hl = tf.cast(bm, dtype=tf.float32)
        padded_map_hl = tf.concat([map_cast_hl, fm], axis=3)
        padded_map_hl = tf.squeeze(padded_map_hl).numpy()
        return padded_map_hl

    def get_coverage_ratio(self):
        return 1.0 - float(np.sum(self.get_remaining_cells())) / float(self.initial_target_cell_count)

    def get_scalars(self):
        return np.array([self.movement_budget])

    def get_num_scalars(self):
        return 1

    def get_goal_target_shape(self):
        return self.h_target.shape

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        padded_rest = pad_centered(self, np.concatenate([np.expand_dims(self.landing_zone, -1),
                                                         np.expand_dims(self.target, -1)], axis=-1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_float_map(self):
        shape = list(self.get_boolean_map().shape)
        shape[2] = 0
        float_map = np.zeros(tuple(shape), dtype=float)
        return float_map

    def get_float_map_shape(self):
        return self.get_float_map().shape

    def is_in_landing_zone(self):
        return self.landing_zone[self.position[1]][self.position[0]]

    def is_in_no_fly_zone(self):
        # Out of bounds is implicitly nfz
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            return self.no_fly_zone[self.position[1], self.position[0]]
        return True

    def add_explored(self, view):
        self.target &= ~view
        self.coverage |= view

    def set_terminal(self, terminal):
        self.terminal = terminal

    def set_landed(self, landed):
        self.landed = landed

    def set_position(self, position):
        self.position = position

    def decrement_movement_budget(self):
        self.movement_budget -= 1

    def is_terminal(self):
        return self.terminal

    def get_local_map(self):
        conv_in = self.get_padded_map()[tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        local_map = tf.squeeze(local_map).numpy()
        return local_map

    def get_global_map(self, global_map_scaling):
        pm = self.get_padded_map()
        # print(f'global map shape: {pm.shape}')
        pm = pad_with_nfz_gm(pm)[tf.newaxis, ...]
        # print(f'global map shape: {pm.shape}')
        self.global_map = AvgPool2D((global_map_scaling, global_map_scaling))(pm)
        self.global_map = tf.squeeze(self.global_map).numpy()
        # print(f'global map shape: {self.global_map.shape}')
        return self.global_map

    def get_global_map_shape(self, gms):
        return self.get_global_map(gms).shape

    def get_initial_mb(self):
        return self.initial_movement_budget
