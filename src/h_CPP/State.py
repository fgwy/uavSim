import numpy as np

from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.CPP.State import CPPState

from tensorflow.image import central_crop
from tensorflow.keras.layers import AvgPool2D
import tensorflow as tf


class H_CPPScenario:
    def __init__(self):
        self.target_path = ""
        self.position_idx = 0
        self.movement_budget = 100
        self.ll_movement_budget = 30



class H_CPPState(CPPState):
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.h_target = np.zeros_like(self.landing_zone)
        # print(self.h_target.shape)
        self.initial_h_target_cell_count = 0
        self.h_coverage = 0
        self.initial_ll_movement_budget = 30
        self.current_ll_mb = None
        self.h_terminal = False
        self.goal_active = False
        self.movement_budget = 100
        self.local_map_size = 17
        self.goal_covered = False

    def reset_h_target(self, h_target):
        # if h_target.shape == self.get_boolean_map_shape():
        #     self.h_target = h_target
        # else:
        self.h_target = self.pad_lm_to_total_size(h_target)
        self.initial_h_target_cell_count = np.sum(self.h_target)
        self.h_coverage = np.zeros(self.h_target.shape, dtype=bool)
        self.reset_ll_mb()
        self.set_terminal_h(False)
        self.goal_covered = False

    def pad_lm_to_total_size(self, h_target):
        """
        pads input of shape local_map to output of total_map_size
        """

        shape_map = self.landing_zone.shape[:2]
        shape_htarget = h_target.shape
        # print(f"input shape: {h_target.shape}")
        if shape_htarget == shape_map:
            return h_target.astype(bool)
        else:
            if self.position == None:
                x, y = 0, 0
            else:
                x, y = self.position
            pad_left = x
            pad_right = shape_map[0] - x - 1
            pad_up = y
            pad_down = shape_map[1] - y - 1

            h_target = h_target*1

            padded = np.pad(h_target, ((pad_up, pad_down), (pad_left, pad_right)))

            lm_as_tm_size = padded[int((shape_htarget[0] - 1) / 2):int(padded.shape[0] - (shape_htarget[0] - 1) / 2),
                            int((shape_htarget[1] - 1) / 2):int(padded.shape[1] - (shape_htarget[1] - 1) / 2)]
            # print(f"output shape: {lm_as_tm_size.shape}")

            return lm_as_tm_size.astype(bool)

    def goal_not_active(self):
        return not self.goal_active

    def get_remaining_h_target_cells(self):
        return np.sum(self.h_target)

    def add_explored_h_target(self, view):
        # print("reached")
        self.h_target &= ~view
        self.h_coverage |= view

    def get_local_map_shape(self):
        return self.get_local_map().shape

    def goal_ultimated(self):

        return not bool(self.h_target is not None or self.get_remaining_h_target_cells() or self.current_ll_mb >= 0)

    def reset_ll_mb(self):
        self.current_ll_mb = min(self.initial_ll_movement_budget, self.movement_budget)

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

    def get_padded_map(self):
        bm = self.get_boolean_map()[tf.newaxis, ...]
        fm = self.get_float_map()[tf.newaxis, ...]
        map_cast_hl = tf.cast(bm, dtype=tf.float32)
        padded_map_hl = tf.concat([map_cast_hl, fm], axis=3)
        padded_map_hl = tf.squeeze(padded_map_hl).numpy()
        return padded_map_hl

    def get_local_map(self):  # TODO: create local map in state to exclude computation from graph
        conv_in = self.get_padded_map()[tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_ll_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        local_map = tf.squeeze(local_map).numpy()
        return local_map

    def get_padded_map_ll(self):
        bm = self.get_boolean_map_ll()[tf.newaxis, ...]
        fm = self.get_float_map_ll()[tf.newaxis, ...]
        map_cast_ll = tf.cast(bm, dtype=tf.float32)
        padded_map_ll = tf.concat([map_cast_ll, fm], axis=3)
        padded_map_ll = tf.squeeze(padded_map_ll).numpy()
        return padded_map_ll

    def get_local_map_ll(self):  # TODO: create local map in state to exclude computation from graph
        conv_in = self.get_padded_map_ll()[tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_ll_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        local_map = tf.squeeze(local_map).numpy()
        return local_map

    def get_global_map(self, global_map_scaling):
        pm = self.get_padded_map()[tf.newaxis, ...]
        self.global_map = AvgPool2D((global_map_scaling, global_map_scaling))(pm)
        self.global_map = tf.squeeze(self.global_map).numpy()
        return self.global_map

    def get_float_map_ll_shape(self):
        return self.get_float_map_ll().shape

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_goal_target_shape(self):
        return self.h_target.shape

    def get_boolean_map_ll_shape(self):
        return self.get_boolean_map_ll().shape



    def get_global_map_shape(self, gms):
        return self.get_global_map(gms).shape


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

    def set_goal_covered(self, covered):
        self.goal_covered = covered

    def get_initial_ll_movement_budget(self):
        return self.initial_ll_movement_budget

    def get_initial_mb(self):
        return self.initial_movement_budget
