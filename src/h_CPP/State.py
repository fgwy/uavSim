import numpy as np

from src.Map.Map import Map
# from src.Map.Mask import load_or_create_mask, load_or_create_distance_mask
from src.StateUtils import pad_centered, pad_with_nfz_gm, flood_fill
from src.CPP.State import CPPState

from tensorflow.image import central_crop
from tensorflow.keras.layers import AvgPool2D
import tensorflow as tf

import matplotlib.pyplot as plt


class H_CPPScenario:
    def __init__(self):
        self.target_path = ""
        self.position_idx = 0
        self.movement_budget = 100
        self.ll_movement_budget = 30



class H_CPPState(CPPState):
    def __init__(self, map_init: Map, diagonal=False):
        super().__init__(map_init)
        self.diagonal = diagonal
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
        self.hierarchical = True
        self.UnreachableMask = None
        self.initialize_masks()
        # self.multimap = False

    def initialize_masks(self):
        pass
        # load_or_create_distance_mask()
        # load_or_create_mask()

    def get_local_map_shape(self):
        # print('lm shape: ', tf.squeeze(self.get_local_map()).numpy().shape)
        return tf.squeeze(self.get_local_map()).numpy().shape

    def get_local_map(self):
        conv_in = self.get_padded_map() # [tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        # lm = central_crop(conv_in, crop_frac)
        # local_map = tf.squeeze(local_map).numpy()
        flood_mask = self.generate_local_flood_mask(local_map)
        lm = np.concatenate((local_map, flood_mask), axis=3)
        # print('get lm',lm)
        return lm

    def reset_target_h(self, h_target):
        # if h_target.shape == self.get_boolean_map_shape():
        #     self.h_target = h_target
        # else:
        self.h_target = self.pad_lm_to_total_size(h_target)
        self.initial_h_target_cell_count = np.sum(self.h_target)
        self.h_coverage = np.zeros(self.h_target.shape, dtype=bool)
        self.reset_ll_mb()
        self.set_terminal_h(False)
        self.set_goal_covered(False)

    def generate_local_flood_mask(self, local_map):
        # print('shape lm', local_map.shape)
        nfz = local_map.numpy()[0][:,:,0]*1
        # print('shape nfz lm', nfz.shape)
        x = int(nfz.shape[0]/2)+1
        flooded_lm = flood_fill(nfz, x, x, 0, 2)
        # print(flooded_lm)
        # print(tf.equal(flooded_lm, 2))
        # mask = tf.where(tf.equal(flooded_lm, 2), False, True)
        mask = np.expand_dims(np.expand_dims(np.equal(flooded_lm, 2), axis=0), axis=3)
        # print(mask.shape)
        # plt.figure()
        # plt.imshow(flooded_lm)
        # plt.show()
        # plt.figure()
        # plt.imshow(mask)
        # plt.show()
        return mask

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

    # def get_local_map_shape(self):
    #     return self.get_local_map().shape

    def goal_ultimated(self):

        return not bool(self.h_target is not None or self.get_remaining_h_target_cells() or self.current_ll_mb >= 0)

    def reset_ll_mb(self):
        self.current_ll_mb = min(self.initial_ll_movement_budget, self.movement_budget)

    def get_boolean_map_ll(self):
        # print("reached bool map ll")
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        padded_rest = pad_centered(self, np.concatenate([np.expand_dims(self.landing_zone, -1),
                                                         np.expand_dims(self.h_target, -1)], axis=-1), 0)

        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_float_map_ll(self):
        shape = list(self.get_boolean_map_ll().shape)
        shape[2] = 0
        float_map = np.zeros(tuple(shape), dtype=float)
        return float_map

    # def get_padded_map(self):
    #     bm = self.get_boolean_map()[tf.newaxis, ...]
    #     fm = self.get_float_map()[tf.newaxis, ...]
    #     map_cast_hl = tf.cast(bm, dtype=tf.float32)
    #     padded_map_hl = tf.concat([map_cast_hl, fm], axis=3)
    #     # padded_map_hl = tf.squeeze(padded_map_hl) #.numpy()
    #     return padded_map_hl

    # def get_local_map(self):  # TODO: create local map in state to exclude computation from graph
    #     conv_in = self.get_padded_map() # [tf.newaxis, ...]
    #     crop_frac = float(self.local_map_size) / float(self.get_boolean_map_ll_shape()[0])
    #     local_map = central_crop(conv_in, crop_frac)
    #     # local_map = tf.squeeze(local_map).numpy()
    #     return local_map

    def get_padded_map_ll(self):
        bm = self.get_boolean_map_ll()[tf.newaxis, ...]
        fm = self.get_float_map_ll()[tf.newaxis, ...]
        map_cast_ll = tf.cast(bm, dtype=tf.float32)
        padded_map_ll = tf.concat([map_cast_ll, fm], axis=3)
        # padded_map_ll = tf.squeeze(padded_map_ll).numpy()
        return padded_map_ll

    def get_local_map_ll(self):  # TODO: create local map in state to exclude computation from graph
        conv_in = self.get_padded_map_ll() # [tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_ll_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        # local_map = tf.squeeze(local_map).numpy()
        return local_map

    def get_local_map_ll_np(self):  # TODO: create local map in state to exclude computation from graph
        conv_in = self.get_padded_map_ll() # [tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_ll_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        local_map = tf.squeeze(local_map).numpy()
        return local_map

    # def get_global_map(self, global_map_scaling, multimap=False):
    #     pm = self.get_padded_map()
    #     # if multimap:
    #     #     pm = pad_with_nfz_gm(pm)
    #     # pm = pm[tf.newaxis, ...]
    #     self.global_map = AvgPool2D((global_map_scaling, global_map_scaling))(pm)
    #     # self.global_map = tf.squeeze(self.global_map).numpy()

        # return self.global_map

    def get_float_map_ll_shape(self):
        return tf.squeeze(self.get_float_map_ll()).numpy().shape

    # def get_boolean_map_shape(self):
    #     return self.get_boolean_map().shape

    def get_goal_target_shape(self):
        return self.h_target.shape

    def get_boolean_map_ll_shape(self):
        return self.get_boolean_map_ll().shape

    # def get_global_map_shape(self, gms):
    #     return tf.squeeze(self.get_global_map(gms)).numpy().shape

    def get_example_goal(self):
        return self.h_target

    def get_scalars_hl(self):
        return np.array([self.movement_budget])

    def get_scalars_ll(self):
        return np.array([self.current_ll_mb])

    def decrement_ll_mb(self, diagonal=False):
        self.movement_budget = np.sqrt(2) if diagonal else 1

    def set_terminal_h(self, terminal):
        self.h_terminal = terminal

    def is_terminal_h(self):
        return self.h_terminal

    def goal_terminated(self):
        self.set_terminal_h(True)
        self.goal_active = False

    def set_goal_covered(self, covered):
        self.goal_covered = covered
        self.set_terminal_h(covered) # added setting terminal

    def get_initial_ll_movement_budget(self):
        return self.initial_ll_movement_budget

    # def get_initial_mb(self):
    #     return self.initial_movement_budget
