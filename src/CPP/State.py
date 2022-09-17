import numpy as np
import os

from src.Map.Map import Map
from src.base.BaseState import BaseState

from src.Map.Map import load_map
from src.a_star.A_star import A_star
from skimage import io

import tensorflow as tf

from src.StateUtils import pad_centered, pad_with_nfz_gm

from tensorflow.image import central_crop
from tensorflow.keras.layers import AvgPool2D

import matplotlib.pyplot as plt

import tqdm


class CPPScenario:
    def __init__(self):
        self.target_path = ""
        self.position_idx = 0
        self.movement_budget = 30


class CPPState(BaseState):
    def __init__(self, map_init: Map):
        super().__init__(map_init)
        self.target = None
        self.position = [0, 0]
        self.movement_budget = None
        self.landed = False
        self.terminal = False

        self.initial_movement_budget = 180
        self.initial_target_cell_count = 0
        self.coverage = None
        self.local_map_size = 17
        self.hierarchical = False
        self.load_or_create_distance_mask('res/urban50.png', self.local_map_size)

    def reset_target(self, target):
        self.target = target
        self.initial_target_cell_count = np.sum(target)
        self.coverage = np.zeros(self.target.shape, dtype=bool)

    def get_remaining_cells(self):
        return np.sum(self.target)

    def get_total_cells(self):
        return self.initial_target_cell_count

    def get_local_map_shape(self):
        return tf.squeeze(self.get_local_map()).numpy().shape

    def get_padded_map(self):
        bm = self.get_boolean_map()[tf.newaxis, ...]
        fm = self.get_float_map()[tf.newaxis, ...]
        map_cast_hl = tf.cast(bm, dtype=tf.float32)
        padded_map_hl = tf.concat([map_cast_hl, fm], axis=3)
        # padded_map_hl = tf.squeeze(padded_map_hl).numpy()
        return padded_map_hl

    def get_coverage_ratio(self):
        return 1.0 - float(np.sum(self.get_remaining_cells())) / float(self.initial_target_cell_count)

    def get_scalars(self):
        return np.array([self.movement_budget])

    def get_num_scalars(self):
        return 1

    # def get_goal_target_shape(self):
    #     return self.h_target.shape

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_boolean_map(self, multimap=False):
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

    def decrement_movement_budget(self, diagonal = False):
        self.movement_budget -= np.sqrt(2) if diagonal else 1

    def is_terminal(self):
        return self.terminal

    def get_local_map(self):
        conv_in = self.get_padded_map() # [tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        # local_map = tf.squeeze(local_map).numpy()
        return local_map

    def get_local_map_np(self):
        conv_in = self.get_padded_map() # [tf.newaxis, ...]
        crop_frac = float(self.local_map_size) / float(self.get_boolean_map_shape()[0])
        local_map = central_crop(conv_in, crop_frac)
        local_map = tf.squeeze(local_map).numpy()
        return local_map

    def get_global_map(self, global_map_scaling, multimap=False):
        pm = self.get_padded_map()
        self.global_map = AvgPool2D((global_map_scaling, global_map_scaling))(pm)
        # self.global_map = tf.squeeze(self.global_map).numpy()
        return self.global_map

    def get_global_map_np(self, global_map_scaling, multimap=False):
        pm = self.get_padded_map()
        self.global_map = AvgPool2D((global_map_scaling, global_map_scaling))(pm)
        self.global_map = tf.squeeze(self.global_map).numpy()
        return self.global_map

    def get_global_map_shape(self, gms):
        return tf.squeeze(self.get_global_map(gms)).numpy().shape

    def get_initial_mb(self):
        return self.initial_movement_budget

    def calculate_distance_mask(self, map_path, save_as, local_map_size):
        print("Calculating distance masks")
        total_map = load_map(map_path)
        obstacles = total_map.nfz
        size = total_map.obstacles.shape[0]
        total = size * size * size * size
        astar = A_star()

        total_distance_map = np.ones((size, size, size, size), dtype=bool)

        print('total distance map shape', total_distance_map.shape)
        with tqdm.tqdm(total=total) as pbar:
            for i, j in np.ndindex(total_map.obstacles.shape):
                distance_map = np.ones((size, size), dtype=bool)

                for ii, jj in np.ndindex(total_map.obstacles.shape):
                    # distance to same pixel is none
                    if ii == i and jj == j:
                        distance = None
                        total_distance_map[j, i][jj, ii] = distance

                    # distance to obstacle is none
                    elif obstacles[j,i] or obstacles[jj,ii]:
                        distance = None
                        total_distance_map[j, i][jj, ii] = distance
                        # total_distance_map[jj, ii][j, i] = distance

                    # already calculated distances are ignored
                    elif total_distance_map[j, i][jj, ii] > 0 and False:
                        continue

                    # else calculate distance and insert it in both slots
                    else:
                        # print('astar running')

                        bla = astar.astar(obstacles, (j,i), (jj,ii))
                        # print('blbla list and shape', bla, len(bla))
                        distance = len(bla)
                        total_distance_map[j, i][jj, ii] = distance
                        # total_distance_map[jj, ii][j, i] = distance
                    pbar.update(1)

        plt.imshow(total_distance_map[0][0])
        plt.show()

        np.save(save_as, total_distance_map)
        return total_distance_map

    def save_image(self, path, image):
        if type(path) is not str:
            raise TypeError('path needs to be a string')
        if image.dtype == bool:
            io.imsave(path, image * np.uint8(255))
        else:
            io.imsave(path, image)

    def load_or_create_distance_mask(self, map_path, local_map_size):
        mask_file_name = os.path.splitext(map_path)[0] + "_masked_distances.npy"
        if os.path.exists(mask_file_name):
            return np.load(mask_file_name)
        else:
            return self.calculate_distance_mask(map_path, mask_file_name, local_map_size)


