import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def pad_centered(state, map_in, pad_value):
    max_map_size = 60

    # if not multimap:
    #     padding_rows = math.ceil(state.no_fly_zone.shape[0] / 2.0)
    #     padding_cols = math.ceil(state.no_fly_zone.shape[1] / 2.0)
    #
    #     position_x, position_y = 0, 0  # state.position
    #     position_row_offset = padding_rows - position_y  # + actual_size_y  # offset is padding for 0,0
    #     position_col_offset = padding_cols - position_x  # + actual_size_x
    #     map_out = np.pad(map_in,
    #                      pad_width=[[padding_rows + position_row_offset - 1, padding_rows - position_row_offset],
    #                                 [padding_cols + position_col_offset - 1, padding_cols - position_col_offset],
    #                                 [0, 0]],  # pad_width paddings
    #                      mode='constant',
    #                      constant_values=pad_value)
    #
    #     return map_out

    if state.no_fly_zone.shape[0] > max_map_size:
        padding_rows = math.ceil(state.no_fly_zone.shape[0] / 2.0)
        padding_cols = math.ceil(state.no_fly_zone.shape[1] / 2.0)
        position_x, position_y = state.position
        position_row_offset = padding_rows - position_y
        position_col_offset = padding_cols - position_x
        map_out = np.pad(map_in,
                         pad_width=[[padding_rows + position_row_offset - 1,
                                     padding_rows - position_row_offset],
                                    [padding_cols + position_col_offset - 1,
                                     padding_cols - position_col_offset],
                                    [0, 0]],  # pad_width paddings
                         mode='constant',
                         constant_values=pad_value)
        return map_out
    else:

        padding_rows = math.ceil(max_map_size / 2.0)
        padding_cols = math.ceil(max_map_size / 2.0)
        pr_for_offset = math.ceil(state.no_fly_zone.shape[0] / 2.0)
        pc_for_offset = math.ceil(state.no_fly_zone.shape[1] / 2.0)

        position_x, position_y = 0, 0  #state.position
        position_row_offset = pr_for_offset - position_y # + actual_size_y  # offset is padding for 0,0
        position_col_offset = pc_for_offset - position_x # + actual_size_x
        map_out = np.pad(map_in,
                         pad_width=[[padding_rows + position_row_offset - 1, padding_rows - position_row_offset],
                                    [padding_cols + position_col_offset - 1, padding_cols - position_col_offset],
                                    [0, 0]],  # pad_width paddings
                         mode='constant',
                         constant_values=pad_value)


        return map_out


def pad_with_nfz_gm(map_in):
    max_map_size = 60 * 2 - 1

    if map_in.shape[0] == max_map_size:
        return map_in
    else:
        msd = int((max_map_size - map_in.shape[0]) / 2)
        # print(msd)

        # print(f'shape 01: {map_in[..., 0].shape}, shape 23: {map_in[..., 2:4].shape}')
        new_map_nfz_0 = tf.pad(map_in[..., 0],
                               paddings=[[msd, msd], [msd, msd]],
                               mode='constant',
                               constant_values=1)
        # print(f'shapenfz0: {new_map_nfz_0.shape}')
        new_map_nfz_1 = tf.pad(map_in[..., 1],
                               paddings=[[msd, msd], [msd, msd]],
                               mode='constant',
                               constant_values=1)
        new_map_rest_0 = tf.pad(map_in[..., 2],
                                paddings=[[msd, msd], [msd, msd]],
                                mode='constant',
                                constant_values=0)
        new_map_rest_1 = tf.pad(map_in[..., 3],
                                paddings=[[msd, msd], [msd, msd]],
                                mode='constant',
                                constant_values=0)

        new_map = np.concatenate(
            [np.expand_dims(new_map_nfz_0, -1), np.expand_dims(new_map_nfz_1, -1), np.expand_dims(new_map_rest_0, -1),
             np.expand_dims(new_map_rest_1, -1)], axis=-1)
        # print(f'newmap: {new_map.shape}')

        return new_map
