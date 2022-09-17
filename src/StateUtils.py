import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def pad_centered(state, map_in, pad_value):
    max_map_size = 60
    # print(map_in.shape)

    msd = int((max_map_size - map_in.shape[0]))
    # print(msd)

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
    # if True:
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
        # print(map_in.shape, map_out.shape, map_out.shape)
        return map_out
    else:

        msd_halves = msd #//2
        padding_rows = math.ceil(max_map_size / 2.0)
        padding_cols = math.ceil(msd / 2.0)
        pr_for_offset = math.ceil(state.no_fly_zone.shape[0] / 2.0)
        pc_for_offset = math.ceil(state.no_fly_zone.shape[1] / 2.0)
        padding_rows = math.ceil(state.no_fly_zone.shape[0] / 2.0)
        padding_cols = math.ceil(state.no_fly_zone.shape[1] / 2.0)

        position_x, position_y = state.position

        # position_x, position_y = 0, 0  #state.position
        # position_row_offset = pr_for_offset - position_y # + actual_size_y  # offset is padding for 0,0
        # position_col_offset = pc_for_offset - position_x # + actual_size_x

        position_row_offset = padding_rows - position_y # + actual_size_y  # offset is padding for 0,0
        position_col_offset = padding_cols - position_x # + actual_size_x
        map_out = np.pad(map_in,
                         pad_width=[[padding_rows + position_row_offset - 1 + msd_halves, padding_rows - position_row_offset + msd_halves],
                                    [padding_cols + position_col_offset - 1 + msd_halves, padding_cols - position_col_offset + msd_halves],
                                    [0, 0]],  # pad_width paddings
                         mode='constant',
                         constant_values=pad_value)
        # print(map_in.shape, map_out.shape)
        #
        # print(position_x, position_y)
        # plt.imshow(map_out[:,:,0])
        # plt.show()
        return map_out


def pad_with_nfz_gm(map_in):
    max_map_size = 60 # * 2 - 1

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

def flood_fill(field, x, y, old, new):
    # y, x = position
    # we need the x and y of the start position, the old value,
    # and the new value    # the flood fill has 4 parts
    # firstly, make sure the x and y are inbounds
    if x < 0 or x >= len(field[0]) or y < 0 or y >= len(field):
        return  # secondly, check if the current position equals the old value
    if field[y][x] != old:
        return

    # thirdly, set the current position to the new value
    field[y][x] = new  # fourthly, attempt to fill the neighboring positions
    flood_fill(field, x + 1, y, old, new)
    flood_fill(field, x - 1, y, old, new)
    flood_fill(field, x, y + 1, old, new)
    flood_fill(field, x, y - 1, old, new)
    return field
