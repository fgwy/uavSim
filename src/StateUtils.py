import math
import numpy as np


def pad_centered(state, map_in, pad_value):
    padding_rows = math.ceil(state.no_fly_zone.shape[0] / 2.0)
    padding_cols = math.ceil(state.no_fly_zone.shape[1] / 2.0)
    position_x, position_y = state.position
    position_row_offset = padding_rows - position_y
    position_col_offset = padding_cols - position_x
    # print(map_in.shape)
    map_out = np.pad(map_in,
                  pad_width=[[padding_rows + position_row_offset - 1, padding_rows - position_row_offset],
                             [padding_cols + position_col_offset - 1, padding_cols - position_col_offset],
                             [0, 0]],
                  mode='constant',
                  constant_values=pad_value)

    return map_out

def pad_with_nfz_gm(map_in):
    max_map_size = 60*2 -1

    if map_in.shape[0] == max_map_size:
        return map_in
    else:
        msd = int((max_map_size-map_in.shape[0])/2)
        # print(msd)

        # print(f'shape 01: {map_in[..., 0].shape}, shape 23: {map_in[..., 2:4].shape}')
        new_map_nfz_0 = np.pad(map_in[..., 0],
                         pad_width=[[msd, msd], [msd, msd]],
                         mode='constant',
                         constant_values=1)
        # print(f'shapenfz0: {new_map_nfz_0.shape}')
        new_map_nfz_1 = np.pad(map_in[..., 1],
                             pad_width=[[msd, msd], [msd, msd]],
                             mode='constant',
                             constant_values=1)
        new_map_rest_0 = np.pad(map_in[..., 2],
                         pad_width=[[msd, msd], [msd, msd]],
                         mode='constant',
                         constant_values=0)
        new_map_rest_1 = np.pad(map_in[..., 3],
                              pad_width=[[msd, msd], [msd, msd]],
                              mode='constant',
                              constant_values=0)

        new_map = np.concatenate([np.expand_dims(new_map_nfz_0, -1), np.expand_dims(new_map_nfz_1, -1), np.expand_dims(new_map_rest_0, -1), np.expand_dims(new_map_rest_1, -1)], axis=-1)
        # print(f'newmap: {new_map.shape}')

        return new_map
