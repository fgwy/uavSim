import numpy as np
import os
import tqdm
from src.Map.Map import load_map

def print_field(field):
    # this function will print the contents of the array
    for y in range(len(field)):
        for x in range(len(field[0])):            # value by column and row
            print(field[y][x], end=' ')

            if x == len(field[0])-1:
                # print a new line at the end of each row
                print('\n')


def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[y0, x0] = False

    while x0 != x1 or y0 != y1:
        if 2 * error - y_dist > x_dist - 2 * error:
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[y0, x0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[y0, x0] = False

# def flood_recursive(matrix):
#     width = len(matrix)
#     height = len(matrix[0])
#
#     def fill(x, y, start_color, color_to_update):
#         # if the square is not the same color as the starting point
#         if matrix[x][y] != start_color:
#             return
#         # if the square is not the new color
#         elif matrix[x][y] == color_to_update:
#             return
#         else:
#             # update the color of the current square to the replacement color
#             matrix[x][y] = color_to_update
#             neighbors = [(x - 1, y), (x + 1, y), (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1),
#                          (x, y - 1), (x, y + 1)]
#             for n in neighbors:
#                 if 0 <= n[0] <= width - 1 and 0 <= n[1] <= height - 1:
#                     fill(n[0], n[1], start_color, color_to_update)
#
#     # pick a random starting point
#     start_x = random.randint(0, width - 1)
#     start_y = random.randint(0, height - 1)
#     start_color = matrix[start_x][start_y]
#     fill(start_x, start_y, start_color, 9)
#     return matrix

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

# if __name__ == "__main__":
#     field = [
#         [1, 1, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 0, 0, 0, 0],
#         [0, 1, 0, 1, 0, 1, 1],
#         [0, 1, 0, 0, 0, 1, 0],
#     ]
#     # print field before the flood fill
#     print_field(field)
#     flood_fill(field, 1, 1, 0, 3)
#     print("Doing flood fill with '3'")
#
#     # print the field after the flood fill
#     print_field(field)


def calculate_unreachable_mask(map_path, save_as, lm_size=17):
    print("Calculating Unreachable Masks")
    total_map = load_map(map_path)
    obstacles = total_map.obstacles
    size = total_map.obstacles.shape[0]
    total = size * size


    total_mask_map = np.ones((size, size, lm_size, lm_size), dtype=bool)
    with tqdm.tqdm(total=total) as pbar:
        for i, j in np.ndindex(total_map.obstacles.shape):
            mask_map = np.ones((size, size), dtype=bool)



            total_mask_map[j, i] = mask_map
            pbar.update(1)

    np.save(save_as, total_mask_map)
    return total_mask_map

def load_or_create_mask(map_path):
    mask_file_name = os.path.splitext(map_path)[0] + "_masked_unreachable.npy"
    if os.path.exists(mask_file_name):
        return np.load(mask_file_name)
    else:
        return calculate_unreachable_mask(map_path, mask_file_name)
