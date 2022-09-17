from scipy.spatial import cKDTree as kd
from tensorflow import argmax as argmx
import numpy as np
import numba as nb
import heapq

# from astar_python.astar import Astar

from time import time, perf_counter_ns, perf_counter
import matplotlib.pyplot as plt

import sys


class A_star:
    def __init__(self, diag=False):
        self.stuff = None
        self.path = None
        self.one_random = True
        self.diag = diag
        self.moves = [(0, -1), (0, 1), (-1, 0), (1, 0)] if not self.diag else [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1),
                                                                 (1, -1), (1, 1)]

        self.action_dict = {'(-1, 0)' : 0, #N
                            '(0, -1)': 1, #E
                            '(1, 0)': 2, #S
                            '(0, 1)': 3, #W
                            '(-1, -1)': 6, #NE
                            '(1, -1)': 7, #SE
                            '(1, 1)': 8, #SW
                            '(-1, 1)': 9, #NW
                            }


    ####### adapted from: https://stackoverflow.com/questions/43306291/find-the-nearest-nonzero-element-and-corresponding-index-in-a-2d-numpy-array

    # Heuristic - Pythagoras' theorem
    @staticmethod
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def astar(self, array, start, goal, diag=None):
        # possible movemnt
        neighbors = []
        if diag is None:
            neighbors = self.moves
        elif diag is True:
            neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1),
                                                                 (1, -1), (1, 1)]
        elif diag is False:
            neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        # if movement:
        #     neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        # else:
        #     neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # tiles not to choose ever again
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}

        # contains all positions considered
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        # checking for open positions
        while oheap:
            # find the one with the smallest f score(overall cost (g+h))
            current = heapq.heappop(oheap)[1]
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data
            # forget that one then
            close_set.add(current)
            # calculate g scores for all possible neighbors
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                if 0 <= neighbor[0] < array.shape[0]:
                    if 0 <= neighbor[1] < array.shape[1]:
                        if not array[neighbor[0]][neighbor[1]] == 1:
                            if array[current[0] + i][current[1]] or array[current[0]][current[1] + j]:
                                continue
                        else:
                            continue
                    else:
                        # array bound y walls
                        continue
                else:
                    # array bound x walls
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

    def display_route(self, route1, start, goal, grid):
        x1 = []
        y1 = []
        distance1 = 0

        # add route into two seperate arrays for plotting
        for i in range(0, len(route1)):
            nx = route1[i][0]
            ny = route1[i][1]
            x1.append(nx)
            y1.append(ny)
            distance1 += np.sqrt((nx) ** 2 + (ny) ** 2)
        # add start point > not included, skips itself as a step
        x1.append(start[0])
        y1.append(start[1])
        print(distance1)

        # implementation
        route2 = self.astar(grid, start, goal, 1)
        x2 = []
        y2 = []
        distance2 = 0

        # add route into two seperate arrays for plotting
        for i in range(0, len(route2)):
            nx = route2[i][0]
            ny = route2[i][1]
            x2.append(nx)
            y2.append(ny)
            distance2 += np.sqrt((nx) ** 2 + (ny) ** 2)
        # add start point > not included, skips itself as a step
        x2.append(start[0])
        y2.append(start[1])
        print(distance2)

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(grid, cmap=plt.cm.binary)
        ax.plot(y1, x1, color="#bdc3c7", linewidth=3, zorder=10, label=str(round(distance1, 1)) + " units")
        ax.plot(y2, x2, color="blue", linewidth=3, zorder=10, label=str(round(distance2, 1)) + " units")
        ax.scatter(start[1], start[0], marker="*", color="#27ae60", s=200, zorder=20)
        ax.scatter(goal[1], goal[0], marker="*", color="#c0392b", s=200, zorder=20)
        plt.title("A* implementations")
        ax.legend()
        plt.show()


    def get_A_star_action(self, state=None, steps_in_smdp=None, obstacles=None, position=None, h_target=None, nfz=None):
        # obstacles = state.no_fly_zone * 1
        obstacles = obstacles*1

        if steps_in_smdp == 1 or self.one_random:
            print('calculating path')
            x, y = position
            start = (y, x)  # TODO: keep track!!
            # obstacles = state.no_fly_zone*1
            # obstacles = np.where(state.no_fly_zone>=1, state.no_fly_zone, 1)
            end = np.where(h_target == 1)
            if obstacles[y, x] or obstacles[end[0], end[1]]:
                print(f'obstacles on position or target: start: {obstacles[y,x]} end: {obstacles[end[0], end[1]]}')
            # self.path = self.astar(obstacles,start,end)
            # try:
            self.path = self.astar(np.array(obstacles), start, end)
            self.one_random = False
            print(f'path: {self.path}')
            if self.path is None:
                self.one_random = True
                print(f'A-star: one random action right away!!')
                return np.random.randint(0, 4)


        if self.path is None:
            print('A-star: random action!!!')
            self.one_random = True

            a = np.zeros(10)
            if not obstacles[y-1,x]:
                a[0]=0
            elif obstacles[y+1,x]:
                a[0] = 0
            elif obstacles[y,x-1]:
                a[0]=0
            elif obstacles[y,x+1]:
                a[0]=0
            return np.random.randint(np.setdiff1d(range(0,10), a))
            # return np.random.randint(0, 4)
        else:
            self.one_random = False
            # print(f'steps in smdp: {steps_in_smdp}')
            try:
                a = self.path[steps_in_smdp][0]-self.path[steps_in_smdp-1][0] # x
                b = self.path[steps_in_smdp][1]-self.path[steps_in_smdp-1][1] # y
            # print(f'Internal NFZ check: {self.is_in_no_fly_zone(self.path[1], obstacles)}')
            except:
                print('path exceptions')
                a = b = None
            action = (a, b)
            # print(f'pre action: {action}')

            if a is None or b is None:
                action1 = 5  # hover and wait
                print(f'@@@A-Star Hover!!!: {action}')
            else:
                print(action)
                action1 = self.action_dict[str(action)]
                print(action1)

                # self.action_dict = {'(1, 0)': 0,  # N
                #                     '(0, 1)': 1,  # E
                #                     '(-1, 0)': 2,  # S
                #                     '(0, -1)': 3,  # W
                #                     '(1, 1)': 6,  # NE
                #                     '(-1, 1)': 7,  # SE
                #                     '(-1, -1)': 8,  # SW
                #                     '(1, -1)': 9,  # NW
                #                     }
        map1 = obstacles + (h_target*3)
        map1[position[1], position[0]] = 5
        plt.imshow(map1)
        plt.show()
        return action1


    def is_in_no_fly_zone(self, position, no_fly_zone):
        # Out of bounds is implicitly nfz
        if 0 <= position[1] < no_fly_zone.shape[0] and 0 <= position[0] < no_fly_zone.shape[1]:
            return no_fly_zone[position[1], position[0]]
        return True

def main():

    # a = np.zeros((5, 5))
    # a[3][2] = 1
    # print(a)
    # print(argmx(a))



    maze = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

    big_maze=np.array([[
[0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
[1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
[0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
[1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
[1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
[1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1],
[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
[1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1],
[1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
[1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1],
[1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
[1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
[1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]]])



    print(np.asarray(maze), np.asarray(maze).shape)
    # print(np.asarray(big_maze), np.asarray(big_maze).shape)

    start = (0, 0)
    end = (0, 9)
    diag = False
    ast = A_star(diag)
    t1= perf_counter()
    path = ast.astar(maze, start, end)
    t2=perf_counter()
    print(path, t2-t1)
    print(len(path))

    t1 = perf_counter()
    path = ast.astar(maze, start, end)
    t2 = perf_counter()
    print('second run', t2 - t1)
    print(len(path))

    # print(f'steps in smdp: {steps_in_smdp}')
    for i in range(15):
        a = path[i+1][0] - path[i][0]  # x
        b = path[i+1][1] - path[i][1]  # y
        # print(f'Internal NFZ check: {self.is_in_no_fly_zone(self.path[1], obstacles)}')
        action = [a, b]
        # print(f'pre action: {action}')
        if action[0] == 1:
            action = 0
        elif action[1] == 1:
            action = 1
        elif action[0] == -1:
            action = 2
        elif action[1] == -1:
            action = 3
        print(f'action {action} path: {[a,b]}')


if __name__ == '__main__':
    main()

    #
    # if steps_in_smdp == 1 or self.one_random:
    #     x, y = state.position
    #     start = (y, x)  # TODO: keep track!!
    #     # obstacles = state.no_fly_zone*1
    #     obstacles = np.where(state.no_fly_zone>=1, state.no_fly_zone, 1)
    #     end = np.where(state.h_target == 1)
    #     if obstacles[y, x] or obstacles[end[0], end[1]]:
    #         print(f'obstacles on position or target: start: {obstacles[y,x]} end: {obstacles[end[0], end[1]]}')
    #     # self.path = self.astar(obstacles,start,end)
    #     # try:
    #     self.path = self.astar(obstacles, start, end)
    #     self.one_random = False
    #     # print(f'path: {self.path}')
    #     if self.path is None:
    #         self.one_random = True
    #         print(f'A-star: one random action right away!!')
    #         return np.random.randint(0, 4)

# if self.diag:
#
#     if action[0] == 1 and action[1] == 0:
#         action1 = 0 #N
#     elif action[0] == 0 and action[1] == 1:
#         action1 = 1 #E
#     elif action[0] == -1 and action[1] == 0:
#         action1 = 2 #S
#     elif action[0] == 0 and action[1] == -1:
#         action1 = 3 #W
#     #diag actions
#     elif action[0] == 1 and action[1] == 1:
#         action1 = 6 #NE
#     elif action[0] == -1 and action[1] == 1:
#         action1 = 7 #SE
#     elif action[0] == -1 and action[1] == 0:
#         action1 = 8 #SW
#     elif action[0] == 1 and action[1] == -1:
#         action1 = 9 #NW
# else:
#     if action[0] == 1 and action[1] == 0:
#         action1 = 0
#     elif action[1] == 1 and action[0] == 0:
#         action1 = 1
#     elif action[0] == -1 and action[1] == 0:
#         action1 = 2
#     elif action[1] == -1 and action[0] == 0:
#         action1 = 3



    # @staticmethod
    # def do_kdtree(combined_x_y_arrays, points):
    #     mytree = kd(combined_x_y_arrays)
    #     dist, indexes = mytree.query(points)
    #     return indexes


    # @staticmethod
    # def nearest_nonzero_idx(map, pos):
    #     x = pos[0]
    #     y = pos[1]
    #
    #     idx = np.argwhere(map)
    #
    #     idx = idx[~(idx == [x, y]).all(1)]
    #
    #     return idx[((idx - [x, y]) ** 2).sum(1).argmin()]
    #
    # @staticmethod
    # def nearest_nonzero_idx_v2(a, pos):
    #     x = pos[0]
    #     y = pos[1]
    #     tmp = a[x, y]
    #     a[x, y] = 0
    #     r, c = np.nonzero(a)
    #     a[x, y] = tmp
    #     min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
    #     return r[min_idx], c[min_idx]

    # def astar(self, maze, start, end):
    #     return self.astar_compiled(self.moves, maze, start, end)


    # @staticmethod
    # @nb.jit(npython=True) # TODO: add input data classez
    # @nb.jit
    # def astar_compiled(moves, maze, start, end):
    # def astar(self, maze, start, end):
    #     """
    #     Returns a list of tuples as a path from the given start to the given end in the given maze
    #     Maze position is initia.lized as (0,0) in top left corner
    #     """
    #
    #     # Create start and end node
    #     start_node = Node(None, start)
    #     start_node.g = start_node.h = start_node.f = 0
    #     end_node = Node(None, end)
    #     end_node.g = end_node.h = end_node.f = 0
    #
    #     # Initialize both open and closed list
    #     open_list = []
    #     closed_list = []
    #
    #     # Add the start node
    #     open_list.append(start_node)
    #
    #     # Loop until you find the end
    #     i = 0
    #     n = 0
    #     m = 0
    #     limit = 6000
    #     while len(open_list) > 0:
    #         print("len list:", len(open_list))
    #         m +=1
    #
    #         # Get the current node
    #         current_node = open_list[0]
    #         current_index = 0
    #         for index, item in enumerate(open_list):
    #             n += 1
    #             if item.f < current_node.f:
    #                 current_node = item
    #                 current_index = index
    #             # if n % limit == 0:
    #             #     # print(f'n: {n}')
    #             #     return None
    #
    #         # Pop current off open list, add to closed list
    #         open_list.pop(current_index)
    #         closed_list.append(current_node)
    #
    #         # Found the goal
    #         if current_node == end_node:
    #             path = []
    #             current = current_node
    #             while current is not None:
    #                 path.append(current.position)
    #                 current = current.parent
    #             # print(f'number of inm is: {i} {n} while: {m}')
    #             return path[::-1] # Return reversed path
    #
    #         # Generate children
    #         children = []
    #         for new_position in self.moves: # Adjacent squares
    #
    #             # Get node position
    #             node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
    #
    #             # Make sure within range
    #             if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
    #                 continue
    #
    #             # Make sure not crossing square when moving diagonally
    #             # if maze[current_node.position[0] + new_position[0]][current_node.position[1]] or maze[current_node.position[0]][current_node.position[1] + new_position[1]]:
    #             #     print('break due to diagonal')
    #             #     continue
    #             # Make sure walkable terrain
    #             if maze[node_position[0]][node_position[1]] != 0:
    #                 continue
    #
    #             # Create new node
    #             new_node = Node(current_node, node_position)
    #
    #             # Append
    #             children.append(new_node)
    #
    #         # Loop through children
    #         for child in children:
    #             i += 1
    #             # if i % limit == 0:
    #             #     # print(f'i: {i}')
    #             #     return None
    #             # print(i)
    #             # Child is on the closed list
    #             for closed_child in closed_list:
    #                 if child == closed_child:
    #                     continue
    #
    #             # Create the f, g, and h values
    #             child.g = current_node.g + 1
    #             #heuristic fior diag act
    #             child.h = np.sqrt(((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2))
    #             child.f = child.g + child.h
    #
    #             # Child is already in the open list
    #             for open_node in open_list:
    #                 if child == open_node and child.g > open_node.g:
    #                     continue
    #
    #             # Add the child to the open list
    #             open_list.append(child)

    # # @nb.experimental.jitclass
    # class Node:
    #     """A node class for A* Pathfinding"""
    #
    #     def __init__(self, parent=None, position=None):
    #         self.parent = parent
    #         self.position = position
    #
    #         self.g = 0
    #         self.h = 0
    #         self.f = 0
    #
    #     def __eq__(self, other):
    #         return self.position == other.position