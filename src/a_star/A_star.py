from scipy.spatial import cKDTree as kd
from tensorflow import argmax as argmx
import numpy as np
import numba as nb

# @nb.experimental.jitclass
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class A_star:
    def __init__(self):
        self.stuff = None
        self.path = None
        self.one_random = True


    ####### adapted from: https://stackoverflow.com/questions/43306291/find-the-nearest-nonzero-element-and-corresponding-index-in-a-2d-numpy-array

    @staticmethod
    def nearest_nonzero_idx(map, pos):
        x = pos[0]
        y = pos[1]

        idx = np.argwhere(map)

        idx = idx[~(idx == [x, y]).all(1)]

        return idx[((idx - [x, y]) ** 2).sum(1).argmin()]

    @staticmethod
    def nearest_nonzero_idx_v2(a, pos):
        x = pos[0]
        y = pos[1]
        tmp = a[x, y]
        a[x, y] = 0
        r, c = np.nonzero(a)
        a[x, y] = tmp
        min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        return r[min_idx], c[min_idx]

    @staticmethod
    # @nb.jit(npython=True) # TODO: add input data classez
    # @nb.jit
    def astar(maze, start, end):
        """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        Maze position is initia.lized as (0,0) in top left corner
        """

        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        i = 0
        n = 0
        m = 0
        while len(open_list) > 0:
            m +=1

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                n +=1
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
                if n > 3000:
                    # print('breaking A0star search!!')
                    return None

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                # print(f'number of inm is: {i} {n} while: {m}')
                return path[::-1] # Return reversed path

            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: #, (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if maze[node_position[0]][node_position[1]] != 0:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                i += 1
                if i > 3000:
                    # print('breaking A0star search!!')
                    return None
                # print(i)
                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)

    def get_A_star_action(self, state, steps_in_smdp):
        # obstacles = state.no_fly_zone * 1

        if steps_in_smdp == 1 or self.one_random:
            x, y = state.position
            start = (y, x) # TODO: keep track!!
            obstacles = state.no_fly_zone*1
            end = np.where(state.h_target == 1)
            # self.path = self.astar(obstacles,start,end)
            try:
                self.path = self.astar(obstacles, start, end)
                self.one_random = False
                # print(f'path: {self.path}')
            except:
                self.one_random = True
                print(f'A-star: one random action right away!!')
                return np.random.randint(0, 4)

        if self.path == None:
            print('A-star: random action!!!')
            return np.random.randint(0, 4)
        else:
            # print(f'steps in smdp: {steps_in_smdp}')
            a = self.path[steps_in_smdp][0]-self.path[steps_in_smdp-1][0] # x
            b = self.path[steps_in_smdp][1]-self.path[steps_in_smdp-1][1] # y
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
            # print(f'A-Star action: {action}')
        return action


    @staticmethod
    def do_kdtree(combined_x_y_arrays, points):
        mytree = kd(combined_x_y_arrays)
        dist, indexes = mytree.query(points)
        return indexes

    def is_in_no_fly_zone(self, position, no_fly_zone):
        # Out of bounds is implicitly nfz
        if 0 <= position[1] < no_fly_zone.shape[0] and 0 <= position[0] < no_fly_zone.shape[1]:
            return no_fly_zone[position[1], position[0]]
        return True

def main():

    a = np.zeros((5, 5))
    a[3][2] = 1
    print(a)
    print(argmx(a))



    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]


    print(np.asarray(maze))

    start = (0, 0)
    end = (9, 9)
    ast = A_star()

    path = ast.astar(maze, start, end)
    print(path)


if __name__ == '__main__':
    main()