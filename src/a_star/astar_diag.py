#!/usr/bin/env python3

# import graph as gh
import os

from heapq import *
from math import inf
from sys import argv, exit
from time import time, perf_counter_ns, perf_counter

import numpy as np
from abc import ABC, abstractmethod
from random import random

class Astar:

    def __init__(self, matrix):
        self.mat = self.prepare_matrix(matrix)

    class Node:
        def __init__(self, x, y, weight=0):
            self.x = x
            self.y = y
            self.weight = weight
            self.heuristic = 0
            self.parent = None

        def __repr__(self):
            return str(self.weight)

    def print(self):
        for y in self.mat:
            print(y)

    def prepare_matrix(self, mat):
        matrix_for_astar = []
        for y, line in enumerate(mat):
            tmp_line = []
            for x, weight in enumerate(line):
                tmp_line.append(self.Node(x, y, weight=weight))
            matrix_for_astar.append(tmp_line)
        return matrix_for_astar

    def equal(self, current, end):
        return current.x == end.x and current.y == end.y

    def heuristic(self, current, other):
        # return abs(current.x - other.x) + abs(current.y - other.y)
        return np.sqrt((current.x - other.x)**2 + (current.y - other.y)**2)

    def neighbours(self, matrix, current):
        neighbours_list = []
        if current.x - 1 >= 0 and current.y - 1 >= 0 and matrix[current.y - 1][current.x - 1].weight is not None:
            neighbours_list.append(matrix[current.y - 1][current.x - 1])
        if current.x - 1 >= 0 and matrix[current.y][current.x - 1].weight is not None:
            neighbours_list.append(matrix[current.y][current.x - 1])
        if current.x - 1 >= 0 and current.y + 1 < len(matrix) and matrix[current.y + 1][
            current.x - 1].weight is not None:
            neighbours_list.append(matrix[current.y + 1][current.x - 1])
        if current.y - 1 >= 0 and matrix[current.y - 1][current.x].weight is not None:
            neighbours_list.append(matrix[current.y - 1][current.x])
        if current.y + 1 < len(matrix) and matrix[current.y + 1][current.x].weight is not None:
            neighbours_list.append(matrix[current.y + 1][current.x])
        if current.x + 1 < len(matrix[0]) and current.y - 1 >= 0 and matrix[current.y - 1][
            current.x + 1].weight is not None:
            neighbours_list.append(matrix[current.y - 1][current.x + 1])
        if current.x + 1 < len(matrix[0]) and matrix[current.y][current.x + 1].weight is not None:
            neighbours_list.append(matrix[current.y][current.x + 1])
        if current.x + 1 < len(matrix[0]) and current.y + 1 < len(matrix) and matrix[current.y + 1][
            current.x + 1].weight is not None:
            neighbours_list.append(matrix[current.y + 1][current.x + 1])
        return neighbours_list

    def build(self, end):
        node_tmp = end
        path = []
        while (node_tmp):
            path.append([node_tmp.x, node_tmp.y])
            node_tmp = node_tmp.parent
        return list(reversed(path))

    def run(self, point_start, point_end):
        matrix = self.mat
        start = self.Node(point_start[0], point_start[1])
        end = self.Node(point_end[0], point_end[1])
        closed_list = []
        open_list = [start]

        while open_list:
            current_node = open_list.pop()

            for node in open_list:
                if node.heuristic < current_node.heuristic:
                    current_node = node

            if self.equal(current_node, end):
                return self.build(current_node)

            for node in open_list:
                if self.equal(current_node, node):
                    open_list.remove(node)
                    break

            closed_list.append(current_node)

            for neighbour in self.neighbours(matrix, current_node):
                if neighbour in closed_list:
                    continue
                if neighbour.heuristic < current_node.heuristic or neighbour not in open_list:
                    neighbour.heuristic = neighbour.weight + self.heuristic(neighbour, end)
                    neighbour.parent = current_node
                if neighbour not in open_list:
                    open_list.append(neighbour)

        return None

def main(maze):
    maze = np.where(maze, 1000, 0)
    print(maze)
    astar = Astar(maze)

    start=[0,0]
    end=[0,9]

    t1 = perf_counter()
    path = astar.run(start, end)
    t2 = perf_counter()

    print(path, t2-t1)




if __name__ == '__main__':
    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]

    main(maze)





class Graph(ABC):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @abstractmethod
    def is_obstacle(self, node):
        pass

    @abstractmethod
    def cost_to(self, from_node, to_node):
        pass

    def exists_node(self, node):
        (x, y) = node
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, node):
        (x, y) = node
        result = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x - 1, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1)]

        return filter(self.exists_node, result)

    def heuristic(self, node, goal):
        (x1, y1) = node
        (x2, y2) = goal

        # Manhattan distance
        # return abs(x1 - x2) + abs(y1 - y2)

        # Chebychev distance
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        D = D2 = 1
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


class RandomObstacleGraph(Graph):
    def __init__(self, width, height, obstacle_chance):
        super().__init__(width, height)

        self.obstacles = []

        for y in range(height):
            for x in range(width):
                if random() < obstacle_chance:
                    self.obstacles.append((x, y))

    def is_obstacle(self, node):
        return node in self.obstacles

    def cost_to(self, node):
        return 1  # all nodes in this Graph class have a weight of 1


class PredefinedGraph(Graph):
    def __init__(self, width, height, nodes_dict):
        super().__init__(width, height)

        self.nodes_dict = nodes_dict

    def is_obstacle(self, node):
        if node in self.nodes_dict:
            # -1 weights represent obstacles
            return self.nodes_dict[node] == -1
        return False

    def cost_to(self, node):
        if node not in self.nodes_dict:
            return 1
        return self.nodes_dict[node]


def a_star_search(start, goal, graph):
    frontier = []
    came_from = {}
    cost = {}

    heappush(frontier, (0, start))
    came_from[start] = None
    cost[start] = 0

    # should never happen if path to goal is possible
    if graph.is_obstacle(start):
        return None

    while frontier:
        current = heappop(frontier)[1]

        if current == goal:
            return came_from

        for neighbor in graph.neighbors(current):
            new_cost = cost[current] + graph.cost_to(neighbor)

            if new_cost < cost.get(neighbor, inf) and not graph.is_obstacle(neighbor):
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + graph.heuristic(neighbor, goal)

                heappush(frontier, (priority, neighbor))
    return None


def reconstruct_path(start, goal, came_from):
    result = []
    node = goal
    while node != None:
        result.append(node)
        node = came_from[node]
    return result


def print_graph(path, graph):
    GREEN = '\033[38;5;82m'
    RED = '\033[38;5;196m'
    YELLOW = '\033[38;5;226m'
    RESET = '\033[0m'

    for y in range(graph.height):
        for x in range(graph.width):
            node = (x, y)

            if graph.is_obstacle(node):
                c = RED + '=' + RESET
            elif graph.cost_to(node) != 1:
                c = YELLOW + str(graph.cost_to(node)) + RESET
            elif node in path:
                c = GREEN + '#' + RESET
            else:
                c = '.'

            print(c + ' ', end='')
        print()  # print newline


def parse_graph_file(fname, width, height):
    nodes = {}
    x = y = 0

    with open(fname) as f:
        while True:
            c = f.read(1)

            if not c:
                return gh.PredefinedGraph(width, height, nodes)

            current = (x, y)

            if c == '\n':
                y += 1
                x = 0
            else:
                x += 1

            if c == '=': nodes[current] = -1
            if c.isdigit(): nodes[current] = int(c)


if __name__ == '__main__':
    width = height = 32
    PREDEF_GRAPH_DIR = 'predef_graphs'

    if len(argv) > 1:
        fname = PREDEF_GRAPH_DIR + os.sep + argv[1]
        if argv[1].isdigit():

            # change from percent
            graph = gh.RandomObstacleGraph(width, height, float(argv[1]) / 100)
            print('Using %s%% obstacle chance.' % argv[1])

        elif os.path.isfile(fname):

            graph = parse_graph_file(fname, width, height)
            print('Using graph file "%s".' % fname)

        else:
            print('Graph file "%s" not found.' % fname)
            exit(1)
    else:
        print('Defaulting to 20% obstacle chance.')
        graph = gh.RandomObstacleGraph(width, height, 0.2)

    start = (0, 0)  # upper left corner
    goal = (31, 31)  # lower right corner

    print('Calculating path...')

    last_time = time()
    came_from = a_star_search(start, goal, graph)

    if came_from != None:
        path = reconstruct_path(start, goal, came_from)
        print_graph(path, graph)
    else:
        print('No path was found.')
        exit(1)

    print('Finished in %fs.' % (time() - last_time))

    exit(0)