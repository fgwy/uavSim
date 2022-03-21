from enum import Enum


class GridActions(Enum):
    NORTH = 0 # down on map array y+1
    EAST = 1 # x+1
    SOUTH = 2 # y-1
    WEST = 3 # x-1
    LAND = 4 #
    HOVER = 5


class GridActionsNoHover(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    LAND = 4
