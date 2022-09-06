from enum import Enum


class GridActions(Enum):
    NORTH = 0 # down on map array y+1
    NORTH_EAST = 1
    EAST = 2 # x+1
    SOUTH_EAST = 3
    SOUTH = 4 # y-1
    SOUTH_WEST = 5
    WEST = 6 # x-1
    NORTH_WEST = 7

    LAND = 8 #
    HOVER = 9


class GridActionsNoHover(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    LAND = 4
