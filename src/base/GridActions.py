from enum import Enum

class GridActionsDiagonal(Enum):
    NORTH = 0 # down on map array y+1
    EAST = 1 # x+1
    SOUTH = 2 # y-1
    WEST = 3 # x-1
    LAND = 4
    HOVER = 5
    NORTH_EAST = 6 #+1+1
    SOUTH_EAST = 7 #-1+1
    SOUTH_WEST = 8 #-1-1
    NORTH_WEST = 9 #+1-1

class GridActions(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    LAND = 4
    HOVER = 5

class GridActionsNoHover(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    LAND = 4
