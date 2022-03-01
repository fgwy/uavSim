from src.ModelStats import ModelStats
import src.Map.Map as Map
from numpy import random as rnd


class BaseGridParams:
    def __init__(self):
        self.movement_range = (150, 250)
        self.movement_range_ll = (20, 50)
        self.map_path = 'res/downtown.png'


class BaseGrid:
    def __init__(self, params: BaseGridParams, stats: ModelStats):
        self.params = params
        self.stats = stats
        self.map_image = Map.load_map(params.map_path)
        self.shape = self.map_image.start_land_zone.shape
        self.starting_vector = self.map_image.get_starting_vector()
        stats.set_env_map_callback(self.get_map_image)

    def get_map_image(self):
        return self.map_image

    def get_grid_size(self):
        return self.shape

    def get_no_fly(self):
        return self.map_image.nfz

    def get_landing_zone(self):
        return self.map_image.start_land_zone
