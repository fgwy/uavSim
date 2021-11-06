from src.h_CPP.State import H_CPPState
from src.CPP.SimpleSquareCamera import SimpleSquareCameraParams, SimpleSquareCamera
from src.ModelStats import ModelStats
from src.base.GridActions import GridActions, GridActionsNoHover
from src.base.GridPhysics import GridPhysics


class H_CPPPhysicsParams:
    def __init__(self):
        self.camera_params = SimpleSquareCameraParams()


class H_CPPPhysics(GridPhysics):
    def __init__(self, params: H_CPPPhysicsParams, stats: ModelStats):
        super().__init__()
        self.landed = False

        self.camera = None

        self.params = params

        self.register_functions(stats)

    def register_functions(self, stats: ModelStats):
        stats.set_evaluation_value_callback(self.get_cral)

        stats.add_log_data_callback('cral', self.get_cral)
        stats.add_log_data_callback('cr', self.get_coverage_ratio)
        stats.add_log_data_callback('successful_landing', self.has_landed)
        stats.add_log_data_callback('boundary_counter', self.get_boundary_counter)
        stats.add_log_data_callback('landing_attempts', self.get_landing_attempts)
        stats.add_log_data_callback('movement_ratio', self.get_movement_ratio)

    def reset(self, state: H_CPPState):
        GridPhysics.reset(self, state)
        self.landed = False

        self.camera = SimpleSquareCamera(self.params.camera_params)

    def step(self, action: GridActions):
        self.movement_step(action)
        if not self.state.terminal:
            self.vision_step()

        if self.state.landed:
            self.landed = True
        if self.state.get_remaining_h_target_cells() == 0:
            self.state.set_terminal_h(self.state.get_remaining_h_target_cells() == 0)

        return self.state

    def vision_step(self):
        view = self.camera.computeView(self.state.position, 0)
        self.state.add_explored(view)
        self.state.add_explored_h_target(view)

    def get_example_action(self):
        return [self.get_example_action_h(), self.get_example_action_l()]

    def get_example_action_l(self):
        return GridActionsNoHover.LAND

    def get_example_action_h(self):
        return None

    def set_terminal_h(self, terminal):
        self.state.set_terminal_h(terminal)
        return self.state

    def is_in_landing_zone(self):
        return self.state.is_in_landing_zone()

    def get_coverage_ratio(self):
        return self.state.get_coverage_ratio()

    def get_movement_budget_used(self):
        return self.state.initial_movement_budget - self.state.movement_budget

    def get_movement_budget_used_ll(self):
        return self.state.initial_movement_budget_ll - self.state.current_mb_ll

    def get_cral(self):
        return self.get_coverage_ratio() * self.landed

    def get_boundary_counter(self):
        return self.boundary_counter

    def get_landing_attempts(self):
        return self.landing_attempts

    def get_movement_ratio(self):
        return float(self.get_movement_budget_used()) / float(self.state.initial_movement_budget)

    def get_movement_ratio_ll(self):
        return float(self.get_movement_budget_used_ll()) / float(self.state.initial_movement_budget_ll)

    def has_landed(self):
        return self.landed

    def reset_h_target(self, goal):
        self.state.reset_h_target(goal)
        return self.state
