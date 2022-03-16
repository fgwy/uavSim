from src.h_CPP.State import H_CPPState
from src.CPP.SimpleSquareCamera import SimpleSquareCameraParams, SimpleSquareCamera
from src.ModelStats import ModelStats
from src.base.GridActions import GridActions, GridActionsNoHover
from src.base.GridPhysics import GridPhysics
from src.h_CPP.Grid import H_CPPGrid


class H_CPPPhysicsParams:
    def __init__(self):
        self.camera_params = SimpleSquareCameraParams()


class H_CPPPhysics(GridPhysics):
    def __init__(self, params: H_CPPPhysicsParams, stats: ModelStats):
        super().__init__()
        self.landed = False


        self.params = params
        self.camera = SimpleSquareCamera(self.params.camera_params)

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

    def step(self, action: GridActions):
        h_term_before = self.state.get_remaining_h_target_cells() == 0
        self.movement_step(action, True)
        if not self.state.terminal:
            self.vision_step()

        if self.state.landed:
            self.landed = True

        # self.state.set_terminal_h(self.state.get_remaining_h_target_cells() == 0)
        if not h_term_before and self.state.get_remaining_h_target_cells() == 0:
            self.state.set_goal_covered(True)

        return self.state

    def vision_step(self):
        view = self.camera.computeView(self.state.position, 0)
        self.state.add_explored(view)
        view_ll = self.camera.computeView(self.state.position, 0, 0)
        self.state.add_explored_h_target(view_ll)

    def get_example_action(self):
        return [self.get_example_action_h(), self.get_example_action_l()]

    def get_example_action_l(self):
        return GridActionsNoHover.WEST

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
        goal = goal.astype(bool)
        self.state.reset_target_h(goal)
        return self.state

    def reset_camera(self, path):
        self.camera.initialize(path)

    def movement_step(self, action: GridActions, hierarchical=False):
        old_position = self.state.position
        x, y = old_position

        # position seems to be initialized as 0 at bottom left corner
        if action == GridActions.NORTH:
            y += 1
        elif action == GridActions.SOUTH:
            y -= 1
        elif action == GridActions.WEST:
            x -= 1
        elif action == GridActions.EAST:
            x += 1
        elif action == GridActions.LAND:
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():
                self.state.set_landed(True)

        self.state.set_position([x, y])
        if self.state.is_in_no_fly_zone():
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        self.state.decrement_movement_budget()
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        # Added code
        # if hierarchical:
        #     self.state.decrement_ll_mb()
        #     self.state.set_terminal_h(self.state.landed or (self.state.current_ll_mb <= 0) or (self.state.movement_budget <= 0) or not (bool(self.state.get_remaining_h_target_cells())))
        self.state.decrement_ll_mb()
        self.state.set_terminal_h(self.state.landed or (self.state.current_ll_mb <= 0) or (self.state.movement_budget <= 0) or not (bool(self.state.get_remaining_h_target_cells())))
        return x, y
