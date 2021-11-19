from src.base.GridActions import GridActions


class GridPhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None

    def movement_step(self, action: GridActions):
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
            print('action landing!!!!!')
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():
                self.state.set_landed(True)

        self.state.set_position([x, y])
        if self.state.is_in_no_fly_zone():
            # Reset state
            # print('@@@@@@@@ in nfz @@@@@@@@@@')
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        self.state.decrement_movement_budget()
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        # Added code
        self.state.decrement_ll_mb()
        self.state.set_terminal_h(self.state.landed or (self.state.current_ll_mb <= 0) or (self.state.movement_budget <= 0) or not (bool(self.state.get_remaining_h_target_cells())))
        # if self.state.is_terminal():
            # print(f'################## state terminal !!!!!!!!!! \n remaining target: {self.state.get_remaining_cells()}')
        # if self.state.is_terminal_h():
            # print(f'Remaining h-target: {self.state.get_remaining_h_target_cells()} \n remaining target: {self.state.get_remaining_cells()}')

        return x, y

    def reset(self, state):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state
