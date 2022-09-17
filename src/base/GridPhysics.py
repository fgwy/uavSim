from src.base.GridActions import GridActionsDiagonal


class GridPhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None

    def movement_step(self, action: GridActionsDiagonal, hierarchical=False):
        old_position = self.state.position
        x, y = old_position
        d = False

        # NORTH = 0  # down on map array y+1
        # NORTH_EAST = 1
        # EAST = 2  # x+1
        # SOUTH_EAST = 3
        # SOUTH = 4  # y-1
        # SOUTH_WEST = 5
        # WEST = 6  # x-1
        # NORTH_WEST = 7

        # position seems to be initialized as 0 at bottom left corner
        if action == GridActionsDiagonal.NORTH:
            y += 1
        elif action == GridActionsDiagonal.SOUTH:
            y -= 1
        elif action == GridActionsDiagonal.WEST:
            x -= 1
        elif action == GridActionsDiagonal.EAST:
            x += 1

        #diagonal actions
        elif action == GridActionsDiagonal.NORTH_EAST:
            y+=1
            x+=1
            d=True
        elif action == GridActionsDiagonal.NORTH_WEST:
            y+=1
            x-=1
            d = True
        elif action == GridActionsDiagonal.SOUTH_EAST:
            y-=1
            x+=1
            d = True
        elif action == GridActionsDiagonal.SOUTH_WEST:
            y-=1
            x-=1
            d = True


        elif action == GridActionsDiagonal.LAND:
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():
                self.state.set_landed(True)

        self.state.set_position([x, y])
        if self.state.is_in_no_fly_zone():
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        # print('checking diagonal corner',
        #       self.state.no_fly_zone[old_position[1], x] or self.state.no_fly_zone[y, old_position[0]])
        if self.state.no_fly_zone[old_position[1], x] or self.state.no_fly_zone[y, old_position[0]]:
            print('hit diagonal corner')
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        self.state.decrement_movement_budget(d)
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        # Added code
        if self.state.hierarchical:
            self.state.decrement_ll_mb(d)
            self.state.set_terminal_h(self.state.landed or (self.state.current_ll_mb <= 0) or (self.state.movement_budget <= 0) or not (bool(self.state.get_remaining_h_target_cells())))

        # print('big printout on state and termination!')
        # self.state.decrement_ll_mb()
        # self.state.set_terminal_h(self.state.landed or (self.state.current_ll_mb <= 0) or (self.state.movement_budget <= 0) or not (bool(self.state.get_remaining_h_target_cells())))
        return x, y

    def reset(self, state):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state
