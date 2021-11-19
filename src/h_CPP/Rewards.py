from src.h_CPP.State import H_CPPState
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewardParams, GridRewards


class H_CPPRewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.cell_multiplier = 0.4
        self.cell_multiplier_ll = 1
        self.invalid_goal_penalty = 1.
        self.goal_reached_bonus = 2.


# Class used to track rewards
class H_CPPRewards(GridRewards):

    def __init__(self, reward_params: H_CPPRewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

        self.l_cumulative_reward: float = 0.0
        self.h_cumulative_reward: float = 0.0

    def calculate_reward_h(self, state: H_CPPState, action, next_state: H_CPPState, valid, tried_landing_and_succeeded):
        reward = self.params.cell_multiplier * (state.get_remaining_cells() - next_state.get_remaining_cells())

        if not valid:
            reward -= self.params.invalid_goal_penalty

        if next_state.goal_covered:
            reward += self.params.goal_reached_bonus

        if tried_landing_and_succeeded == None:
            # print(f'Landing reward: passing {landed_and_succeded}')
            pass
        elif tried_landing_and_succeeded == True:
            print(f'Landing reward: succeded {tried_landing_and_succeeded}')
            # reward += 2
        elif tried_landing_and_succeeded == False:
            print(f'Landing reward: failed {tried_landing_and_succeeded}')
            reward -= self.params.boundary_penalty

        # Cumulative reward
        self.h_cumulative_reward += reward



        return reward

    # def calculate_reward_h_per_step(self, state: H_CPPState, action: GridActions, next_state: H_CPPState, valid,
    #                                 landed_and_succeeded):
    #
    #     # Reward the collected data
    #     # reward = self.params.cell_multiplier * (state.get_remaining_cells() - next_state.get_remaining_cells())
    #
    #     if landed_and_succeeded == None:
    #         # print(f'Landing reward: passing {landed_and_succeded}')
    #         pass
    #     elif landed_and_succeeded == True:
    #         print(f'Landing reward: succeded {landed_and_succeeded}')
    #         # reward += 2
    #     elif landed_and_succeeded == False:
    #         print(f'Landing reward: failed {landed_and_succeeded}')
    #         # reward -= 2
    #
    #     # Cumulative reward
    #     self.h_cumulative_reward += reward
    #
    #     return reward

    def calculate_reward_l(self, state: H_CPPState, action: GridActions, next_state: H_CPPState):
        reward = self.calculate_motion_rewards_l(state, action, next_state)

        # reward collected target area
        reward += self.params.cell_multiplier_ll * (
                    state.get_remaining_h_target_cells() - next_state.get_remaining_h_target_cells())

        # cumulative reward_l
        self.l_cumulative_reward += reward

        return reward

    def calculate_motion_rewards_l(self, state, action: GridActions, next_state):
        reward = 0.0
        reward -= self.params.movement_penalty

        # Penalize not moving (This happens when it either tries to land or fly into a boundary or hovers or fly into
        # a cell occupied by another agent)
        if state.position == next_state.position and not next_state.landed and not action == GridActions.HOVER:
            reward -= self.params.boundary_penalty

        # Penalize battery dead
        if next_state.current_ll_mb == 0 and not next_state.landed:
            reward -= self.params.empty_battery_penalty

        return reward
