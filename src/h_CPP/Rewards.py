from src.h_CPP.State import H_CPPState
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewardParams, GridRewards

class H_CPPRewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.cell_multiplier = 0.4
        self.invalid_goal_penalty = -10


# Class used to track rewards
class H_CPPRewards(GridRewards):

    def __init__(self, reward_params: H_CPPRewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

        self.l_cumulative_reward: float = 0.0
        self.h_cumulative_reward: float = 0.0

    def calculate_reward_h(self, state: H_CPPState, action: GridActions, next_state: H_CPPState):
        reward = self.calculate_motion_rewards(state, action, next_state)

        # Reward the collected data
        reward += self.params.cell_multiplier * (state.get_remaining_cells() - next_state.get_remaining_cells())

        # Cumulative reward
        self.h_cumulative_reward += reward

        return reward

    def calculate_reward_l(self, state: H_CPPState, action: GridActions, next_state: H_CPPState):
        reward = self.calculate_motion_rewards(state, action, next_state)

        #reward collected target area
        reward += self.params.cell_multiplier * (state.get_remaining_h_target_cells() - next_state.get_remaining_h_target_cells())

        #cumulative reward_l
        self.l_cumulative_reward += reward

        return reward

    def invalid_goal_penalty(self):
        reward = self.params.invalid_goal_penalty
        self.h_cumulative_reward += reward

        return reward

