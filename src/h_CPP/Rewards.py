from src.h_CPP.State import h_CPPState
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewardParams
from src.CPP.Rewards import CPPRewards


class CPPRewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.cell_multiplier = 0.4


# Class used to track rewards
class H_CPPRewards(CPPRewards):

    def __init__(self, reward_params: CPPRewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

        self.l_cumulative_reward: float = 0.0
        self.h_cumulative_reward: float = 0.0

    def calculate_reward_h(self, state: h_CPPState, action: GridActions, next_state: h_CPPState):
        reward = self.calculate_motion_rewards(state, action, next_state)

        # Reward the collected data
        reward += self.params.cell_multiplier * (state.get_remaining_cells() - next_state.get_remaining_cells())

        # Cumulative reward
        self.h_cumulative_reward += reward

        return reward

    def calculate_reward_l(self, state: h_CPPState, action: GridActions, next_state: h_CPPState):
        reward = self.calculate_motion_rewards(state, action, next_state)

        #reward collected target area
        reward += self.params.cell_multiplier * (state.get_remaining_h_target_cells() - next_state.get_remaining_h_target_cells())

        #cumulative reward_l
        self.l_cumulative_reward += reward

        return reward
