import numpy as np
import pygame

from src.base.evaluator import PyGameHumanMouse
from src.gym.hcpp import HCPPGym, HCPPGymParams

if __name__ == "__main__":
    params = HCPPGymParams()
    params.map_path = ["res/manhattan32.png"]
    human = PyGameHumanMouse(key_action_mapping=[(pygame.K_SPACE, int(np.prod(params.target_shape).astype(int)))])
    params.render = True
    grid_gym = HCPPGym(params)
    while True:
        obs, info = grid_gym.reset()
        terminated = False
        kill = False
        user_terminate = False
        while not terminated:
            mask = grid_gym.get_action_mask()
            print(mask)
            while True:
                action, user_terminate, kill = human.get_action(grid_gym.position)

                if kill or user_terminate:
                    break
                if mask[action]:
                    break

            if kill or user_terminate:
                break
            next_obs, reward, terminated, truncated, info = grid_gym.step(action)
            print("reward", reward)
            print(info)
        if kill:
            break
        if user_terminate:
            continue
        # Wait for key press
        _, _, kill = human.get_action(grid_gym.position)
        if kill:
            break

    grid_gym.close()