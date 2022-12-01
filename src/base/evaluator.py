from dataclasses import dataclass

import numpy as np
import pygame
from tqdm import tqdm


class PyGameHuman:
    def __init__(self, key_action_mapping: list[tuple[int, int]], terminate_key=pygame.K_t, kill_key=pygame.K_q):
        self.kill_key = kill_key
        self.terminate_key = terminate_key
        self.key_action_mapping = key_action_mapping

    def get_action(self, position) -> (int, bool, bool):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return 0, False, True
                elif event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[self.kill_key]:
                        return 0, False, True
                    if keys[self.terminate_key]:
                        return 0, True, False
                    else:
                        for key, action in self.key_action_mapping:
                            if keys[key]:
                                return action, False, False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = np.array(pygame.mouse.get_pos())

                    print(pos)

    def get_action_non_blocking(self, position) -> (int, bool, bool):
        keys = pygame.key.get_pressed()
        if keys[self.kill_key]:
            return None, False, True
        if keys[self.terminate_key]:
            return None, True, False
        else:
            for key, action in self.key_action_mapping:
                if keys[key]:
                    return action, False, False
        return None, False, False


class PyGameHumanMouse:
    def __init__(self, key_action_mapping: list[tuple[int, int]], window_size=768, cells=32, target_size=(17, 17),
                 terminate_key=pygame.K_t,
                 kill_key=pygame.K_q):
        self.kill_key = kill_key
        self.terminate_key = terminate_key
        self.key_action_mapping = key_action_mapping
        self.window_size = window_size
        self.cells = cells
        self.target_size = np.array(target_size)

    def get_action(self, position) -> (int, bool, bool):
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return 0, False, True
                elif event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[self.kill_key]:
                        return 0, False, True
                    if keys[self.terminate_key]:
                        return 0, True, False
                    else:
                        for key, action in self.key_action_mapping:
                            if keys[key]:
                                return action, False, False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = np.array(pygame.mouse.get_pos())
                    action = (pos * self.cells / self.window_size).astype(int)
                    action = (action - position) + self.target_size // 2

                    if (action >= 0).all() and (action < self.target_size).all():
                        flat_action = action[0] * self.target_size[0] + action[1]
                        return flat_action, False, False

    def get_action_non_blocking(self, position) -> (int, bool, bool):
        keys = pygame.key.get_pressed()
        if keys[self.kill_key]:
            return None, False, True
        if keys[self.terminate_key]:
            return None, True, False
        pressed = pygame.mouse.get_pressed()
        if pressed[0]:
            pos = np.array(pygame.mouse.get_pos())
            action = (pos * self.cells / self.window_size).astype(int)
            action = (action - position) + self.target_size // 2

            if (action >= 0).all() and (action < self.target_size).all():
                flat_action = action[0] * self.target_size[0] + action[1]
                return flat_action, False, False
        return None, False, False


@dataclass
class EvaluatorParams:
    show_eval: bool = False
    use_softmax: bool = False


class Evaluator:

    def __init__(self, params: EvaluatorParams, trainer, gym, human=None):
        self.params = params
        self.trainer = trainer
        self.gym = gym.__class__(gym.params)
        for render in gym.render_registry:
            self.gym.register_render(render["render_func"], render["shape"])
        if self.params.show_eval:
            self.gym.params.render = True

        self.human = human
        self.mode = "step"

    def evaluate_episode(self):
        state, info = self.gym.reset()
        obs = self.trainer.prepare_observations(state)
        terminal = False
        while not terminal:
            if self.params.use_softmax:
                action = self.trainer.get_action(obs)
            else:
                action = self.trainer.get_exploit_action(obs)
            state, reward, terminal, truncated, info = self.gym.step(action)
            obs = self.trainer.prepare_observations(state)

        if self.params.show_eval:
            self.gym.close()
        return info

    def evaluate_multiple_episodes(self, n):
        stats = []
        for _ in tqdm(range(n)):
            stats.append(self.evaluate_episode())
        return stats

    def evaluate_episode_interactive(self):
        if self.human is None:
            print("No interface configured. Cannot do interactive.")
            return
        state, info = self.gym.reset()
        obs = self.trainer.prepare_observations(state)
        terminal = False
        while not terminal:
            action = self.handle_events()
            if action is None:
                if self.params.use_softmax:
                    action = self.trainer.get_action(obs)
                else:
                    action = self.trainer.get_exploit_action(obs)
            state, reward, terminal, truncated, info = self.gym.step(action)
            obs = self.trainer.prepare_observations(state)
        return info

    def handle_events(self):
        while True:
            action = None
            key_pressed = False
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    keys = pygame.key.get_pressed()
                    key_pressed = True
                    if keys[pygame.K_h]:
                        self.mode = "human"
                    elif keys[pygame.K_s]:
                        self.mode = "step"
                    elif keys[pygame.K_r]:
                        self.mode = "run"
                    action, terminate, kill = self.human.get_action_non_blocking(self.gym.position)
                    if kill:
                        exit(0)

            if self.mode == "run":
                return None
            if self.mode == "human":
                if action is not None:
                    return action
            if self.mode == "step":
                if key_pressed:
                    return None
