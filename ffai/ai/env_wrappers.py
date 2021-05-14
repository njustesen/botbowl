from abc import ABC

from ffai.ai.env import FFAIEnv
import gym
import numpy as np


class FFAI_actionWrapper(gym.ActionWrapper, ABC):
    def __init__(self, env: FFAIEnv):
        super().__init__(env)

        self.x_max = env.action_space["x"].n
        self.y_max = env.action_space["y"].n
        self.board_squares = self.x_max * self.y_max

        self.non_spatial_action_types = env.simple_action_types + \
                                        env.defensive_formation_action_types + \
                                        env.offensive_formation_action_types

        self.num_spat_actions = len(self.env.positional_action_types)
        self.num_non_spat_action = len(self.non_spatial_action_types)

        self.action_space = gym.spaces.Discrete(
            self.num_non_spat_action + self.board_squares * self.num_spat_actions)

        self.action_mask = None

    def action(self, action):
        action_type, x, y = self.compute_action(action)

        return {'action-type': action_type,
                'x': x,
                'y': y}

    def compute_action(self, action_idx):

        assert action_idx in self.action_mask.nonzero()[0]

        if action_idx < self.num_non_spat_action:
            return self.non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - self.num_non_spat_action
        spatial_pos_idx = spatial_idx % self.board_squares
        spatial_y = int(spatial_pos_idx / self.x_max)
        spatial_x = int(spatial_pos_idx % self.x_max)
        spatial_action_type_idx = int(spatial_idx / self.board_squares)
        spatial_action_type = self.env.positional_action_types[spatial_action_type_idx]

        return spatial_action_type, spatial_x, spatial_y

    def compute_action_masks(self):

        mask = np.zeros(self.action_space.n)

        for action_type in self.env.available_action_types():
            if action_type in self.non_spatial_action_types:
                index = self.non_spatial_action_types.index(action_type)
                mask[index] = 1
            elif action_type in self.env.positional_action_types:
                action_start_index = self.num_non_spat_action + \
                                     self.env.positional_action_types.index(action_type)*self.board_squares

                for pos in self.env.available_positions(action_type):
                    mask[action_start_index + pos.x + pos.y * self.x_max] = 1

        assert 1 in mask, "No available action in action_mask"
        self.action_mask = mask
        return mask
