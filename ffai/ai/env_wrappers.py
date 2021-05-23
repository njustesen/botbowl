from abc import ABC

from pytest import set_trace

from ffai.ai.env import FFAIEnv
import gym
import numpy as np


def make_wrapped_env(**kwargs):
    env = FFAIEnv(**kwargs)
    env = FFAI_actionWrapper(env)
    env = FFAI_observation_Wrapper(env, include_action_mask_in_obs=True)
    return env


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


class FFAI_observation_Wrapper(gym.ObservationWrapper, ABC):
    def __init__(self, env: FFAIEnv, include_action_mask_in_obs=False):
        super().__init__(env)

        non_spat_keys = ['state', 'procedures', 'available-action-types']
        num_non_spat_obs = sum([env.observation_space[s].shape[0] for s in non_spat_keys])

        spat_obs = env.observation_space['board']
        non_spat_obs = gym.spaces.Box(low=0, high=1, shape=(num_non_spat_obs,))

        if include_action_mask_in_obs:
            assert isinstance(env, FFAI_actionWrapper)
            action_mask_obs = gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,))
            self.observation_space = gym.spaces.Tuple((spat_obs, non_spat_obs, action_mask_obs))
            self.observation = self.observation_with_action_mask
        else:
            self.observation_space = gym.spaces.Tuple((spat_obs, non_spat_obs))
            self.observation = self.observation_without_action_mask

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.observation(observation) if not done else observation
        return observation, reward, done, info

    def observation_with_action_mask(self, obs):
        return *self.observation_without_action_mask(obs), self.compute_action_masks()

    @staticmethod
    def observation_without_action_mask(obs):
        spatial_obs = np.array(list(obs['board'].values()))

        non_spatial_obs = np.array(list(obs['state'].values()) +
                                   list(obs['procedures'].values()) +
                                   list(obs['available-action-types'].values()))

        #non_spatial_obs = np.expand_dims(non_spatial_obs, axis=0)

        return spatial_obs, non_spatial_obs




