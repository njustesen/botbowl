from tests.util import *
import gym
from tests.performance.run_env import get_random_action_from_env

import numpy as np

import pytest
import ffai.ai.env as env
import ffai.ai.env_wrappers as wrappers


def test_observation_ranges():
    def find_first_index(array_, value_):
        indices = (array_ == value_).nonzero()
        return [x[0] for x in indices]

    env = gym.make("FFAI-v3")
    rnd = np.random.RandomState(np.random.randint(0, 2 ** 16))

    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            for layer_name, array in obs['board'].items():
                max_val = np.max(array)
                assert max_val <= 1.0, \
                    f"obs['board']['{layer_name}'][{find_first_index(array, max_val)}] is too high ({max_val})"
                min_val = np.min(array)
                assert min_val >= 0.0, \
                    f"obs['board']['{layer_name}'][{find_first_index(array, min_val)}] is too low ({min_val})"

            for obs_key in ['state', 'procedures', 'available-action-types']:
                for key_name, value in obs[obs_key].items():
                    assert 0.0 <= value <= 1.0, \
                        f"obs['{obs_key}']['{key_name}'] is too {'high' if value>1.0 else 'low'}: {value}"

            obs, _, done, _ = env.step(get_random_action_from_env(env, rnd))
    env.close()


def test_action_wrapper():
    env = gym.make("FFAI-v3")
    env = wrappers.FFAI_actionWrapper(env)
    env.reset()

    non_spatial_action_types = env.simple_action_types + env.defensive_formation_action_types + \
                   env.offensive_formation_action_types

    done = False
    while not done:
        action_mask = env.compute_action_masks()
        action_index = np.random.choice(action_mask.nonzero()[0])
        action_type, x, y = env.compute_action(action_index)

        assert action_type in env.available_action_types()
        if action_type in env.positional_action_types:
            assert Square(x, y) in env.available_positions(action_type)
        else:
            assert action_type in non_spatial_action_types

        _, _, done, _ = env.step(action_index)


def test_observation_wrapper():
    env = gym.make("FFAI-v3")
    env = wrappers.FFAI_observation_Wrapper(env)
    obs = env.reset()

    assert len(obs) == 2
    assert obs[0].shape == (len(env.layers), 17, 28)
    assert obs[1].shape == (116, )


def test_fully_wrapped():
    env = gym.make("FFAI-wrapped-v3")
    obs = env.reset()
    done = False
    while not done:
        assert len(obs) == 3
        assert obs[0].shape == (len(env.layers), 17, 28)
        assert obs[1].shape == (116,)
        assert obs[2].shape == (8117,)

        _, _, action_mask = obs
        action_index = np.random.choice(action_mask.nonzero()[0])
        obs, _, done, _ = env.step(action_index)


