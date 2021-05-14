from tests.util import *
import pytest
import ffai.ai.env as env
import ffai.ai.env_wrappers as wrappers
import gym


def test_action_wrapper():
    env = gym.make("FFAI-v3")
    env = wrappers.FFAI_actionWrapper(env)
    env.reset()

    non_spatial_action_types = env.simple_action_types + env.defensive_formation_action_types + \
                   env.offensive_formation_action_types

    done = False
    while not done:
        mask = env.compute_action_masks()
        action_index = np.random.choice(mask.nonzero()[0])
        action_type, x, y = env.compute_action(action_index)

        assert action_type in env.available_action_types()
        if action_type in env.positional_action_types:
            assert Square(x, y) in env.available_positions(action_type)
        else:
            assert action_type in non_spatial_action_types

        _, _, done, _ = env.step(action_index)






