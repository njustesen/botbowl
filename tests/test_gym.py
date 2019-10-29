import pytest
import gym
import numpy as np

envs = [
    "FFAI-1-v1",
    "FFAI-3-v1",
    "FFAI-5-v1",
    "FFAI-7-v1",
    "FFAI-11-v1"
]


@pytest.mark.parametrize("env", envs)
def test_gym(env):
    env = gym.make(env)
    seed = 0
    env.seed(seed)
    rnd = np.random.RandomState(seed)
    steps = 0
    obs = env.reset()
    done = False
    while not done:
        action_types = env.available_action_types()
        assert len(action_types) > 0
        action_type = rnd.choice(action_types)
        available_positions = env.available_positions(action_type)
        assert obs is not None
        pos = rnd.choice(available_positions) if len(available_positions) > 0 else None
        action = {
            'action-type': action_type,
            'x': pos.x if pos is not None else None,
            'y': pos.y if pos is not None else None
        }
        obs, reward, done, info = env.step(action)
        assert reward is not None
        steps += 1
    assert steps > 10


