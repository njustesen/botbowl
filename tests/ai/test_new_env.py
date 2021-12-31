import unittest.mock

from botbowl.ai.new_env import NewBotBowlEnv, EnvConf
import numpy as np


def test_fast_new_env():
    aa_layer_first_index = len(EnvConf.layers) - len(EnvConf.positional_action_types)
    layers = EnvConf.layers[aa_layer_first_index:]

    with unittest.mock.patch('botbowl.ai.new_env.EnvConf.layers', layers):
        test_env()


def test_env():
    env = NewBotBowlEnv()
    env.reset()

    done = False
    reward = None
    _, _, mask = env.get_state()

    while not done:
        aa = np.where(mask > 0.0)[0]
        action_idx = np.random.choice(aa, 1)[0]
        obs, reward, done, info = env.step(action_idx)
        mask = info['action_mask']

    if reward != 0:
        assert env.game.state.home_team.state.score != env.game.state.away_team.state.score
