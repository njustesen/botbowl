import unittest.mock
from itertools import chain
from random import randint

from botbowl import Action, Square, Formation
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


def test_compute_action():
    env = NewBotBowlEnv()

    for action_type in chain(EnvConf.positional_action_types, EnvConf.simple_action_types):
        if type(action_type) is Formation:
            continue

        sq = None
        if action_type in EnvConf.positional_action_types:
            sq = Square(x=randint(0, env.width-1), y=randint(0, env.height-1))

        action = Action(action_type, position=sq)
        same_action = env._compute_action(env._compute_action_idx(action), flip=False)[0]
        assert action.action_type == same_action.action_type, f"Wrong type: {action} != {same_action}"
        assert action.position == same_action.position, f"Wrong position: {action} != {same_action}"
