import unittest.mock
from itertools import chain
from random import randint
from typing import Optional

import pytest

from botbowl import Action, Square, Formation, Game, ActionType
from botbowl.ai.env import BotBowlEnv, ScriptedActionWrapper, RewardWrapper, EnvConf
import numpy as np

from examples.a2c.a2c_env import A2C_Reward
import gym


@pytest.mark.parametrize("name", ['botbowl-v4',
                                  'botbowl-11-v4',
                                  'botbowl-7-v4',
                                  'botbowl-5-v4',
                                  'botbowl-3-v4',
                                  'botbowl-1-v4'])
def test_gym_registry(name):
    env = gym.make(name)
    env.reset()

    done = False
    _, _, mask = env.get_state()

    while not done:
        aa = np.where(mask > 0.0)[0]
        action_idx = np.random.choice(aa, 1)[0]
        obs, reward, done, info = env.step(action_idx)
        mask = info['action_mask']


def test_compute_action():
    env = BotBowlEnv()

    for action_type in chain(env.env_conf.positional_action_types, env.env_conf.simple_action_types):
        if type(action_type) is Formation:
            continue

        sq = None
        if action_type in env.env_conf.positional_action_types:
            sq = Square(x=randint(0, env.width-1), y=randint(0, env.height-1))

        action = Action(action_type, position=sq)
        same_action = env._compute_action(env._compute_action_idx(action), flip=False)[0]
        assert action.action_type == same_action.action_type, f"Wrong type: {action} != {same_action}"
        assert action.position == same_action.position, f"Wrong position: {action} != {same_action}"


def test_reward_and_scripted_wrapper():

    reward_func = A2C_Reward()

    def scripted_func(game) -> Optional[Action]:
        available_action_types = [action_choice.action_type for action_choice in game.get_available_actions()]

        if len(available_action_types) == 1 and len(game.get_available_actions()[0].positions) == 0 and len(game.get_available_actions()[0].players) == 0:
            return Action(available_action_types[0])

        if ActionType.END_PLAYER_TURN in available_action_types and randint(1, 5) == 2:
            return Action(ActionType.END_PLAYER_TURN)

        return None

    env = BotBowlEnv(EnvConf(size=1))
    env = ScriptedActionWrapper(env, scripted_func)
    env = RewardWrapper(env, home_reward_func=reward_func)

    rewards = []
    own_tds = []
    opp_tds = []

    for _ in range(10):
        env.reset()
        done = False
        _, _, mask = env.get_state()

        ep_reward = 0.0

        while not done:
            aa = np.where(mask)[0]
            action_idx = np.random.choice(aa, 1)[0]
            obs, reward, done, info = env.step(action_idx)
            ep_reward += reward
            mask = info['action_mask']

        rewards.append(ep_reward)
        own_tds.append(env.game.state.home_team.state.score)
        opp_tds.append(env.game.state.away_team.state.score)

    print("\n\n")
    print(np.mean(rewards))
    print(f"td rate: {np.mean(own_tds)} vs. {np.mean(opp_tds)}")


def test_observation_ranges():
    def find_first_index(array_: np.ndarray, value_: float):
        indices = (array_ == value_).nonzero()
        return [x[0] for x in indices]

    env = BotBowlEnv()

    for _ in range(2):
        done = False
        env.reset()

        while not done:
            _, non_spatial_obs, mask = env.get_state()

            # Spatial observation are within [0, 1]
            for layer in env.env_conf.layers:
                layer_name = layer.name()
                array = layer.produce(env.game)

                max_val = np.max(array)
                min_val = np.min(array)

                assert max_val <= 1.0, \
                    f"['{layer_name}'][{find_first_index(array, max_val)}] is too high ({max_val})"

                assert min_val >= 0.0, \
                    f"['{layer_name}'][{find_first_index(array, min_val)}] is too low ({min_val})"

            max_val = np.max(non_spatial_obs)
            min_val = np.min(non_spatial_obs)

            assert min_val >= 0.0, \
                f"non_spatial_obs[{find_first_index(non_spatial_obs, min_val)}] is too low ({min_val})"

            assert max_val <= 1.0, \
                f"non_spatial_obs[{find_first_index(non_spatial_obs, max_val)}] is too high ({max_val})"

            aa = np.where(mask > 0.0)[0]
            action_idx = np.random.choice(aa, 1)[0]
            _, _, done, _ = env.step(action_idx, skip_observation=True)

    env.close()
