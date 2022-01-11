import unittest.mock
from itertools import chain
from random import randint
from typing import Optional

from botbowl import Action, Square, Formation, Game, ActionType
from botbowl.ai.new_env import NewBotBowlEnv, ScriptedActionWrapper, RewardWrapper, EnvConf
import numpy as np

from examples.a2c.a2c_env import A2C_Reward


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


def test_compute_action():
    env = NewBotBowlEnv()

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
    """
    class RewardFunc():
        def __init__(self):
            self.last_ball_pos_x = None

        def __call__(self, game: Game) -> float:
            ball_pos = game.get_ball_position()
            ball_pos_x = ball_pos.x if ball_pos is not None else None
            reward = 0.0
            if ball_pos_x is not None and self.last_ball_pos_x is not None:
                reward = ball_pos_x - self.last_ball_pos_x
                reward *= -1.0 if game.active_team is game.state.home_team else 1.0
            self.last_ball_pos_x = ball_pos_x
            return reward
    reward_func = RewardFunc()
    """
    reward_func = A2C_Reward()

    def scripted_func(game) -> Optional[Action]:
        available_action_types = [action_choice.action_type for action_choice in game.get_available_actions()]

        if len(available_action_types) == 1 and len(game.get_available_actions()[0].positions) == 0 and len(game.get_available_actions()[0].players) == 0:
            return Action(available_action_types[0])

        if ActionType.END_PLAYER_TURN in available_action_types:
            return Action(ActionType.END_PLAYER_TURN)
        else:
            return None

    env = NewBotBowlEnv(EnvConf(size=1))
    #env = ScriptedActionWrapper(env, scripted_func)
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
