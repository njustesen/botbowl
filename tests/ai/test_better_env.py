from botbowl.ai.new_env import NewBotBowlEnv
import numpy as np


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
