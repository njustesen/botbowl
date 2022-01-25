#!/usr/bin/env python3

import gym
import numpy as np
from botbowl import BotBowlEnv


def main():
    env = BotBowlEnv()
    steps = 0

    # Play 10 games
    for _ in range(10):
        env.reset()
        done = False
        _, _, mask = env.get_state()

        while not done:
            env.render(feature_layers=True)
            aa = np.where(mask > 0.0)[0]
            action_idx = np.random.choice(aa, 1)[0]
            _, _, done, info = env.step(action_idx)
            mask = info['action_mask']
            steps += 1
            print(steps)


if __name__ == "__main__":
    main()
