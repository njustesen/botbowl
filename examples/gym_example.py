#!/usr/bin/env python3

import gym
import numpy as np
import botbowl


if __name__ == "__main__":

    # Create environment
    # env = gym.make("botbowl-v3")

    # Smaller variants
    # env = gym.make("botbowl-7-v3")
    # env = gym.make("botbowl-5-v3")
    # env = gym.make("botbowl-3-v3")
    env = gym.make("botbowl-1-v3")

    # env.config.pathfinding_enabled = True

    # Get observations space (layer, height, width)
    obs_space = env.observation_space

    # Get action space
    act_space = env.action_space

    # Set seed for reproducibility
    seed = 0
    env.seed(seed)

    # Create random state for action selection
    rnd = np.random.RandomState(seed)

    # Play 10 games
    steps = 0
    for i in range(100):

        # Reset environment
        obs = env.reset()
        done = False

        # Take actions as long as game is not done
        while not done:

            # Extract non-spatial features
            state_arr = list(obs['state'].values())
            board = list(obs['board'].values())
            procedure = obs['procedures']

            # Sample random action type
            action_types = env.available_action_types()
            action_type = rnd.choice(action_types)

            # Sample random position - if any
            available_positions = env.available_positions(action_type)
            pos = rnd.choice(available_positions) if len(available_positions) > 0 else None

            # Create action object
            action = {
                'action-type': action_type,
                'x': pos.x if pos is not None else None,
                'y': pos.y if pos is not None else None
            }

            # Gym step function
            obs, reward, done, info = env.step(action)
            steps += 1

            print(steps)

            # Render
            env.render(feature_layers=True)

    print(steps)
