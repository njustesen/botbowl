from bb.web.api import *
import gym
import numpy as np


if __name__ == "__main__":

    # Create environment
    env = gym.make("FFAI-v1")

    # Smaller variants
    # env = gym.make("FFAI-7-v1")
    # env = gym.make("FFAI-5-v1")
    # env = gym.make("FFAI-3-v1")

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
    for i in range(10):

        # Reset environment
        obs = env.reset()
        done = False

        # Take actions as long as game is not done
        while not done:

            # Extract non-spatial features
            non_spatial = obs["non-spatial"]
            non_spatial_arr = list(non_spatial.values())
            spatial = obs["spatial"]
            spatial_arr = list(spatial.values())

            # Sample random action type
            action_type = rnd.choice(env.available_action_types())

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

            # Render
            env.render(feature_layers=False)

    print(steps)
