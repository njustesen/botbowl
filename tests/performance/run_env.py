#!/usr/bin/env python3

import gym
import numpy as np
import ffai

import cProfile
import io 
import pstats 
#from pstats import SortKey

def run_env(n, env_name="FFAI-11-v2"):  
    # Create environment
    env = gym.make( env_name )

    # Get observations space (layer, height, width)
    obs_space = env.observation_space

    # Get action space
    act_space = env.action_space

    # Set seed for reproducibility
    seed = 0
    env.seed(seed)

    # Create random state for action selection
    rnd = np.random.RandomState(seed)

    # Play n games
    steps = 0
    for i in range(n):

        # Reset environment
        obs = env.reset()
        done = False

        # Take actions as long as game is not done
        while not done:

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


if __name__ == "__main__":
    
    pr = cProfile.Profile() 
    
    pr.enable()
    run_env(10)
    pr.disable()
    
    s = io.StringIO()
    #sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(50)
    print(s.getvalue())
    