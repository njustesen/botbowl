#!/usr/bin/env python3

import gym
import numpy as np
import botbowl
from multiprocessing import Process, Pipe
import botbowl


def worker(remote, parent_remote, env):
    parent_remote.close()

    # Get observations space (layer, height, width)
    obs_space = env.observation_space

    # Get action space
    act_space = env.action_space

    # Create random state for action selection
    seed = env.get_seed()
    rnd = np.random.RandomState(seed)

    # Play 10 games
    steps = 0

    # Reset environment
    obs = env.reset()

    while True:
        command = remote.recv()

        if command == 'step':

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

            # Render - Does not work when running multiple processes
            # env.render(feature_layers=False)

            if done:
                obs = env.reset()

            remote.send((obs, reward, done, info))

        elif command == 'reset':

            # Reset environment
            obs = env.reset()
            done = False

        elif command == 'close':

            # Close environment
            env.close()
            break


if __name__ == "__main__":

    renderer = botbowl.Renderer()

    nenvs = 8
    envs = [gym.make("botbowl-1-v3") for _ in range(nenvs)]
    for i in range(len(envs)):
        envs[i].seed()

    remotes, work_remotes = zip(*[Pipe() for _ in range(nenvs)])

    ps = [Process(target=worker, args=(work_remote, remote, env))
          for (work_remote, remote, env) in zip(work_remotes, remotes, envs)]

    for p in ps:
        p.daemon = True  # If the main process crashes, we should not cause things to hang
        p.start()

    for remote in work_remotes:
        remote.close()

    for i in range(1000):
        print(i)
        for remote in remotes:
            remote.send('step')
        results = [remote.recv() for remote in remotes]
        for j in range(len(results)):
            obs, reward, done, info = results[j]
            renderer.render(obs, j)

    for remote in remotes:
        remote.send('close')

    for p in ps:
        p.join()
