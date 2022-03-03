#!/usr/bin/env python3

import gym
import numpy as np
from multiprocessing import Process, Pipe
from botbowl.ai.env_render import EnvRenderer


def worker(remote, parent_remote, env):
    parent_remote.close()

    # Create random state for action selection
    rnd = np.random.RandomState(env.seed())

    # Reset environment
    spatial_obs, non_spatial_obs, mask = env.reset()
    while True:
        command = remote.recv()

        if command == 'step':
            aa = np.where(mask > 0.0)[0]
            action_idx = rnd.choice(aa, 1)[0]
            (spatial_obs, non_spatial_obs, mask), reward, done, info = env.step(action_idx)

            if done:
                spatial_obs, non_spatial_obs, mask = env.reset()
            remote.send(env.game)

        elif command == 'reset':
            spatial_obs, non_spatial_obs, mask = env.reset()
            remote.send(env.game)

        elif command == 'close':
            env.close()
            break


def main():
    nenvs = 4
    envs = []
    renderers = []

    for _ in range(nenvs):
        env = gym.make("botbowl-1-v4")
        envs.append(env)
        renderers.append(EnvRenderer(env))

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
        received_games = [remote.recv() for remote in remotes]

        for game, renderer in zip(received_games, renderers):
            renderer.env.game = game
            renderer.render()

    for remote in remotes:
        remote.send('close')

    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
