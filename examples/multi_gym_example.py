import gym
import numpy as np
import ffai.ai
from ffai.ai.env import FFAIEnv
from multiprocessing import Process, Pipe
import numpy as np
from ffai.ai.renderer import Renderer


def worker(remote, parent_remote, env):
    parent_remote.close()

    # Smaller variants
    # env = gym.make("FFAI-7-v1")
    # env = gym.make("FFAI-5-v1")
    # env = gym.make("FFAI-3-v1")

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

            # Render
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


if __name__ == "__main__":

        renderer = Renderer()

        nenvs = 2
        envs = [gym.make("FFAI-v1") for _ in range(nenvs)]
        for i in range(len(envs)):
            seed = envs[i].seed()

        remotes, work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        ps = [Process(target=worker, args=(work_remote, remote, env))
              for (work_remote, remote, env) in zip(work_remotes, remotes, envs)]

        for p in ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()

        for remote in work_remotes:
            remote.close()

        for i in range(100000):
            for remote in remotes:
                remote.send('step')
            results = [remote.recv() for remote in remotes]
            for i in range(len(results)):
                obs, reward, done, info = results[i]
                renderer.render(obs, i)

        for remote in remotes:
            remote.send('close')

        for p in ps:
            p.join()
