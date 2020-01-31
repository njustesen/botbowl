import pytest
import gym
import ffai.ai
from multiprocessing import Process, Pipe
import numpy as np

envs = [
    "FFAI-1-v2",
    "FFAI-3-v2",
    "FFAI-5-v2",
    "FFAI-7-v2",
    "FFAI-11-v2"
]


@pytest.mark.parametrize("env", envs)
def test_gym(env):
    env = gym.make(env)
    seed = 0
    env.seed(seed)
    rnd = np.random.RandomState(seed)
    steps = 0
    obs = env.reset()
    done = False
    while not done:
        action_types = env.available_action_types()
        assert len(action_types) > 0
        action_type = rnd.choice(action_types)
        available_positions = env.available_positions(action_type)
        assert obs is not None
        pos = rnd.choice(available_positions) if len(available_positions) > 0 else None
        action = {
            'action-type': action_type,
            'x': pos.x if pos is not None else None,
            'y': pos.y if pos is not None else None
        }
        obs, reward, done, info = env.step(action)
        assert reward is not None
        steps += 1
    assert steps > 10


@pytest.mark.parametrize("env", envs)
def test_multiple_gyms(env):

    def worker(remote, parent_remote, env):
        parent_remote.close()
        seed = env.get_seed()
        rnd = np.random.RandomState(seed)
        steps = 0
        obs = env.reset()
        while True:
            command = remote.recv()
            if command == 'step':
                action_types = env.available_action_types()
                action_type = rnd.choice(action_types)
                available_positions = env.available_positions(action_type)
                pos = rnd.choice(available_positions) if len(available_positions) > 0 else None
                action = {
                    'action-type': action_type,
                    'x': pos.x if pos is not None else None,
                    'y': pos.y if pos is not None else None
                }
                obs, reward, done, info = env.step(action)
                steps += 1
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif command == 'reset':
                obs = env.reset()
                done = False
            elif command == 'close':
                env.close()
                break

    seed = 0
    nenvs = 2
    envs = [gym.make(env) for _ in range(nenvs)]
    for i in range(len(envs)):
        envs[i].seed(seed)
    remotes, work_remotes = zip(*[Pipe() for _ in range(nenvs)])
    ps = [Process(target=worker, args=(work_remote, remote, env))
          for (work_remote, remote, env) in zip(work_remotes, remotes, envs)]
    for p in ps:
        p.daemon = True  # If the main process crashes, we should not cause things to hang
        p.start()
    for remote in work_remotes:
        remote.close()
    for i in range(20):
        for remote in remotes:
            remote.send('step')
        results = [remote.recv() for remote in remotes]
        for j in range(len(results)):
            obs, reward, done, info = results[j]
            assert reward is not None
            assert obs is not None
    for remote in remotes:
        remote.send('close')
    for p in ps:
        p.join()
    assert True
