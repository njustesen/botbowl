from itertools import cycle, islice
from random import random

import gym
import torch
from pytest import set_trace

from examples.a3c.a3c_agent import CNNPolicy, A3CAgent
from examples.a3c.a3c_worker_environment import WorkerMemory, VectorMemory, Runner, VectorEnvMultiProcess, VectorEnv
from ffai.ai.bots.random_bot import *


def test_worker_memory():
    env = gym.make("FFAI-1-v3")
    env.reset()

    ac_agent = CNNPolicy(env, hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name="FFAI-1-v3", policy=ac_agent)

    worker_memory = WorkerMemory(10, env)

    for i in range(5):
        (action_dict, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = agent.act_when_training(env)
        env.step(action_dict)
        reward_shaped = i
        action_idx = i
        worker_memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)

    final_value = 10

    worker_memory.calculate_returns(final_value, gamma=0.99)

    assert all([mem == i for mem, i in zip(worker_memory.actions[:5], range(5))])

    assert worker_memory.returns[2] == 2 + 3*0.99 + 4*0.99**2 + final_value*0.99**3
    assert worker_memory.returns[3] == 3 + 4*0.99 + final_value*0.99**2
    assert worker_memory.returns[4] == 4 + final_value*0.99
    assert all([mem == 0 for mem in worker_memory.returns[5:]])


def test_worker_memory_multiple_dones():
    env = gym.make("FFAI-1-v3")
    env.reset()

    ac_agent = CNNPolicy(env, hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name="FFAI-1-v3", policy=ac_agent)

    worker_memory = WorkerMemory(10, env)

    for i in range(10):
        (action_dict, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = agent.act_when_training(env)
        env.step(action_dict)
        reward_shaped = i
        action_idx = i
        worker_memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)

        if i == 4:
            worker_memory.set_done()

    worker_memory.calculate_returns(gamma=1)

    assert worker_memory.returns[0] == 1 + 2 + 3 + 4
    assert worker_memory.returns[1] == 1 + 2 + 3 + 4
    assert worker_memory.returns[2] == 2 + 3 + 4
    assert worker_memory.returns[3] == 3 + 4
    assert worker_memory.returns[4] == 4

    assert worker_memory.returns[5] == 5 + 6 + 7 + 8 + 9
    assert worker_memory.returns[6] == 6 + 7 + 8 + 9
    assert worker_memory.returns[7] == 7 + 8 + 9
    assert worker_memory.returns[8] == 8 + 9
    assert worker_memory.returns[9] == 9

    worker_memory.reset()

    assert worker_memory.step == 0
    assert len(worker_memory.done_indices) == 0

def test_vector_memory():
    env = gym.make("FFAI-1-v3")
    env.reset()

    ac_agent = CNNPolicy(env, hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name="FFAI-1-v3", policy=ac_agent)

    vector_memory = VectorMemory(25, env)

    for i in range(4):
        worker_memory = WorkerMemory(10, env)
        for j in range(10):
            (action_dict, action_idx, action_masks, _, spatial_obs, non_spatial_obs) = agent.act_when_training(env)
            env.step(action_dict)
            reward_shaped = i*j
            action_idx = i*j
            worker_memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)
        worker_memory.calculate_returns()

        vector_memory.insert_worker_memory(worker_memory)

    assert not vector_memory.not_full()

    assert vector_memory.step == vector_memory.size
    assert vector_memory.actions[0] == 0
    assert vector_memory.actions[1] == 0
    assert vector_memory.actions[10] == 0
    assert vector_memory.actions[11] == 1
    assert vector_memory.actions[12] == 2
    assert vector_memory.actions[20] == 0
    assert vector_memory.actions[21] == 2
    assert vector_memory.actions[22] == 4
    assert vector_memory.actions[23] == 6
    assert vector_memory.actions[24] == 8

    try:
        _ = vector_memory.actions[25]
    except IndexError:
        pass
    else:
        raise AssertionError("statement in try block above should've a raised IndexError")


def test_runner():
    mem_size = 10

    env = gym.make("FFAI-1-v3")
    env.reset()

    ac_agent = CNNPolicy(env, hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name="FFAI-1-v3", policy=ac_agent)

    worker_mem = WorkerMemory(mem_size, env)
    runner = Runner(env, agent, worker_mem)

    done = False
    i = 0
    reports = []
    while not done:
        memory, rx_reports = runner.run()
        reports.extend(rx_reports)
        done = len(reports) > 0
        i += 1

        assert memory.step == mem_size

    assert i > 2
    assert reports[0].episodes == 1


def test_simple_vector_environment():
    env_name = "FFAI-1-v3"
    num_procs = 3
    es = [gym.make(env_name) for _ in range(num_procs)]

    ac_agent = CNNPolicy(es[0], hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name=env_name, policy=ac_agent)

    envs = VectorEnv(es, 30, 10)

    completed_episodes = 0

    while completed_episodes < num_procs + 2:

        memory, reports = envs.step(agent)
        completed_episodes += len(reports)



def test_multiprocess_vector_environment():
    env_name = "FFAI-1-v3"
    num_procs = 3
    es = [gym.make(env_name) for _ in range(num_procs)]

    ac_agent = CNNPolicy(es[0], hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name=env_name, policy=ac_agent)

    envs = VectorEnvMultiProcess(es, 30, 10)

    completed_episodes = 0

    while completed_episodes < num_procs+2:
        envs.update_trainee(agent)
        memory, reports = envs.step(agent)
        completed_episodes += len(reports)

    envs.close()

