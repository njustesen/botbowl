from itertools import cycle, islice
from random import random

import gym
import torch
from pytest import set_trace

from examples.a3c.a3c_agent import CNNPolicy, A3CAgent
from examples.a3c.a3c_worker_environment import WorkerMemory, VectorMemory, Runner, VectorEnvMultiProcess
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

    worker_memory.insert_epside_end(final_value)

    assert not worker_memory.looped
    assert all([mem == i for mem, i in zip(worker_memory.actions[:5], range(5))])

    assert worker_memory.returns[2] == 2 + 3*0.99 + 4*0.99**2 + final_value*0.99**3
    assert worker_memory.returns[3] == 3 + 4*0.99 + final_value*0.99**2
    assert worker_memory.returns[4] == 4 + final_value*0.99
    assert all([mem == 0 for mem in worker_memory.returns[5:]])


def test_worker_memory_looped():
    env = gym.make("FFAI-1-v3")
    env.reset()

    ac_agent = CNNPolicy(env, hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name="FFAI-1-v3", policy=ac_agent)

    worker_memory = WorkerMemory(10, env)

    for i in range(15):
        (action_dict, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = agent.act_when_training(env)
        env.step(action_dict)
        reward_shaped = i
        action_idx = i
        worker_memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)

    worker_memory.insert_epside_end()

    assert worker_memory.looped
    assert all([mem == i for mem, i in zip(worker_memory.actions, range(10, 15))])

    assert worker_memory.returns[4] == 14

    # 9 + 10 * 0.99 + 11 * 0.99 ** 2 + 12 * 0.99 ** 3 + 13 * 0.99 ** 4 + 14 * 0.99 ** 5
    assert worker_memory.returns[-1] == sum([reward*0.99**exponent for reward, exponent in zip(range(9, 15), range(0, 6))])


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
        worker_memory.insert_epside_end()

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
    report = None
    while not done:
        memory, report = runner.run()
        done = report.episodes > 0
        i += 1

    assert i > 2
    assert report.episodes == 1
    assert runner.episode_steps == 0
    assert runner.total_reward == 0


#def test_simple_vector_environment():
#    pass


def test_multiprocess_vector_environment():
    env_name = "FFAI-1-v3"
    num_procs = 3
    es = [gym.make(env_name) for _ in range(num_procs)]

    ac_agent = CNNPolicy(es[0], hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name=env_name, policy=ac_agent)

    envs = VectorEnvMultiProcess(es, agent, 30, 10)

    completed_episodes = 0

    while completed_episodes < num_procs+2:
        memory, report = envs.step(agent)
        completed_episodes += report.episodes

    envs.close()

