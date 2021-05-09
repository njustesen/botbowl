import gym

from examples.a3c.a3c_agent import CNNPolicy, A3CAgent
from examples.a3c.a3c_worker_environment import VectorEnvMultiProcess, VectorMemory, EndOfGameReport

import torch.optim as optim
from ffai.ai.layers import *


import time

import a3c_config as conf


# Make directories
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)




def main(training_name):


    env_name = "FFAI-1-v3"
    num_procs = 16
    worker_mem_size = 120
    es = [gym.make(env_name) for _ in range(num_procs)]

    ac_agent = CNNPolicy(es[0], hidden_nodes=10, kernels=[10, 10])
    agent = A3CAgent("trainee", env_name=env_name, policy=ac_agent)

    envs = VectorEnvMultiProcess(es, min_batch_size=worker_mem_size * num_procs, worker_memory_size=worker_mem_size)

    completed_episodes = 0
    total_steps = 0
    start_time = time.time()

    while completed_episodes < num_procs:
        memory, reports = envs.step(agent)
        completed_episodes += len(reports)
        total_steps += memory.step
        elapsed = time.time() - start_time
        print(
            f"episodes = {completed_episodes} in {total_steps} steps within {elapsed:.1f} seconds . rate = {total_steps // elapsed}")



def make_env(worker_id):
    print(f"Initializing worker {worker_id}, env={conf.env_name}")
    env = gym.make(conf.env_name)
    return env


if __name__ == "__main__":
    main("a3c_pathfinding_test")

