import warnings
from dataclasses import dataclass

import numpy as np
from pytest import set_trace

import Curriculum as gc
from multiprocessing import Process, Pipe, Queue
from queue import Empty
import torch
import torch.nn as nn
import torch.nn.functional as F

from ffai.core.table import OutcomeType

from Reward import reward_function

warnings.filterwarnings('ignore')

worker_max_steps = 400

@dataclass
class EndOfGameReport:
    own_td: int = 0
    opp_td: int = 0
    total_reward: float = 0

    def add(self, other):
        raise NotImplementedError()


class Memory(object):
    def __init__(self, steps_per_update, env):
        spatial_obs_shape = env.get_spatial_obs_shape()
        non_spatial_obs_shape = env.get_non_spatial_obs_shape()
        action_space = env.get_action_shape()

        self.step = 0
        self.max_steps = steps_per_update

        self.spatial_obs = torch.zeros(steps_per_update, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, 1)
        self.returns = torch.zeros(steps_per_update, 1)
        self.td_outcome = torch.zeros(steps_per_update, 1)  # Todo, remove td outcome

        action_shape = 1
        self.actions = torch.zeros(steps_per_update, action_shape)
        self.actions = self.actions.long()
        self.action_masks = torch.zeros(steps_per_update, action_space, dtype=torch.uint8)

    def cuda(self):
        pass

    def clear_memory(self):
        pass  # TODO

    def insert_worker_memory(self, worker_mem):
        steps_to_copy = worker_mem.get_steps_to_copy()

        # check that there's space left.
        if self.max_steps - self.step < steps_to_copy:
            steps_to_copy = self.max_steps - self.step

        begin = self.step
        end = self.step + steps_to_copy

        self.spatial_obs[begin:end].copy_(worker_mem.spatial_obs[:steps_to_copy])
        self.non_spatial_obs[begin:end].copy_(worker_mem.non_spatial_obs[:steps_to_copy])
        self.rewards[begin:end].copy_(worker_mem.rewards[:steps_to_copy])
        self.returns[begin:end].copy_(worker_mem.returns[:steps_to_copy])
        self.td_outcome[begin:end].copy_(worker_mem.td_outcome[:steps_to_copy])

        self.actions[begin:end].copy_(worker_mem.actions[:steps_to_copy])
        self.action_masks[begin:end].copy_(worker_mem.action_masks[:steps_to_copy])

        self.step += steps_to_copy

    def not_full(self):
        return 0.9 * self.max_steps > self.step


class WorkerMemory(object):
    def __init__(self, max_steps, env):

        spatial_obs_shape = env.get_spatial_obs_shape()
        non_spatial_obs_shape = env.get_non_spatial_obs_shape()
        action_space = env.get_action_shape()

        self.max_steps = max_steps
        self.looped = None
        self.step = None
        self.reset()  # Sets step=0 and looped=False

        self.spatial_obs = torch.zeros(max_steps, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(max_steps, *non_spatial_obs_shape)
        self.rewards = torch.zeros(max_steps, 1)
        self.returns = torch.zeros(max_steps, 1)
        self.td_outcome = torch.zeros(max_steps, 1)

        self.actions = torch.zeros(max_steps, 1).long()  # action_shape = 1
        self.action_masks = torch.zeros(max_steps, action_space, dtype=torch.uint8)

    def cuda(self):
        pass

    def reset(self):
        # TODO: consider clearing the variables
        self.step = 0
        self.looped = False

    def insert_network_step(self, spatial_obs, non_spatial_obs, action, reward, action_masks):
        # The observation and the action, reward and action mask is inserted in the same step.
        # It means that this observation lead to this action. This is not the case in the tutorial from Njustesen!
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.action_masks[self.step].copy_(action_masks)
        self.spatial_obs[self.step].copy_(spatial_obs)
        self.non_spatial_obs[self.step].copy_(non_spatial_obs)

        self.step += 1
        if self.step == self.max_steps:
            self.step = 0
            self.looped = True

    def insert_epside_end(self):
        gamma = 0.99

        # Compute returns
        assert not (self.step == 0 and self.looped == False)
        if self.step != 0:

            self.returns[self.step - 1] = self.rewards[self.step - 1]
            for i in reversed(range(self.step - 1)):
                self.returns[i] = self.returns[i + 1] * gamma + self.rewards[i]

        if self.looped:
            self.returns[-1] = gamma * self.returns[0] + self.rewards[-1]
            for i in reversed(range(self.step + 1, self.max_steps - 1)):
                self.returns[i] = self.returns[i + 1] * gamma + self.rewards[i]

    def get_steps_to_copy(self):
        return self.max_steps if self.looped else self.step


class VecEnv():
    def __init__(self, envs, starting_agent, memory_size):
        """
        envs: list of FFAI environments to run in subprocesses
        """
        self.closed = False

        self.results_queue = Queue()
        self.ps_message = [Queue() for _ in range(len(envs))]

        self.ps = [Process(target=worker, args=(
            self.results_queue, msg_queue, env, envs.index(env), starting_agent))
                   for (msg_queue, env) in zip(self.ps_message, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()

        self.memory = Memory(memory_size, envs[0])

    def step(self):

        report = EndOfGameReport()

        while self.memory.not_full():
            data = self.results_queue.get()  # Blocking call
            self.memory.insert_worker_memory(data[0])

            report.add(data[1])

        return self.memory, report

    def update_trainee(self, agent):

        for ps_msg in self.ps_message:
            # Todo: check that queue is empty, otherwise process is very slow.
            ps_msg.put(('swap trainee', agent))

    def close(self):
        if self.closed:
            return

        for ps_msg in self.ps_message:
            ps_msg.put(('close', None))
            ps_msg.close()

        self.results_queue.close()

        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.ps)


def worker(results_queue, msg_queue, env, worker_id, trainee):
    worker_running = True
    steps = 0
    reset_steps = 2000

    with torch.no_grad():

        env.reset()
        memory = WorkerMemory(worker_max_steps, env)

        while worker_running:

            # Updates from master process?
            if not msg_queue.empty():
                command, data = msg_queue.get()
                if command == 'swap trainee':
                    trainee = data
                    # print(f"swap trainee on step - {steps}") # TODO, is quite a lot of steps...
                elif command == 'close':
                    break
                else:
                    raise Exception(f"Unknown command to worker: {command}")

            # Agent takes step  and insert to memory
            steps += 1

            # Todo, to big tuple to return
            (action, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = trainee.act_when_training(env)
            (_, _, done, info) = env.step(action)
            reward_shaped = reward_function(env, info)

            memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)

            # Check progress and report back
            if done:
                memory.insert_epside_end()
                result_report = {"score": 0}
                results_queue.put((memory, result_report))

                env.reset()
                memory.reset()
                steps = 0

            if steps >= reset_steps:
                # If we  get stuck or something - reset the environment
                print("Max. number of steps exceeded! Consider increasing the number.")
                env.reset()
                memory.reset()
                steps = 0

    msg_queue.cancel_join_thread()
    results_queue.cancel_join_thread()

    print(f"Worker {worker_id} - Quitting!")
