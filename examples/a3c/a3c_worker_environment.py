import warnings
from asyncio import sleep
from itertools import chain

from typing import Optional

import time

from multiprocessing import Process, Queue
import torch

from examples.a3c import a3c_config
from examples.a3c.a3c_agent import RL_Agent
from examples.a3c.a3c_reward import reward_function
from examples.a3c.reports import EndOfGameReport
from ffai import FFAIEnv

warnings.filterwarnings('ignore')


class Memory:
    def __init__(self, size, env):
        self.step = 0
        self.size = size

        self.nbr_of_spat_layers = env.get_nbr_of_spat_layers()
        self.layer_width = env.get_layer_width()
        self.layer_height = env.get_layer_height()
        self.non_spatial_obs_shape = env.get_non_spatial_inputs()
        self.action_space = env.get_nbr_of_output_actions()

        self.spatial_obs = torch.zeros(size, self.nbr_of_spat_layers, self.layer_width, self.layer_height)
        self.non_spatial_obs = torch.zeros(size, self.non_spatial_obs_shape)
        self.rewards = torch.zeros(size, 1)
        self.returns = torch.zeros(size, 1)
        self.actions = torch.zeros(size, 1)
        self.actions = self.actions.long()
        self.action_masks = torch.zeros(size, self.action_space, dtype=torch.uint8)

    def reset(self):
        self.step = 0


class VectorMemory(Memory):
    def __init__(self, size, env):
        super().__init__(size, env)

    def insert_worker_memory(self, worker_mem):
        steps_to_copy = worker_mem.get_steps_to_copy()

        # check that there's space left.
        if self.size - self.step < steps_to_copy:
            steps_to_copy = self.size - self.step

        begin = self.step
        end = self.step + steps_to_copy

        self.spatial_obs[begin:end].copy_(worker_mem.spatial_obs[:steps_to_copy])
        self.non_spatial_obs[begin:end].copy_(worker_mem.non_spatial_obs[:steps_to_copy])
        self.rewards[begin:end].copy_(worker_mem.rewards[:steps_to_copy])
        self.returns[begin:end].copy_(worker_mem.returns[:steps_to_copy])

        self.actions[begin:end].copy_(worker_mem.actions[:steps_to_copy])
        self.action_masks[begin:end].copy_(worker_mem.action_masks[:steps_to_copy])

        self.step += steps_to_copy

    def not_full(self):
        return 0.9 * self.size > self.step


class WorkerMemory(Memory):
    def __init__(self, size, env: FFAIEnv):
        super().__init__(size, env)
        self.done_indices = set()

    def reset(self):
        super().reset()
        self.done_indices.clear()

    def insert_network_step(self, spatial_obs, non_spatial_obs, action, reward, action_masks):
        # The observation and the action, reward and action mask is inserted in the same step.
        # It means that this observation lead to this action. This is not the case in the a2c tutorial!
        assert self.step < self.size
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.action_masks[self.step].copy_(action_masks)
        self.spatial_obs[self.step].copy_(spatial_obs)
        self.non_spatial_obs[self.step].copy_(non_spatial_obs.squeeze())

        self.step += 1

    def calculate_returns(self, estimated_future_reward=0, gamma=0.99):
        previous_return = estimated_future_reward
        for i in reversed(range(self.step)):
            self.returns[i] = self.rewards[i] + gamma * previous_return * (i not in self.done_indices)
            previous_return = self.returns[i]

    def set_done(self):
        self.done_indices.add(self.step - 1)

    def get_steps_to_copy(self):
        return self.step


class AbstractVectorEnv:

    def step(self, agent) -> (VectorMemory, EndOfGameReport):
        raise NotImplementedError("To be over written by subclass")


class VectorEnv(AbstractVectorEnv):
    """
    This class is here for debugging purposes. It's easier to analyze the trace of just one process.
    """

    def __init__(self, envs, min_batch_size, worker_memory_size):
        self.memory = VectorMemory(min_batch_size + worker_memory_size, envs[0])
        worker_memories = [WorkerMemory(worker_memory_size, envs[0]) for _ in envs]

        self.runners = [Runner(env, None, w_mem) for env, w_mem in zip(envs, worker_memories)]

        for runner in self.runners:
            runner.reset()

    def step(self, agent: RL_Agent) -> (VectorMemory, [EndOfGameReport]):
        result_reports = []

        for runner in self.runners:
            runner.agent = agent
            worker_mem, reports = runner.run()
            self.memory.insert_worker_memory(worker_mem)
            result_reports.extend(reports)

        return self.memory, result_reports


class VectorEnvMultiProcess(AbstractVectorEnv):
    def __init__(self, envs, min_batch_size, worker_memory_size):
        """
        envs: list of FFAI environments to run in subprocesses
        """
        self.closed = False

        self.results_queue = Queue()
        self.ps_message = [Queue() for _ in range(len(envs))]

        self.ps = [Process(target=worker, args=(
            self.results_queue, msg_queue, env, envs.index(env), worker_memory_size))
                   for (msg_queue, env) in zip(self.ps_message, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()

        self.memory = VectorMemory(min_batch_size + worker_memory_size, envs[0])

        self.min_batch_size = min_batch_size
        self.worker_memory_size = worker_memory_size

    def step(self, agent) -> (VectorMemory, [EndOfGameReport]):

        self.memory.reset()

        # print(f"queue size = {self.results_queue.qsize()}")
        self.update_trainee(agent)

        reports = []
        while self.memory.step < self.min_batch_size:
            data = self.results_queue.get()  # Blocking call

            worker_mem = data[0]
            received_reports = data[1]
            self.memory.insert_worker_memory(worker_mem)

            reports.extend(received_reports)

        assert self.results_queue.qsize() == 0

        return self.memory, reports

    def update_trainee(self, agent):
        self._send_to_children('swap trainee', agent)

    def _send_to_children(self, command, data=None):
        for ps_msg in self.ps_message:
            ps_msg.put((command, data))

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

    def __del__(self):
        self.close()


class Runner:
    def __init__(self, env, agent, memory: WorkerMemory):
        self.env = env
        self.agent = agent
        self.memory = memory

        self.episode_steps = 0
        self.total_reward = 0

    def run(self) -> (WorkerMemory, [EndOfGameReport]):

        self.memory.reset()

        end_at_step_count = self.episode_steps + self.memory.size
        reports = []

        while self.episode_steps < end_at_step_count:

            (action_dict, action_idx, action_masks, _, spatial_obs, non_spatial_obs) = \
                self.agent.act_when_training(self.env)

            (_, _, done, info) = self.env.step(action_dict, skip_obs=True)

            reward = reward_function(self.env, info)
            self.total_reward += reward
            self.memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward, action_masks)
            self.episode_steps += 1

            if done:
                self.memory.set_done()
                reports.append(EndOfGameReport(time_steps=self.episode_steps, game=self.env.game,
                                               reward=self.total_reward))
                self.reset()
                end_at_step_count = self.memory.size - self.memory.step

        if self.episode_steps >= 4000:
            print("Max. number of steps exceeded! Consider increasing the number.")
            pass  # Todo

        if self.episode_steps == 0:
            value = 0
        else:
            (_, _, _, value, _, _) = self.agent.act_when_training(self.env)

        self.memory.calculate_returns(value)

        return self.memory, reports

    def reset(self):
        self.episode_steps = 0
        self.total_reward = 0
        self.env.reset(skip_obs=True)


def worker(results_queue, msg_queue, env, worker_id, worker_memory_size):
    pause = True
    time_of_last_message = time.time()
    memory = WorkerMemory(worker_memory_size, env)
    runner = Runner(env, agent=None, memory=memory)

    runner.reset()

    while True:

        # Updates from master process?
        if not msg_queue.empty():
            command, data = msg_queue.get()
            time_of_last_message = time.time()

            if command == 'swap trainee':
                runner.agent = data
                pause = False
            elif command == 'close':
                break
            elif command == 'pause':
                pause = True
            else:
                raise Exception(f"Unknown command to worker: {command}")

        if not pause:
            memory, reports = runner.run()
            results_queue.put((memory, reports))
            pause = True

        else:
            sleep(0.005)
            if time.time() - time_of_last_message > 10:
                print(f"Worker {worker_id} timeout!")
                break

    msg_queue.cancel_join_thread()
    results_queue.cancel_join_thread()

    print(f"Worker {worker_id} - Quitting!")
