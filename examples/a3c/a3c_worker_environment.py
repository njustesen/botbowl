import warnings
from asyncio import sleep
from itertools import chain
import os
from typing import Optional

import psutil

import time

from multiprocessing import Process, Queue
import torch

from examples.a3c.a3c_agent import RL_Agent
from examples.a3c.a3c_reward import reward_function
from ffai import FFAIEnv

warnings.filterwarnings('ignore')

worker_max_steps = 200


def get_process_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


class EndOfGameReport:
    def __init__(self, time_steps, reward, game=None):
        self.time_steps = time_steps
        self.own_td = game.state.home_team.state.score if game is not None else 0
        self.opp_td = game.state.away_team.state.score if game is not None else 0
        self.total_reward = reward
        self.episodes = 1
        self.win_rate = 1.0 if self.own_td > self.opp_td else 0.5 if self.own_td == self.opp_td else 0
        self.memory_usage = get_process_memory_usage()

    def merge(self, other):
        self.time_steps += other.time_steps
        self.own_td += other.own_td
        self.opp_td += other.opp_td
        self.total_reward += other.total_reward
        self.win_rate = (self.win_rate * self.episodes + other.win_rate * other.episodes) / (
                self.episodes + other.episodes)

        self.memory_usage = (self.memory_usage * self.episodes + other.memory_usage * other.episodes) / (
                self.episodes + other.episodes)

        self.episodes += other.episodes

    def get_dict_repr(self):
        return {"Win rate": f"{self.win_rate:.2f}",
                "TD rate": f"{self.own_td / self.episodes:.2f}",
                "TD rate opp": f"{self.opp_td / self.episodes:.2f}",
                "Mean reward": f"{self.total_reward / self.episodes:.3f}"}

    def __repr__(self):
        s = ""
        for key, val in self.get_dict_repr().items():
            s += f"{key}: {val}, "

        return s[:-2]  # Remove trailing colon and blank


class Memory:
    def __init__(self, size, env):
        self.step = 0
        self.max_steps = size

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
        if self.max_steps - self.step < steps_to_copy:
            steps_to_copy = self.max_steps - self.step

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
        return 0.9 * self.max_steps > self.step


class WorkerMemory(Memory):
    def __init__(self, size, env: FFAIEnv):
        super().__init__(size, env)
        self.looped = False

    def reset(self):
        super().reset()
        self.looped = False

    def insert_network_step(self, spatial_obs, non_spatial_obs, action, reward, action_masks):
        # The observation and the action, reward and action mask is inserted in the same step.
        # It means that this observation lead to this action. This is not the case in the a2c tutorial!
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.action_masks[self.step].copy_(action_masks)
        self.spatial_obs[self.step].copy_(spatial_obs)
        self.non_spatial_obs[self.step].copy_(non_spatial_obs.squeeze())

        self.step += 1
        if self.step == self.max_steps:
            self.step = 0
            self.looped = True

    def insert_epside_end(self):
        gamma = 0.99

        # Compute returns
        if self.looped:
            order = chain(range(self.step, self.max_steps), range(self.step))
            order = reversed(list(order))
        else:
            order = reversed(range(self.step))

        previous_return = 0
        for i in order:
            self.returns[i] = self.rewards[i] + gamma * previous_return
            previous_return = self.returns[i]

    def get_steps_to_copy(self):
        return self.max_steps if self.looped else self.step


class AbstractVectorEnv:

    def step(self, agent) -> (VectorMemory, EndOfGameReport):
        raise NotImplementedError("To be over written by subclass")


class VectorEnv(AbstractVectorEnv):
    """
    This class is here for debugging purposes. It's easier to analyze the trace of just one process.
    """

    def __init__(self, envs, starting_agent, memory_size):
        self.envs = envs
        self.memory = VectorMemory(memory_size, envs[0])

    def step(self, agent: RL_Agent) -> (VectorMemory, EndOfGameReport):
        result_report = None
        for env in self.envs:
            worker_mem, report = run_environment_to_done(env, agent)
            self.memory.insert_worker_memory(worker_mem)

            if result_report is None:
                result_report = report
            else:
                result_report.merge(report)

        return self.memory, result_report


class VectorEnvMultiProcess(AbstractVectorEnv):
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

        self.memory = VectorMemory(memory_size, envs[0])

    def step(self, agent) -> (VectorMemory, EndOfGameReport):
        self.memory.reset()
        self.update_trainee(agent)

        result_report = None
        while self.memory.not_full():
            data = self.results_queue.get()  # Blocking call
            worker_mem = data[0]
            report = data[1]
            self.memory.insert_worker_memory(worker_mem)

            if result_report is None:
                result_report = report
            else:
                result_report.merge(report)

        self._send_to_children('pause')

        return self.memory, result_report

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


def run_environment_to_done(env: FFAIEnv, agent: RL_Agent, memory: Optional[WorkerMemory]) -> (
WorkerMemory, EndOfGameReport):
    with torch.no_grad():

        env.reset()

        if memory is None:
            memory = WorkerMemory(worker_max_steps, env)
        else:
            memory.reset()

        steps = 0
        total_reward = 0
        done = False
        while not done:

            (action_dict, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = agent.act_when_training(env)
            (_, _, done, info) = env.step(action_dict, skip_obs=True)
            reward_shaped = reward_function(env, info)
            total_reward += reward_shaped
            memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)
            steps += 1

            if steps >= 4000:
                # If we  get stuck or something - reset the environment
                print("Max. number of steps exceeded! Consider increasing the number.")
                env.reset()
                memory.reset()
                steps = 0


        memory.insert_epside_end()
        result_report = EndOfGameReport(time_steps=steps, game=env.game, reward=total_reward)

        return memory, result_report


def worker(results_queue, msg_queue, env, worker_id, trainee):
    pause = True
    time_of_last_message = time.time()
    memory = WorkerMemory(worker_max_steps, env)

    while True:

        # Updates from master process?
        if not msg_queue.empty():
            command, data = msg_queue.get()
            time_of_last_message = time.time()

            if command == 'swap trainee':
                trainee = data
                pause = False
            elif command == 'close':
                break
            elif command == 'pause':
                pause = True
            else:
                raise Exception(f"Unknown command to worker: {command}")

        if pause:
            sleep(0.005)
        else:
            memory, report = run_environment_to_done(env, trainee, memory)
            results_queue.put((memory, report))
            del report

        if time.time() - time_of_last_message > 10:
            print(f"Worker {worker_id} timout!")
            break

    msg_queue.cancel_join_thread()
    results_queue.cancel_join_thread()

    print(f"Worker {worker_id} - Quitting!")
