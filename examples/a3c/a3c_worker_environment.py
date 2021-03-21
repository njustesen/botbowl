import warnings
from itertools import chain

from multiprocessing import Process, Queue
import torch

from examples.a3c.a3c_agent import RL_Agent
from examples.a3c.a3c_reward import reward_function
from ffai import FFAIEnv

warnings.filterwarnings('ignore')

worker_max_steps = 400


class EndOfGameReport:
    def __init__(self, time_steps, reward, game):
        self.time_steps = time_steps
        self.own_td = game.state.home_team.state.score
        self.opp_td = game.state.away_team.state.score
        self.total_reward = reward
        self.episodes = 1
        self.win_rate = 1.0 if self.own_td>self.opp_td else 0.5 if self.own_td == self.opp_td else 0

    def merge(self, other):
        self.time_steps += other.time_steps
        self.own_td += other.own_td
        self.opp_td += other.opp_td
        self.total_reward += other.total_reward
        self.win_rate = (self.win_rate * self.episodes + other.win_rate * other.episodes) / (
                    self.episodes + other.episodes)
        self.episodes += other.episodes

    def __repr__(self):
        s = f"Episodes: {self.episodes}, " \
            f"Timesteps: {self.time_steps}, " \
            f"Win rate: {self.win_rate:.2f}, " \
            f"TD rate: {self.own_td/self.episodes:.2f}, " \
            f"TD rate opp: {self.opp_td/self.episodes:.2f}, " \
            f"Mean reward: {self.total_reward/self.episodes:.3f}, "
        return s


class Memory:
    def __init__(self, size, env):
        self.step = 0
        self.max_steps = size

        nbr_of_spat_layers = env.get_nbr_of_spat_layers()
        layer_width = env.get_layer_width()
        layer_height = env.get_layer_height()
        non_spatial_obs_shape = env.get_non_spatial_inputs()
        action_space = env.get_nbr_of_output_actions()

        self.spatial_obs = torch.zeros(size, nbr_of_spat_layers, layer_width, layer_height)
        self.non_spatial_obs = torch.zeros(size, non_spatial_obs_shape)
        self.rewards = torch.zeros(size, 1)
        self.returns = torch.zeros(size, 1)
        self.actions = torch.zeros(size, 1)
        self.actions = self.actions.long()
        self.action_masks = torch.zeros(size, action_space, dtype=torch.uint8)


class VectorMemory(Memory):
    def __init__(self, size, env):
        super().__init__(size, env)

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
        # TODO: consider clearing the variables
        self.step = 0
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

        return self.memory, result_report

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


def run_environment_to_done(env: FFAIEnv, agent: RL_Agent) -> (WorkerMemory, EndOfGameReport):
    with torch.no_grad():

        env.reset()
        memory = WorkerMemory(worker_max_steps, env)

        steps = 0
        total_reward = 0
        while True:

            (action_dict, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = agent.act_when_training(env)
            (_, _, done, info) = env.step(action_dict)
            reward_shaped = reward_function(env, info)

            memory.insert_network_step(spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)

            if done:
                memory.insert_epside_end()
                result_report = EndOfGameReport(time_steps=steps, game=env.game, reward=total_reward)

                return memory, result_report

            if steps >= 4000:
                # If we  get stuck or something - reset the environment
                print("Max. number of steps exceeded! Consider increasing the number.")
                env.reset()
                memory.reset()
                steps = 0
            steps += 1


def worker(results_queue, msg_queue, env, worker_id, trainee):
    while True:

        # Updates from master process?
        if not msg_queue.empty():
            command, data = msg_queue.get()
            if command == 'swap trainee':
                trainee = data
            elif command == 'close':
                break
            else:
                raise Exception(f"Unknown command to worker: {command}")

        memory, report = run_environment_to_done(env, trainee)

        results_queue.put((memory, report))

    msg_queue.cancel_join_thread()
    results_queue.cancel_join_thread()

    print(f"Worker {worker_id} - Quitting!")
