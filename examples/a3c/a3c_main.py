import gym

from examples.a3c.a3c_agent import CNNPolicy, A3CAgent
from examples.a3c.a3c_worker_environment import VectorEnvMultiProcess, VectorEnv, VectorMemory, EndOfGameReport
from ffai import FFAIEnv
from pytest import set_trace
from torch.autograd import Variable
import torch.optim as optim
from ffai.ai.layers import *
import torch
import torch.nn as nn
import time

import a3c_config as conf

# Make directories
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def update_agent_policy(ac_agent: CNNPolicy, optimizer, memory: VectorMemory):
    # ### Evaluate the actions taken ###
    spatial = Variable(memory.spatial_obs[:memory.step])
    non_spatial = Variable(memory.non_spatial_obs[:memory.step])
    non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

    actions = Variable(torch.LongTensor(memory.actions[:memory.step].view(-1, 1)))
    actions_mask = Variable(memory.action_masks[:memory.step])

    action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

    # ### Compute loss and back propagate ###
    advantages = Variable(memory.returns[:memory.step]) - values
    value_loss = advantages.pow(2).mean()

    action_loss = -(Variable(advantages.data) * action_log_probs).mean()

    optimizer.zero_grad()

    total_loss = (value_loss * conf.value_loss_coef + action_loss - dist_entropy * conf.entropy_coef)
    total_loss.backward()

    nn.utils.clip_grad_norm_(ac_agent.parameters(), conf.max_grad_norm)

    optimizer.step()
    return value_loss, action_loss

class PrintProgress:
    def __init__(self):

        self.report_to_print = None
        self.num_prints = 0

        # Setup column headers
        d = EndOfGameReport(0, 0)
        self.column_headers = ["updates, "]
        for k in d.get_dict_repr():
            self.column_headers.append(k + ", ")
        self.column_headers_lengths = [len(s) for s in self.column_headers]

    def update(self, report):
        if self.report_to_print is None:
            self.report_to_print = deepcopy(report)
        else:
            self.report_to_print.merge(report)

    def print(self, updates):
        s = ""

        if self.num_prints % 20 == 0:
            s += self.print_column_headers() + "\n"

        d = self.report_to_print.get_dict_repr()

        values = [f"{updates}"]
        for k in d:
            values.append(d[k])

        for header_len, value in zip(self.column_headers_lengths, values):
            extra_space = header_len - len(value)
            extra_space = extra_space if extra_space > 0 else 0
            if extra_space > 3:
                s += " " * (extra_space - 3) + value + " " * 3

        self.report_to_print = None
        self.num_prints += 1
        return s

    def print_column_headers(self):
        return "".join(self.column_headers)


def main(training_name):
    ensure_dir("logs/")
    log_dir = f"logs/{conf.env_name}/"
    ensure_dir(log_dir)
    #exp_id = str(uuid.uuid1())
    log_path = os.path.join(log_dir, f"{training_name}.p")

    # Clear log file
    try:
        os.remove(conf.log_filename)
    except OSError:
        pass

    es = [make_env(i) for i in range(conf.num_processes)]

    ac_agent = CNNPolicy(es[0], hidden_nodes=conf.num_hidden_nodes, kernels=conf.num_cnn_kernels)
    optimizer = optim.RMSprop(ac_agent.parameters(), conf.learning_rate)
    agent = A3CAgent("trainee", env_name=conf.env_name, policy=ac_agent)

    envs = VectorEnvMultiProcess([es[i] for i in range(conf.num_processes)], agent, 400)

    # Setup logging
    reports = []
    report_to_print = None
    total_steps = 0
    updates = 0
    total_episodes = 0
    time_last_report = time.time()
    seconds_between_saves = time.time()

    printer = PrintProgress()

    while updates < conf.max_updates:

        memory, report = envs.step(agent)
        value_loss, action_loss = update_agent_policy(ac_agent, optimizer, memory)

        #  Logging, saving and printing
        printer.update(report)
        elapsed_time = time.time() - time_last_report
        time_last_report = time.time()
        updates += 1

        report.elapsed_time = elapsed_time
        report.updates = 1
        report.value_loss = value_loss
        report.action_loss = action_loss

        reports.append(report)

        if time.time() - seconds_between_saves > 60:
            pickle.dump(reports, open(log_path, "wb"))
            seconds_between_saves = time.time()

        if updates % 5 == 0:
            print(printer.print(updates))

    pickle.dump(reports, open(log_path, "wb"))
    print("closing workers!")

    envs.close()
    print("main() quit!")


def make_env(worker_id):
    print(f"Initializing worker {worker_id}, pf={conf.pathfinding_enabled}")
    env = gym.make(conf.env_name)
    env.config.pathfinding_enabled = conf.pathfinding_enabled
    return env


if __name__ == "__main__":
    main("pathfinding_a3c")
    conf.pathfinding_enabled = False
    main("without_pf_a3c")
