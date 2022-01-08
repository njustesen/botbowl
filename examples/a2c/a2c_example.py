import os
from operator import itemgetter
from typing import Tuple, Iterable

import gym
from botbowl import BotBowlEnv
from torch.autograd import Variable
import torch.optim as optim
from multiprocessing import Process, Pipe
from botbowl.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from .a2c_agent import A2CAgent
from .a2c_env import make_env
import botbowl
import random
import uuid


# Training configuration
num_steps = 1000000
num_processes = 8
steps_per_update = 20
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
log_interval = 50
save_interval = 10
ppcg = False


reset_steps = 5000  # The environment is reset after this many steps it gets stuck

# Self-play
selfplay = False  # Use this to enable/disable self-play
selfplay_window = 1
selfplay_save_steps = int(num_steps / 10)
selfplay_swap_steps = selfplay_save_steps

# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]

# When using A2CAgent, remember to set exclude_pathfinding_moves = False if you train with pathfinding_enabled = True


# Make directories
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir("logs/")
ensure_dir("models/")
ensure_dir("plots/")
exp_id = str(uuid.uuid1())
log_dir = f"logs/{env_name}/"
model_dir = f"models/{env_name}/"
plot_dir = f"plots/{env_name}/"
ensure_dir(log_dir)
ensure_dir(model_dir)
ensure_dir(plot_dir)


class Memory(object):
    def __init__(self, steps_per_update, num_processes, spatial_obs_shape, non_spatial_obs_shape, action_space):
        self.spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, num_processes, 1)
        self.value_predictions = torch.zeros(steps_per_update + 1, num_processes, 1)
        self.returns = torch.zeros(steps_per_update + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(steps_per_update + 1, num_processes, action_space, dtype=torch.uint8)

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.value_predictions = self.value_predictions.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.action_masks = self.action_masks.cuda()

    def insert(self, step, spatial_obs, non_spatial_obs, action, value_pred, reward, mask, action_masks):
        self.spatial_obs[step + 1].copy_(spatial_obs)
        self.non_spatial_obs[step + 1].copy_(non_spatial_obs)
        self.actions[step].copy_(action)
        self.value_predictions[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.action_masks[step].copy_(action_masks)

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]


class CNNPolicy(nn.Module):
    def __init__(self, spatial_shape, non_spatial_inputs, hidden_nodes, kernels, actions):
        super(CNNPolicy, self).__init__()

        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_shape[0], out_channels=kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding=1)

        # Non-spatial input stream
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # Linear layers
        stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        stream_size += hidden_nodes
        self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(hidden_nodes, actions)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.actor.weight.data.mul_(relu_gain)
        self.critic.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through two convolutional layers
        x1 = self.conv1(spatial_input)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)

        # Concatenate the input streams
        flatten_x1 = x1.flatten(start_dim=1)

        x2 = self.linear0(non_spatial_input)
        x2 = F.relu(x2)

        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)

        # Fully-connected layers
        x3 = self.linear1(concatenated)
        x3 = F.relu(x3)
        #x2 = self.linear2(x2)
        #x2 = F.relu(x2)

        # Output streams
        value = self.critic(x3)
        actor = self.actor(x3)

        # return value, policy
        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs


def worker(remote, parent_remote, env, worker_id):
    parent_remote.close()

    steps = 0
    tds = 0
    tds_opp = 0
    next_opp = botbowl.make_bot('random')

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action, dif = data[0], data[1]
            spatial_obs, reward, done, info = env.step(action)
            non_spatial_obs = info['non_spatial_obs']
            action_mask = info['action_mask']

            game: Game = env.game

            tds_scored = game.state.home_team.state.score - tds
            tds_opp_scored = game.state.away_team.state.score - tds_opp
            tds = game.state.home_team.state.score
            tds_opp = game.state.away_team.state.score

            # PPCG
            if dif < 1.0:
                ball_carrier = game.get_ball_carrier()
                if ball_carrier and ball_carrier.team == env.game.state.home_team:
                    extra_endzone_squares = int((1.0 - dif) * 25.0)
                    distance_to_endzone = ball_carrier.position.x - 1
                    if distance_to_endzone <= extra_endzone_squares:
                        game.state.stack.push(Touchdown(env.game, ball_carrier))

            if done or steps >= reset_steps:
                # If we get stuck or something - reset the environment
                if steps >= reset_steps:
                    print("Max. number of steps exceeded! Consider increasing the number.")
                    done = True
                #env.opp_actor = next_opp
                env.reset()
                spatial_obs, non_spatial_obs, action_mask = env.get_state()
                steps = 0
                tds = 0
                tds_opp = 0
            remote.send((spatial_obs, non_spatial_obs, action_mask, reward, tds_scored, tds_opp_scored, done))

        elif command == 'reset':
            steps = 0
            tds = 0
            tds_opp = 0
            env.opp_actor = next_opp
            env.reset()
            spatial_obs, non_spatial_obs, action_mask = env.get_state()
            remote.send((spatial_obs, non_spatial_obs, action_mask, 0.0, 0, 0, False))

        elif command == 'swap':
            next_opp = data
        elif command == 'close':
            break


class VecEnv:
    def __init__(self, envs):
        """
        envs: list of botbowl environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions: Iterable[int], difficulty=1.0) -> Tuple[np.ndarray, ...]:
        """
        Takes one step in each environment, returns the results as stacked numpy arrays
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action, difficulty]))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def reset(self, difficulty=1.0):
        for remote in self.remotes:
            remote.send(('reset', difficulty))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def swap(self, agent):
        for remote in self.remotes:
            remote.send(('swap', agent))

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)


def main():
    assert not selfplay, "Self play is not implemented yet"

    env = make_env()
    env.reset()
    obs, _, _, info = env.get_state()
    spatial_obs_space = obs.shape
    non_spatial_obs_space = info['non_spatial_obs'].shape
    action_space = len(info['action_mask'])

    # MODEL
    ac_agent = CNNPolicy(spatial_obs_space,
                         non_spatial_obs_space,
                         hidden_nodes=num_hidden_nodes,
                         kernels=num_cnn_kernels,
                         actions=action_space)

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

    # MEMORY STORE
    memory = Memory(steps_per_update, num_processes, spatial_obs_space, (1, non_spatial_obs_space), action_space)

    # PPCG
    difficulty = 0.0 if ppcg else 1.0
    dif_delta = 0.01

    # Reset environments
    envs = VecEnv([make_env() for _ in range(num_processes)])
    spatial_obs, non_spatial_obs, action_masks, reward, tds_scored, tds_opp_scored, done = envs.reset(difficulty)

    # Add obs to memory
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)

    # Variables for storing stats
    all_updates = 0
    all_episodes = 0
    all_steps = 0
    episodes = 0
    proc_rewards = np.zeros(num_processes)
    proc_tds = np.zeros(num_processes)
    proc_tds_opp = np.zeros(num_processes)
    episode_rewards = []
    episode_tds = []
    episode_tds_opp = []
    wins = []
    value_losses = []
    policy_losses = []
    log_updates = []
    log_episode = []
    log_steps = []
    log_win_rate = []
    log_td_rate = []
    log_td_rate_opp = []
    log_mean_reward = []
    log_difficulty = []

    # self-play
    selfplay_next_save = selfplay_save_steps
    selfplay_next_swap = selfplay_swap_steps
    selfplay_models = 0
    if selfplay:
        model_name = f"{exp_id}_selfplay_0.nn"
        model_path = os.path.join(model_dir, model_name)
        torch.save(ac_agent, model_path)
        envs.swap(A2CAgent(name=model_name, env_name=env_name, filename=model_path))
        selfplay_models += 1

    while all_steps < num_steps:

        for step in range(steps_per_update):

            action_masks = torch.tensor(action_masks, dtype=torch.bool)

            values, actions = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]),
                Variable(action_masks))

            action_objects = map(itemgetter(0), actions)

            spatial_obs, non_spatial_obs, action_masks, shaped_reward, tds_scored, tds_opp_scored, done = envs.step(action_objects, difficulty=difficulty)

            proc_rewards += shaped_reward
            proc_tds += tds_scored
            proc_tds_opp += tds_opp_scored
            episodes += done.sum()

            # If done then clean the history of observations.
            for i in range(num_processes):
                if done[i]:
                    if proc_tds[i] > proc_tds_opp[i]:  # Win
                        wins.append(1)
                        difficulty += dif_delta
                    elif proc_tds[i] < proc_tds_opp[i]:  # Loss
                        wins.append(0)
                        difficulty -= dif_delta
                    else:  # Draw
                        wins.append(0.5)
                        difficulty -= dif_delta
                    if ppcg:
                        difficulty = min(1.0, max(0, difficulty))
                    else:
                        difficulty = 1
                    episode_rewards.append(proc_rewards[i])
                    episode_tds.append(proc_tds[i])
                    episode_tds_opp.append(proc_tds_opp[i])
                    proc_rewards[i] = 0
                    proc_tds[i] = 0
                    proc_tds_opp[i] = 0

            # insert the step taken into memory
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            shaped_reward = torch.from_numpy(np.expand_dims(np.stack(shaped_reward), 1)).float()

            memory.insert(step, spatial_obs, non_spatial_obs,
                          actions.data, values.data, shaped_reward, masks, action_masks)

        next_value = ac_agent(Variable(memory.spatial_obs[-1], requires_grad=False), Variable(memory.non_spatial_obs[-1], requires_grad=False))[0].data

        # Compute returns
        memory.compute_returns(next_value, gamma)

        spatial = Variable(memory.spatial_obs[:-1])
        spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs[:-1])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.action_masks[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        values = values.view(steps_per_update, num_processes, 1)
        action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()
        #value_losses.append(value_loss)

        # Compute loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        #policy_losses.append(action_loss)

        optimizer.zero_grad()

        total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

        # Updates
        all_updates += 1
        # Episodes
        all_episodes += episodes
        episodes = 0
        # Steps
        all_steps += num_processes * steps_per_update

        # Self-play save
        if selfplay and all_steps >= selfplay_next_save:
            selfplay_next_save = max(all_steps+1, selfplay_next_save+selfplay_save_steps)
            model_name = f"{exp_id}_selfplay_{selfplay_models}.nn"
            model_path = os.path.join(model_dir, model_name)
            print(f"Saving {model_path}")
            torch.save(ac_agent, model_path)
            selfplay_models += 1

        # Self-play swap
        if selfplay and all_steps >= selfplay_next_swap:
            selfplay_next_swap = max(all_steps + 1, selfplay_next_swap+selfplay_swap_steps)
            lower = max(0, selfplay_models-1-(selfplay_window-1))
            i = random.randint(lower, selfplay_models-1)
            model_name = f"{exp_id}_selfplay_{i}.nn"
            model_path = os.path.join(model_dir, model_name)
            print(f"Swapping opponent to {model_path}")
            envs.swap(A2CAgent(name=model_name, env_name=env_name, filename=model_path))

        # Logging
        if all_updates % log_interval == 0 and len(episode_rewards) >= num_processes:
            td_rate = np.mean(episode_tds)
            td_rate_opp = np.mean(episode_tds_opp)
            episode_tds.clear()
            episode_tds_opp.clear()
            mean_reward = np.mean(episode_rewards)
            episode_rewards.clear()
            win_rate = np.mean(wins)
            wins.clear()
            #mean_value_loss = np.mean(value_losses)
            #mean_policy_loss = np.mean(policy_losses)    
            
            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_td_rate_opp.append(td_rate_opp)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)

            log = "Updates: {}, Episodes: {}, Timesteps: {}, Win rate: {:.2f}, TD rate: {:.2f}, TD rate opp: {:.2f}, Mean reward: {:.3f}, Difficulty: {:.2f}" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            # Save to files
            log_path = os.path.join(log_dir, f"{exp_id}.dat")
            print(f"Save log to {log_path}")
            with open(log_path, "a") as myfile:
                myfile.write(log_to_file)

            print(log)

            episodes = 0
            value_losses.clear()
            policy_losses.clear()

            # Save model
            model_name = f"{exp_id}.nn"
            model_path = os.path.join(model_dir, model_name)
            torch.save(ac_agent, model_path)
            
            # plot
            n = 3
            if ppcg:
                n += 1
            fig, axs = plt.subplots(1, n, figsize=(4*n, 5))
            axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[0].plot(log_steps, log_mean_reward)
            axs[0].set_title('Reward')
            #axs[0].set_ylim(bottom=0.0)
            axs[0].set_xlim(left=0)
            axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[1].plot(log_steps, log_td_rate, label="Learner")
            axs[1].set_title('TD/Episode')
            axs[1].set_ylim(bottom=0.0)
            axs[1].set_xlim(left=0)
            if selfplay:
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[1].plot(log_steps, log_td_rate_opp, color="red", label="Opponent")
            axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[2].plot(log_steps, log_win_rate)
            axs[2].set_title('Win rate')            
            axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
            axs[2].set_xlim(left=0)
            if ppcg:
                axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[3].plot(log_steps, log_difficulty)
                axs[3].set_title('Difficulty')
                axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
                axs[3].set_xlim(left=0)
            fig.tight_layout()
            plot_name = f"{exp_id}_{'_selfplay' if selfplay else ''}.png"
            plot_path = os.path.join(plot_dir, plot_name)
            fig.savefig(plot_path)
            plt.close('all')

    model_name = f"{exp_id}.nn"
    model_path = os.path.join(model_dir, model_name)
    torch.save(ac_agent, model_path)
    envs.close()


if __name__ == "__main__":
    main()
