import gym
from ffai import FFAIEnv
from torch.autograd import Variable
import torch.optim as optim
from multiprocessing import Process, Pipe
from ffai.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


# Training configuration
num_steps = 10000000
num_processes = 8
num_updates = 100000
steps_per_update = 10
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
log_interval = 50
save_interval = 500

# Environment
env_name = "FFAI-1-v1"
reset_steps = 200  # The environment is reset after this many steps in case of bugs

# Architecture
hidden_nodes = 25

model_name = env_name
log_filename = "logs/" + model_name + ".dat"

# Reward function
rewards_own = {
    OutcomeType.TOUCHDOWN: 1,
    OutcomeType.END_OF_GAME_WINNER: 1,
    OutcomeType.BALL_CAUGHT: 0.1,
    OutcomeType.INTERCEPTION: 0.2,
    OutcomeType.SUCCESSFUL_PICKUP: 0.1,
    OutcomeType.TURNOVER: -0.05
}
rewards_opp = {
    OutcomeType.TOUCHDOWN: 0.0,
    OutcomeType.END_OF_GAME_WINNER: -1,
    OutcomeType.BALL_CAUGHT: -0.1,
    OutcomeType.INTERCEPTION: -0.2,
    OutcomeType.KNOCKED_DOWN: 0.05,
    OutcomeType.KNOCKED_OUT: 0.05,
    OutcomeType.CASUALTY: 0.1,
    OutcomeType.BALL_DROPPED: 0.2,
    OutcomeType.TURNOVER: 0.05
}


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
        # self.positions[step].copy_(position)
        self.value_predictions[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.action_masks[step].copy_(action_masks)

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]


class CNNPolicy(nn.Module):
    def __init__(self, spatial_shape, non_spatial_inputs, hidden_nodes, actions):
        super(CNNPolicy, self).__init__()

        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Non-spatial input stream
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # Linear layers
        stream_size = 32 * spatial_shape[1] * spatial_shape[2]
        stream_size += hidden_nodes
        #self.linear1 = nn.Linear(stream_size, hidden_nodes)
        #self.linear2 = nn.Linear(hidden_nodes, hidden_nodes)

        # The outputs
        self.critic = nn.Linear(stream_size, 1)
        self.actor = nn.Linear(stream_size, actions)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        #self.linear1.weight.data.mul_(relu_gain)
        self.linear0.weight.data.mul_(relu_gain)
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
        #x2 = self.linear1(concatenated)
        #x2 = F.relu(x2)
        #x2 = self.linear2(x2)
        #x2 = F.relu(x2)

        # Output streams
        value = self.critic(concatenated)
        actor = self.actor(concatenated)

        # return value, policy
        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        # print(action_probs.tolist())
        actions = action_probs.multinomial(1)
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        # gets values, actions, and positions
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

    def reward_function(env, shaped=False):
        r = 0
        for outcome in env.get_outcomes():
            if not shaped and outcome.outcome_type != OutcomeType.TOUCHDOWN:
                continue
            team = None
            if outcome.player is not None:
                team = outcome.player.team
            elif outcome.team is not None:
                team = outcome.team
            if team == env.own_team and outcome.outcome_type in rewards_own:
                r += rewards_own[outcome.outcome_type]
            if team == env.opp_team and outcome.outcome_type in rewards_opp:
                r += rewards_opp[outcome.outcome_type]
        return r

    steps = 0

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action = data
            obs, reward, done, info = env.step(action)
            reward, reward_shaped = reward_function(env), reward_function(env, shaped=True)
            if done:
                obs = env.reset()
                steps = 0
            elif steps >= reset_steps:
                obs, reward, done, info = env.step(action)
                obs = env.reset()
                steps = 0
            remote.send((obs, reward, reward_shaped, done, info))
        elif command == 'reset':
            steps = 0
            obs = env.reset()
            remote.send(obs)
        elif command == 'render':
            env.render()
        elif command == 'close':
            break


class VecEnv():
    def __init__(self, envs):
        """
        envs: list of blood bowl game environments to run in subprocesses
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

    def step(self, actions):
        cumul_rewards = None
        cumul_shaped_rewards = None
        cumul_dones = None
        #print("Sending step")
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        #print("Returned step")

        obs, rews, rews_shaped, dones, infos = zip(*results)
        if cumul_rewards is None:
            cumul_rewards = np.stack(rews)
            cumul_shaped_rewards = np.stack(rews_shaped)
        else:
            cumul_rewards += np.stack(rews)
            cumul_shaped_rewards += np.stack(rews_shaped)
        if cumul_dones is None:
            cumul_dones = np.stack(dones)
        else:
            cumul_dones |= np.stack(dones)
        return np.stack(obs), cumul_rewards, cumul_shaped_rewards, cumul_dones, infos

    def reset(self):
        #print("Sending reset")
        for remote in self.remotes:
            remote.send(('reset', None))
        #print("Reset returned")
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        return

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
    es = [make_env(i) for i in range(num_processes)]
    envs = VecEnv([es[i] for i in range(num_processes)])

    spatial_obs_space = es[0].observation_space.spaces['board'].shape
    board_dim = (spatial_obs_space[1], spatial_obs_space[2])
    board_squares = spatial_obs_space[1] * spatial_obs_space[2]

    non_spatial_obs_space = es[0].observation_space.spaces['state'].shape[0] + es[0].observation_space.spaces['procedures'].shape[0] + es[0].observation_space.spaces['available-action-types'].shape[0]
    non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
    num_non_spatial_action_types = len(non_spatial_action_types)
    spatial_action_types = FFAIEnv.positional_action_types
    num_spatial_action_types = len(spatial_action_types)
    num_spatial_actions = num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
    action_space = num_non_spatial_action_types + num_spatial_actions

    def compute_action_masks(observations):
        masks = []
        m = False
        for ob in observations:
            mask = np.zeros(action_space)
            i = 0
            for action_type in non_spatial_action_types:
                mask[i] = ob['available-action-types'][action_type.name]
                i += 1
            for action_type in spatial_action_types:
                if ob['available-action-types'][action_type.name] == 0:
                    mask[i:i+board_squares] = 0
                elif ob['available-action-types'][action_type.name] == 1:
                    position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
                    position_mask_flatten = np.reshape(position_mask, (1, board_squares))
                    '''
                    if action_type == ActionType.MOVE:
                        print("-------------")
                        print(len(masks))
                        print(position_mask)
                        print(position_mask_flatten)
                        m = True
                    '''
                    for j in range(board_squares):
                        mask[i + j] = position_mask_flatten[0][j]
                i += board_squares
            assert 1 in mask
            if m:
                print(mask)
            masks.append(mask)
        return masks

    def compute_action(action_idx):
        if action_idx < len(non_spatial_action_types):
            return non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - num_non_spatial_action_types
        spatial_pos_idx = spatial_idx % board_squares
        spatial_y = int(spatial_pos_idx / board_dim[1])
        spatial_x = int(spatial_pos_idx % board_dim[1])
        spatial_action_type_idx = int(spatial_idx / board_squares)
        spatial_action_type = spatial_action_types[spatial_action_type_idx]
        return spatial_action_type, spatial_x, spatial_y

    # MODEL
    ac_agent = CNNPolicy(spatial_obs_space, non_spatial_obs_space, hidden_nodes=hidden_nodes, actions=action_space)

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

    # MEMORY STORE
    memory = Memory(steps_per_update, num_processes, spatial_obs_space, (1, non_spatial_obs_space), action_space)

    # Reset environments
    obs = envs.reset()
    spatial_obs, non_spatial_obs = update_obs(obs)

    # Add obs to memory
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)

    all_updates = 0
    all_episodes = 0
    all_steps = 0
    #renderer = Renderer()
    rewards = 0
    rewards_shaped = 0
    episodes = 0
    proc_rewards = np.zeros(num_processes)
    proc_shaped_rewards = np.zeros(num_processes)
    episode_rewards = []
    episode_shaped_rewards = []
    value_losses = []
    policy_losses = []

    for update in range(num_updates):

        for step in range(steps_per_update):

            action_masks = compute_action_masks(obs)
            action_masks = torch.tensor(action_masks, dtype=torch.bool)

            values, actions = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]),
                Variable(action_masks))

            action_objects = []

            for action in actions:
                action_type, x, y = compute_action(action.numpy()[0])
                action_object = {
                    'action-type': action_type,
                    'x': x,
                    'y': y
                }
                '''
                if action_type == ActionType.MOVE:
                    print(action_type, x, y)
                '''
                action_objects.append(action_object)

            obs, reward, shaped_reward, done, info = envs.step(action_objects)

            '''
            if render:
                for i in range(num_processes):
                    renderer.render(obs[i], i)
            '''
            #shaped_reward = torch.from_numpy(np.expand_dims(np.stack(shaped_reward), 1)).float()
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            rewards += reward.sum().item()
            #rewards_shaped += shaped_reward.sum().item()
            r = reward.numpy()[0]
            #rs = shaped_reward.numpy()[0]
            for i in range(len(r)):
                proc_rewards[i] += r[i]
            #    proc_shaped_rewards[i] += rs[i]

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            dones = masks.squeeze()
            episodes += num_processes - int(dones.sum().item())
            for i in range(len(done)):
                if done[i]:
                    episode_rewards.append(proc_rewards[i])
            #        episode_shaped_rewards.append(proc_shaped_rewards[i])
                    proc_rewards[i] = 0
            #        proc_shaped_rewards[i] = 0

            # Update the observations returned by the environment
            spatial_obs, non_spatial_obs = update_obs(obs)

            # insert the step taken into memory
            #memory.insert(step, torch.from_numpy(spatial_obs).float(), torch.from_numpy(non_spatial_obs).float(),
            #              torch.tensor(actions), torch.tensor(values), reward, masks)
            memory.insert(step, spatial_obs, non_spatial_obs,
                          actions.data, values.data, reward, masks, action_masks)

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

        # Logging
        if all_updates % log_interval == 0 and len(episode_rewards) >= num_processes:

            mean_reward_pr_episode = np.mean(episode_rewards)
            #mean_shaped_reward_pr_episode = np.mean(episode_shaped_rewards)
            episode_rewards.clear()
            #episode_shaped_rewards.clear()
            #mean_value_loss = np.mean(value_losses)
            #mean_policy_loss = np.mean(policy_losses)

            log = "Updates: {}, Episodes: {}, Timesteps: {}, Reward/Episode: {:.2f}, TD/Episode: {:.2f}" \
                .format(all_updates, all_episodes, all_steps, mean_reward_pr_episode, mean_reward_pr_episode)

            log_to_file = "{}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, mean_reward_pr_episode, mean_reward_pr_episode)

            print(log)

            # Save to files
            with open(log_filename, "a") as myfile:
                myfile.write(log_to_file)

            # Saving the agent
            torch.save(ac_agent, "models/" + model_name)

            rewards = 0
            rewards_shaped = 0
            episodes = 0
            value_losses.clear()
            policy_losses.clear()

        if all_steps >= num_steps:
            # Saving the agent
            torch.save(ac_agent, "models/" + model_name)
            envs.close()
            return


def update_obs(observations):
    """
    Takes the observation returned by the environment and transforms it to an numpy array that contains all of
    the feature layers and non-spatial info
    """
    spatial_obs = []
    non_spatial_obs = []

    for obs in observations:
        spatial_ob = np.stack(obs['board'].values())

        state = list(obs['state'].values())
        procedures = list(obs['procedures'].values())
        actions = list(obs['available-action-types'].values())

        non_spatial_ob = np.stack(state+procedures+actions)

        # feature_layers = np.expand_dims(feature_layers, axis=0)
        non_spatial_ob = np.expand_dims(non_spatial_ob, axis=0)

        spatial_obs.append(spatial_ob)
        non_spatial_obs.append(non_spatial_ob)

    return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()


def make_env(worker_id):
    print("Initializing worker", worker_id, "...")
    env = gym.make(env_name)
    return env


if __name__ == "__main__":
    main()
