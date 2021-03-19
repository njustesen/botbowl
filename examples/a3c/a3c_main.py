import gym
from ffai import FFAIEnv
from pytest import set_trace
from torch.autograd import Variable
import torch.optim as optim
from ffai.ai.layers import *
import torch
import torch.nn as nn
from a2c_agent import A2CAgent, CNNPolicy

from VectorEnvironment import VecEnv
from Curriculum import Academy
import Lectures

# Training configuration
max_updates = 20000
num_processes = 6
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
prediction_loss_coeff = 0.1
max_grad_norm = 0.05

# Environment
env_name = "FFAI-1-v3"

# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]

# Pathfinding-assisted paths enabled?
pathfinding_enabled = True

model_name = env_name
log_filename = "logs/" + model_name + ".dat"


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


def calc_network_shape(env):
    spatial_obs_space = env.observation_space.spaces['board'].shape
    board_dim = (spatial_obs_space[1], spatial_obs_space[2])
    board_squares = spatial_obs_space[1] * spatial_obs_space[2]

    non_spatial_obs_space = env.observation_space.spaces['state'].shape[0] + \
                            env.observation_space.spaces['procedures'].shape[0] + \
                            env.observation_space.spaces['available-action-types'].shape[0]
    non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
    num_non_spatial_action_types = len(non_spatial_action_types)
    spatial_action_types = FFAIEnv.positional_action_types
    num_spatial_action_types = len(spatial_action_types)
    num_spatial_actions = num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
    action_space = num_non_spatial_action_types + num_spatial_actions

    shape = {'spat_obs': spatial_obs_space,
             'non_spat_obs': non_spatial_obs_space,
             'board_dim': board_dim,
             'num_spat_action_types': num_spatial_action_types,
             'num_spat_actions': num_spatial_actions,
             'num_non_spat_actions': num_non_spatial_action_types,
             'action_space': action_space}
    return shape


def main():

    # Clear log file
    try:
        os.remove(log_filename)
    except OSError:
        pass

    es = [make_env(i) for i in range(num_processes)]
    shape = calc_network_shape(es[0])

    # MODEL
    ac_agent = CNNPolicy(shape['spat_obs'], shape['non_spat_obs'], hidden_nodes=num_hidden_nodes,
                         kernels=num_cnn_kernels, actions=shape['action_space'])

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

    # Create the agent
    torch.save(ac_agent, "models/" + model_name)
    agent = A2CAgent("trainee", env_name=env_name, filename="models/" + model_name)

    # send agent to environments
    envs = VecEnv([es[i] for i in range(num_processes)], agent, 1000)

    updates = 0

    while updates < max_updates:
        envs.memory.step = 0  # TODO: This is naughty!
        # Step until memory is filled, access memory through envs.memory
        envs.step()
        memory = envs.memory

        # ### Evaluate the actions taken ###
        spatial = Variable(memory.spatial_obs[:memory.step])
        # spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs[:memory.step])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(torch.LongTensor(memory.actions[:memory.step].view(-1, 1)))
        actions_mask = Variable(memory.action_masks[:memory.step])

        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        # ### Compute loss and back propagate ###
        # values = values.view(steps_per_update, num_processes, 1)
        # action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns[:memory.step]) - values
        value_loss = advantages.pow(2).mean()

        outcome_prediction_error = Variable(memory.td_outcome[:memory.step]) - td_pred

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()

        total_loss = (
                    value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()

        updates += 1

        # send updated agent to workers
        agent.policy = ac_agent

        envs.update_trainee(agent)
        if updates % 5 == 0:
            print(f"Update {updates}/{max_updates}")

    # torch.save(ac_agent, "models/" + model_name)
    print("closing workers!")

    envs.close()
    print("main() quit!")


def make_env(worker_id):
    print("Initializing worker", worker_id, "...")
    env = gym.make(env_name)
    return env


if __name__ == "__main__":
    main()
