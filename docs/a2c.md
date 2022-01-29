# Reinforcement Learning II: A2C
In this tutorial we will train a reinforcement learning agent to play Blood Bowl using the synchronous advantage actor-critic 
(A2C) algorithm, which is a simpler variant of A3C (Asynchronous Advantage Actor-Critic). This tutorial will be hard to follow without 
a basic understanding of deep reinforcement learning, including concepts such as neural networks, convolution, back-propagation, gradient descent,
temporal difference learning, and parallelization in RL. If you are unfamiliar with some of these concepts, we recommend that you first read through some of these resources:

- Deep learning ([Deep Learning by Youshua Bengio](https://www.google.com/search?q=yoshua+bengio+deep+learning+book))
- Reinforcement Learning ([Richard Sutton's book Reinforcement Learning: An Introduction](https://www.google.com/search?q=richard+sutton+introduction+reinforcement+learning))
- Asynchronous/Synchronous Advantage Actor-Critic methods ([Asynchronous Methods for Deep Reinforcement Learning by Mnih et al.](https://arxiv.org/pdf/1602.01783.pdf) and [OpenAI Baselines: ACKTR & A2C](https://openai.com/blog/baselines-acktr-a2c/)) 

The code presented in this tutorial is based on [examples/a2c/a2c_example.py](https://github.com/njustesen/botbowl/blob/master/examples/a2c/a2c_example.py) which is inspired by a [pytorch implementation 
of A2C](https://github.com/p-kar/a2c-acktr-vizdoom) by Pratyush Kar and the [botbowl-A2C](https://github.com/lasseuth1/blood_bowl2) 
repository by Lasse Møller Uth and Christopher Jakobsen, which is no longer compatible with the newer versions of botbowl. 

In this tutorial, we focus on three small variants of Blood Bowl: 

- **botbowl-1-v2:** 1 player on a 4x3 pitch for each team.
- **botbowl-3-v2:** 3 player on a 12x5 pitch for each team.
- **botbowl-5-v2:** 5 player on a 16x9 pitch for each team.

It was previously shown in the paper [Blood Bowl: A New Board Game Challenge
and Competition for AI](https://njustesen.github.io/njustesen/publications/justesen2019blood.pdf) that A2C with reward shaping 
was able to learn strong policies in these variants against a random opponent.

In future tutorials, we will attempt to: 

1. scale this approach to the 11-player variant of the game
2. apply self-play to achieve a robust policy against diverse opponents


## Environment 
We will use the [default observation space and action space](gym.md) in botbowl, we'll use a reward wrapper to define 
our custom reward (more details in the next section). For now we won't use the scripted wrapper. 

```python 
# Environment
env_size = 1  # Options are 1,3,5,7,11
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=False)

def make_env():
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env
```
The pathfinding option is explored in [Reinforcement Learning V: Pathfinding Assistance](a2c-pathfinding.md)

## Reward Shaping

The default rewards in botbowl are based on the game outcome. To ease the learning, we will shape the reward function in 
several ways. First, we define the reward value of certain events. E.g. if the agent scores a touchdown it is rewarded with a value of 1 and if 
the opponent scores it is rewarded (punished) with a value of -1.

Additionally, we give the agent a reward for moving the ball toward the opponent endzone. Here, we give a reward of 0.005 for 
every square the ball is moved closer. The reward function itself is defined like this:

```python
# found in examples/a2c/a2c_env.py 
from botbowl.core.game import Game 
from botbowl.core.table import OutcomeType

class A2C_Reward:
    # --- Reward function ---
    rewards_own = {
        OutcomeType.TOUCHDOWN: 1,
        OutcomeType.SUCCESSFUL_CATCH: 0.1,
        OutcomeType.INTERCEPTION: 0.2,
        OutcomeType.SUCCESSFUL_PICKUP: 0.1,
        OutcomeType.FUMBLE: -0.1,
        OutcomeType.KNOCKED_DOWN: -0.1,
        OutcomeType.KNOCKED_OUT: -0.2,
        OutcomeType.CASUALTY: -0.5
    }
    rewards_opp = {
        OutcomeType.TOUCHDOWN: -1,
        OutcomeType.SUCCESSFUL_CATCH: -0.1,
        OutcomeType.INTERCEPTION: -0.2,
        OutcomeType.SUCCESSFUL_PICKUP: -0.1,
        OutcomeType.FUMBLE: 0.1,
        OutcomeType.KNOCKED_DOWN: 0.1,
        OutcomeType.KNOCKED_OUT: 0.2,
        OutcomeType.CASUALTY: 0.5
    }
    ball_progression_reward = 0.005

    def __init__(self):
        self.last_report_idx = 0
        self.last_ball_x = None
        self.last_ball_team = None

    def __call__(self, game: Game):
        if len(game.state.reports) < self.last_report_idx:
            self.last_report_idx = 0

        r = 0.0
        own_team = game.active_team
        opp_team = game.get_opp_team(own_team)

        for outcome in game.state.reports[self.last_report_idx:]:
            team = None
            if outcome.player is not None:
                team = outcome.player.team
            elif outcome.team is not None:
                team = outcome.team
            if team == own_team and outcome.outcome_type in A2C_Reward.rewards_own:
                r += A2C_Reward.rewards_own[outcome.outcome_type]
            if team == opp_team and outcome.outcome_type in A2C_Reward.rewards_opp:
                r += A2C_Reward.rewards_opp[outcome.outcome_type]
        self.last_report_idx = len(game.state.reports)

        ball_carrier = game.get_ball_carrier()
        if ball_carrier is not None:
            if self.last_ball_team is own_team and ball_carrier.team is own_team:
                ball_progress = self.last_ball_x - ball_carrier.position.x
                if own_team is game.state.away_team:
                    ball_progress *= -1  # End zone at max x coordinate
                r += A2C_Reward.ball_progression_reward * ball_progress

            self.last_ball_team = ball_carrier.team
            self.last_ball_x = ball_carrier.position.x
        else:
            self.last_ball_team = None
            self.last_ball_x = None

        return r
```

## Policy Network

We employ a relatively simple neural network architecture for the policy. The spatial feature layers are fed through two convolutional layers that use 
3x3 filters with padding and stride 1. This maintains the size of the spatial input. The output of the this stream is flattened and then concatenated 
with the non-spatial input. The resulting vector then goes through a fully-connected layer and is then split up into the action output and state-value output. 
The action output is a fully-connected layer with a softmax activation function forming a distribution over actions. The state-value output 
is also fully-connected layer but without any activation function. 

![Network architecture](img/architecture.png?raw=true "Network architecture")

We used pytorch to implement this policy network:

````python
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

    ...

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

        # Output streams
        value = self.critic(x3)
        actor = self.actor(x3)

        # return value, policy
        return value, actor
````

To sample an action from the action output we use the following two functions. 

````python
class CNNPolicy(nn.Module):
    ...
    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)  # Sample actions
        return values, actions
    
    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs
````

Notice that an action mask is used, i.e. we only sample for actions that are actually available. We do this by overriding the action output of all 
unavailable actions, setting them to the value ```float('-inf')``` since softmax will compute such values to 0. Notice, that these functions are called 
with batches of observations and thus batches of actions are returned.
 

## Parallelization
A2C uses parallel worker processes to speed up the collection of experiences. First, we define a worker function that 
we will run in a new process:

```python
# found in examples/a2c/a2c_example.py 
from botbowl.ai.env import BotBowlWrapper 
reset_steps = 3000

def worker(remote, parent_remote, env: BotBowlWrapper, worker_id):
    parent_remote.close()

    steps = 0
    tds = 0
    tds_opp = 0

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action = data[0]

            spatial_obs, reward, done, info = env.step(action)
            non_spatial_obs = info['non_spatial_obs']
            action_mask = info['action_mask']

            game = env.game
            tds_scored = game.state.home_team.state.score - tds
            tds_opp_scored = game.state.away_team.state.score - tds_opp
            tds = game.state.home_team.state.score
            tds_opp = game.state.away_team.state.score

            if done or steps >= reset_steps:
                # If we get stuck or something - reset the environment
                if steps >= reset_steps:
                    print("Max. number of steps exceeded! Consider increasing the number.")
                done = True
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
            env.reset()
            spatial_obs, non_spatial_obs, action_mask = env.get_state()
            remote.send((spatial_obs, non_spatial_obs, action_mask, 0.0, 0, 0, False))

        elif command == 'close':
            break
```

Notice the ```while``` loop that will run until it is terminated (when a 'close' command is called) and will execute steps in the environment when queried. 
We use a meta-environment class called ```VecEnv``` that abstracts the communication with these workers by acting as list of 
botbowl environments. We can thus call ```step()``` with a list of actions, one for each environment, and it returns a list of 
future observations and rewards. This is practical as we can query our policy network with a batch of observations.

```python
# found in examples/a2c/a2c_example.py 

from multiprocessing import Process, Pipe
from typing import Iterable, Tuple

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
```

## A2C Configurations

We use the following configurations:

```python
# Training configuration
num_steps = 100000
num_processes = 8
steps_per_update = 20
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
log_interval = 50
save_interval = 10


# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]
```


## Training

We make the code agnostic to which variant of botbowl we are using by first computing the size of action and observation space.

We then instantiate our policy model using the computed sizes and an [RMSprop](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) optimizer. 
A memory store is then created that will contain trajectories of our agent. Trajectories will be 20 steps long for each worker. 

```python
def main():
    envs = VecEnv([make_env() for _ in range(num_processes)])

    env = make_env()
    env.reset()
    spat_obs, non_spat_obs, action_mask = env.get_state()
    del env
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)

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

    # Reset environments
    spatial_obs, non_spatial_obs, action_masks, _, _, _, _ = map(torch.from_numpy, envs.reset(difficulty))

    # Add obs to memory
    non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=1)
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)
    memory.action_masks[0].copy_(action_masks)

```

When the memory is full (20 steps has been reached), reward values are discounted and training is performed on the data in the memory where after it is flushed.

Take a look for yourself at the remaining code in [examples/a2c/a2c_example.py](https://github.com/njustesen/botbowl/blob/master/examples/a2c/a2c_example.py) to understand how the learning works. 
Because the neural network is written in pytorch, you can run the code in debug mode and go through the code line by line.

## Results

The training process is plotted as the script runs. Here are the results we got on the three small botbowl environments.

### botbowl-1-v2

![botbowl-1-v2 Results](img/botbowl-1-v2.png?raw=true "botbowl-1-v2 Results")

### botbowl-3-v2

![botbowl-3-v2 Results](img/botbowl-3-v2.png?raw=true "botbowl-3-v2 Results")

### botbowl-5-v2

![botbowl-5-v2 Results](img/botbowl-5-v2.png?raw=true "botbowl-5-v2 Results")

Run the script to see if you can reproduce some of the results. Can you find better hyper-parameters or reward shaping rules?

## Next Steps
We were able to learn strong policies for the one, three, and five-player variants of Blood Bowl against a random opponent. 
In the next tutorial, we will improve our code to use several enhancements to reach the same results on the 11-player variant.
