# Reinforcement Learning III: The Full Board

In the previous tutorial we learned how to apply the reinforcement learning (RL) algorithm A2C to botbowl's gym environments, however, only on the 
variants with smaller board sizes. In this tutorial, we will first describe our attempt to apply the same approach to the variant with 
the full board size, where the bot doesn't learn much within 100M steps. After that, we apply a trick that increases the difficulty of the task 
as the agent learns.

## Vanilla A2C

Let's do a quick estimate of how many steps we will need to learn a policy on the full board with our current approach. 
In the smaller variants, we achieved a reasonably good skill level (at least against _random_) after the following number of training steps:

- 1 player: 200K
- 3 players: 8M
- 5 players: 80M

If we were to continue onto the 7-player, 9-player (actually this doesn't exist) and 11-player variants, we would probably need hundreds or even thousands of millions of steps.
In any case, it's a lot, so let's just try and run 100M steps as we did in the five player variant.
Before running the training, the number of hidden nodes in the fully-connected layer was doubled: ```num_hidden_nodes = 512```. 
However, I would suspect that the whole model would need even more parameters (also in the convolutional layers) to play the full game well. 
This experiment took me an entire week to run on my desktop machine.

![A2C on botbowl-v2](img/botbowl-v2.png?raw=true "A2C on botbowl-v2")

The rewards do increase as it learns to make a few blocks and avoid making risky dodges, while it is far from learning how to score consistently. 
It is, however, very slowly learning to find the endzone with an average TD rate per game of ~0.02 (sorry for the really bad rendering of the plot). 
Remember that draws here count as half a win, so if none of the agents ever win, the win rate is 50%.

## Progressively Increasing the Difficulty

There are possibly many ways to further guide the agent towards the endzone and I hope we will see many different approaches to this problem.
One tecnnique that I previously have had success with is to start out with very simple tasks and then progressively increase the difficulty as the agent learns as 
a form of curriculum learning. In my previous experiments (which you can read more about in our [paper](https://arxiv.org/abs/1806.10729)) we used a procedural level generator to initially generate 
easy video game levels and when the agent completed a level the difficulty was increased. In contrast, when the agent did not complete a level, the difficulty was decreased. 

While it seems a bit weird to generate different Blood Bowl game boards for the agent, we can control the difficulty in a very simple way.
We keep the normal board structure but increase the width of the endzone such that for an easy "level" the agent scores a touchdown just by carrying the ball on any square on the board. As the difficulty increases,
the width of the endzone gets smaller and of course only for the learning agent, not the opponent. Hopefully, we can then reach the maximum difficulty such that the agent learns to score 
on the 1-square wide endzone as in the real game.

In the [examples/a2c/a2c_example.py](https://github.com/njustesen/botbowl/blob/master/examples/a2c/a2c_example.py), you can enable this feature by setting ```ppcg = True```.
What it does is simply to run the ```Touchdown()``` procedure directly in the game when the agent has the ball while within a certain distance of the endzone.
This is done with the `PPCGWrapper`: 

```python
class PPCGWrapper(BotBowlWrapper):
    difficulty: float

    def __init__(self, env, difficulty=1.0):
        super().__init__(env)
        self.difficulty = difficulty

    def step(self, action: int, skip_observation: bool = False):
        self.env.step(action, skip_observation=True)

        if self.difficulty < 1.0:
            game = self.game
            ball_carrier = game.get_ball_carrier()
            if ball_carrier and ball_carrier.team == game.state.home_team:
                extra_endzone_squares = int((1.0 - self.difficulty) * 25.0)
                distance_to_endzone = ball_carrier.position.x - 1
                if distance_to_endzone <= extra_endzone_squares:
                    game.state.stack.push(Touchdown(game, ball_carrier))
                    game.set_available_actions()
                    self.env.step(None, skip_observation=True)  # process the Touchdown-procedure 

        return self.root_env.get_step_return(skip_observation=skip_observation)
````

The difficulty is adjusted in the training loop like this:

```python
for i in range(num_processes):
    if done[i]:
        if r[i] > 0:
            wins.append(1)
            difficulty += dif_delta
        elif r[i] < 0:
            wins.append(0)
            difficulty -= dif_delta
        else:
            wins.append(0.5)
            difficulty -= dif_delta
        if ppcg:
            difficulty = min(1.0, max(0, difficulty))
        else:
            difficulty = 1
```

And the difficulity is sent to the worker and processed like this: 

```python
def worker(remote, parent_remote, env: BotBowlWrapper, worker_id):
    # ... 
    ppcg_wrapper: Optional[PPCGWrapper] = env.get_wrapper_with_type(PPCGWrapper)

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action, dif = data[0], data[1]
            if ppcg_wrapper is not None:
                ppcg_wrapper.difficulty = dif
```


Here, we use ```dif_delta = 0.01```. Note, that if ````ppcg = False```` we always set ```difficulty = 1``` so that a normal endzone is used.

Let's see how the results are, again using _just_ 100M training steps.

![A2C on botbowl-v2 with PPCG](img/botbowl-ppcg-v2.png?raw=true "A2C on botbowl-v2 with PPCG")

We see that the difficulty reaches ~1 as we had hoped. Unfortunately, the win rate only reaches around 80%. This is, however, a really nice start.

## Watch it Play

At the moment, the policy is just a neural network with some code around it so that it works for the gym environments but not the rest of the botbowl framework. 
Additionally, the bot was only trained to play as the home team and would not know how to play on the other side of the field. Let's fix these things so we can watch 
our agent play, and even play against it. The code that will be presented can also be used to submit your own neural network based bot to the Bot Bowl competition.

The [examples/a2c/a2c_agent.py](https://github.com/njustesen/botbowl/blob/master/examples/a2c/a2c_agent.py) script implements the ```Agent``` class just like the 
scripted bots in our previous tutorials. In the constructor of our Agent class, we load in our neural network policy.

```python
...
model_filename = "my-model"
class A2CAgent(Agent):
    env: BotBowlEnv

    def __init__(self, name,
                 env_conf: EnvConf,
                 scripted_func: Callable[[Game], Optional[Action]] = None,
                 filename=model_filename,
                 exclude_pathfinding_moves=True):
        super().__init__(name)
        self.env = BotBowlEnv(env_conf)
        self.exclude_pathfinding_moves = exclude_pathfinding_moves

        self.scripted_func = scripted_func
        self.action_queue = []

        # MODEL
        self.policy = torch.load(filename)
        self.policy.eval()
        self.end_setup = False
```

The env_name argument is particularly important as it should be the environment name that the model was trained on. 
If the model was trained on ```botbowl-11-v2``` (where pathfinding is disabled)  use that environment name when instantiating A2CAgent. 
The agent can still play in games with pathfinding enabled. It will do this by excluding pathfinding-assisted move actions later.

In the agent's ```act()``` implementation, we will steal a bit of code from our training loop that calls our neural network. 

Note that if the agent is playing as the away team, we need to flip the spatial observation of the board, as it is now playing on 
the opposite side of the board. Luckily, the environment will do that for us, so we don't need to consider it here. And the environment 
helps us flip any spatial action too. 

```python
    def _filter_actions(self):
        """
        Remove pathfinding-assisted non-adjacent or block move actions if pathfinding is disabled.
        """
        actions = []
        for action_choice in self.env.game.state.available_actions:
            if action_choice.action_type == ActionType.MOVE:
                positions, block_dice, rolls = [], [], []
                for i in range(len(action_choice.positions)):
                    position = action_choice.positions[i]
                    roll = action_choice.paths[i].rolls[0]
                    # Only include positions where there are not players
                    if self.env.game.get_player_at(position) is None:
                        positions.append(position)
                        rolls.append(roll)
                actions.append(ActionChoice(ActionType.MOVE, team=action_choice.team, positions=positions, rolls=rolls))
            else:
                actions.append(action_choice)
        self.env.game.state.available_actions = actions

    @staticmethod
    def _update_obs(array: np.ndarray):
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def act(self, game):
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        if self.scripted_func is not None:
            scripted_action = self.scripted_func(game)
            if scripted_action is not None:
                return scripted_action

        self.env.game = game

        # Filter out pathfinding-assisted move actions
        if self.exclude_pathfinding_moves and self.env.game.config.pathfinding_enabled:
            self._filter_actions()

        spatial_obs, non_spatial_obs, action_mask = map(A2CAgent._update_obs, self.env.get_state())
        non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=0)

        _, actions = self.policy.act(
            Variable(spatial_obs.float()),
            Variable(non_spatial_obs.float()),
            Variable(action_mask))

        action_idx = actions[0]
        action_objects = self.env._compute_action(action_idx, flip=self.env._flip_x_axis())

        self.action_queue = action_objects
        return self.action_queue.pop(0)
```

The rest of the file should be familiar if you followed our tutorials on scripted bots.

Try training an agent with ```ppcg = True``` and see if you can add it to an agent and watch it play.
The bot I trained won  against the random baseline, and an example game looks like this.

<iframe src="https://www.youtube.com/embed/feKzUzRo9DM" 
    width="100%" 
    height="420"
    frameborder="0" 
    allowfullscreen>
</iframe>

It seems that the policy learned the following:

- If the opponent kicks the ball out of bounds (happens quite often for the _random_ bot) give the ball to a player on the line of scrimmage (that has an assist). Then 
blitz and move towards the endzone and hopefully score.
- If the ball lands in the backfield, move players randomly around in the hope of picking it up. If the ball is close enough to the line of scrimmage it can 
sometimes score from this situation.
- If standing adjacent to the opponent's ball carrier, block him.
- Perform reasonable blocks at the line of scrimmage.
- It's hard to see, but it seems that agent is fouling a lot on the line of scrimmage as well.

Can you come up with better ways to guide the RL agent during training? Can you achieve a higher win rate with more parameters and more training?
