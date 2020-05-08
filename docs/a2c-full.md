# Reinforcement Learning II: The Full Board

In the previous tutorial we learned how to apply the reinforcement learning (RL) algorithm to FFAI's Gym environments, however, only on the 
variants with smaller board sizes. In this tutorial, we will first describe our attempt to apply the same approach to the variant with 
the full board size, where the bot doesn't learn much within 100M steps. After that, we apply a trick that increases the difficulty of the task 
as the agent learns.

## Vanilla A2C

Let's do a quick estimate how many steps we will need to learn a policy on the full board with our current approach. 
In the smaller variants, we achieved a reasonably good skill level (at least against _random_) after the following number of training steps:

- 1 player: 200K
- 3 players: 8M
- 5 players: 80M

Let's make a guess and say that if we were to continue onto the 7 player, 9 player (actually this doesn't exist) and 11 player variant, they would require  times more compute each time we increase the complexity, i.e.:

- 7 players: 800M
- 9 players: 8000M
- 11 players: 80000M

That is 8 billion steps to learn a policy for the full game. Maybe this is completely wrong and it doesn't grow linearly. 
In any case, I don't have anywhere near those resource, so let's just try and run 100M steps as we did in the five player variant.
Before running the training, the number of hidden nodes in the fully-connected layer were set to ```num_hidden_nodes = 512```. 
However, I would suspect that the whole model would need even more parameters (also in the convolutional layers) to play the full game well. 
This experiment took me an entire week to run on my desktop machine, so I would definitely suggest you to acquire some compute online.

![A2C on FFAI-v2](img/FFAI-v2.png?raw=true "A2C on FFAI-v2")

The rewards do increase as it probably learns to make a few blocks and avoid making risky dodges, while it is far from learning how to score consistently. 
It does, however, seem like it's very slowly starting to learn how to find the endzone with an average TD rate per game around 0.02.

## Progressively Increasing the Difficulty

There are possibly many ways to further guide the agent towards the endzone and I hope we will see many different approaches to this problem.
On tecnnique that I previously had success with is to start out with very simple tasks and then progressively increase the difficulty as the agent learns as 
a form of curriculum learning. In my previous experiments (which you can read more about in our [paper](https://arxiv.org/abs/1806.10729)) we used a procedural level generator to initially generate 
easy levels and when the agent completed a level the difficulty was increased. In contrast, when the agent did not complete a level, the difficulty was decreased. 

While it seems a bit weird to generate different Blood Bowl game boards for the agent, starting simple and then slowly increasing the difficulty, we can do this in a very simple way.
We keep the normal board but increase the size of the endzone such that for an easy "level" the agent scores a touchdown just by carrying the ball on any square on the board. As the difficulty increases,
the width of the endzone gets smaller and ofcourse only for the learning agent, not the opponent. Hopefully, we can then reach the maximum difficulty such that the agent learns to score 
on the 1-square wide endzone as in the real game.

In the [examples/a2c/a2c_example.py](https://github.com/njustesen/ffai/blob/master/examples/a2c/a2c_example.py), you can enable this feature by setting ```ppcg = True```.
What it does is to simply run the Touchdown procedure directly in the game when the agent has the ball in is within a certain distance of the endzone.
The is done with the following code:

```python
def worker(remote, parent_remote, env, worker_id):
    ...
    while True:
        command, data = remote.recv()
        if command == 'step':
            ...
            ball_carrier = env.game.get_ball_carrier()
            # PPCG
            if dif < 1.0:
                if ball_carrier and ball_carrier.team == env.game.state.home_team:
                    extra_endzone_squares = int((1.0 - dif) * 25.0)
                    distance_to_endzone = ball_carrier.position.x - 1
                    if distance_to_endzone <= extra_endzone_squares:
                        #reward_shaped += rewards_own[OutcomeType.TOUCHDOWN]
                        env.game.state.stack.push(Touchdown(env.game, ball_carrier))
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

Here, we use ```dif_delta = 0.01```. Note, that if ````ppcg = False```` we always set ```difficulty = 1``` so that a normal endzone is used.

Let's see how the results are, again using _just_ 100M training steps.

![A2C on FFAI-v2 with PPCG](img/FFAI-ppcg-v2.png?raw=true "A2C on FFAI-v2 with PPCG")

We see that the difficulty reaches ~1 as we had hoped. Unfortunately, the win rate only reaches around 80%. This is, however, a really nice start.

## Watch it Play

Atm. the policy is just a neural network and we have a bunch of code around it that works just for the gym environments but not the rest of the FFAI framework. 
Additionally, the bot was only trained to play as the home time and would not know how to play on the other side of the field. Let's fix these things, so we can watch 
our agent play, and even play against it. The code that will be presented can also be used to submit your own neural network based bot to the Bot Bowl competition.

The [examples/a2c/a2c_example.py](https://github.com/njustesen/ffai/blob/master/examples/a2c/a2c_agent.py) script implements the ```Agent``` class just like the 
scripted bots in our previous tutorials. In the constructor of our Agent class, we load in our neural network policy.

```python
...
model_filename = "my-model"
class A2CAgent(Agent):

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.env = self.make_env('FFAI-v2')

        self.spatial_obs_space = self.env.observation_space.spaces['board'].shape
        self.board_dim = (self.spatial_obs_space[1], self.spatial_obs_space[2])
        self.board_squares = self.spatial_obs_space[1] * self.spatial_obs_space[2]

        self.non_spatial_obs_space = self.env.observation_space.spaces['state'].shape[0] + \
                                self.env.observation_space.spaces['procedures'].shape[0] + \
                                self.env.observation_space.spaces['available-action-types'].shape[0]
        self.non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
        self.num_non_spatial_action_types = len(self.non_spatial_action_types)
        self.spatial_action_types = FFAIEnv.positional_action_types
        self.num_spatial_action_types = len(self.spatial_action_types)
        self.num_spatial_actions = self.num_spatial_action_types * self.spatial_obs_space[1] * self.spatial_obs_space[2]
        self.action_space = self.num_non_spatial_action_types + self.num_spatial_actions
        self.is_home = True

        # MODEL
        self.policy = torch.load(model_filename)
        self.policy.eval()
        self.end_setup = False
    ...
```

In the agent's ``act()``` implementation, we will steal a bit of code from out training loop that calls our neural network. 
Additionally, if the agent is playing as the away team, we need to flip the spatial observation of the board, as it is now playing on 
the opposite side of the board.

```python
    def _flip(self, board):
        flipped = {}
        for name, layer in board.items():
            flipped[name] = np.flip(layer, 1)
        return flipped

    def act(self, game):

        if self.end_setup:
            self.end_setup = False
            return ffai.Action(ActionType.END_SETUP)

        # Get observation
        self.env.game = game
        observation = self.env.get_observation()

        # Flip board observation if away team - we probably only trained as home team
        if not self.is_home:
            observation['board'] = self._flip(observation['board'])

        obs = [observation]
        spatial_obs, non_spatial_obs = self._update_obs(obs)

        action_masks = self._compute_action_masks(obs)
        action_masks = torch.tensor(action_masks, dtype=torch.bool)

        values, actions = self.policy.act(
            Variable(spatial_obs),
            Variable(non_spatial_obs),
            Variable(action_masks))

        # Create action from output
        action = actions[0]
        value = values[0]
        action_type, x, y = self._compute_action(action.numpy()[0])
        position = Square(x, y) if action_type in FFAIEnv.positional_action_types else None

        # Flip position
        if not self.is_home and position is not None:
            position = Square(game.arena.width - 1 - position.x, position.y)

        action = ffai.Action(action_type, position=position, player=None)

        # Let's just end the setup right after picking a formation
        if action_type.name.lower().startswith('setup'):
            self.end_setup = True

        # Return action to the framework
        return action
```

The rest of the file should be familiar if you followed our tutorials on scripted bots.

Try training an agent with ````ppcg = True``` and see if you can add it to an agent and watch it play.
The bot I trained won  against the random baseline, and an example game looks like this.

<iframe src="https://www.youtube.com/embed/feKzUzRo9DM" 
    width="560" 
    height="315"
    frameborder="0" 
    allowfullscreen>
</iframe>

It seems that the policy learned the following:

- If the opponent kicks the ball out of bounds (happens quite often for the _random_ bot) give the ball to the blitzer on the line of scrimmage. Then 
blitz and move forward towards the endzone and hopefulle score.
- If the ball lands in the backfield, move players randomly around in the hope of picking it up. If the ball is close enough to the line of scrimmage it can 
sometimes score from this situation.
- If standing adjacent to the opponent's ball carrier, block him.

Can you come up with ways to improve upon this? Can you achieve a higher win rate with more parameters and more training?