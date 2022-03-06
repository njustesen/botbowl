# Reinforcement Learning V: Pathfinding Assistance

The new pathfinding-assisted move actions in botbowl allow bots to move players to any reachable square on the board. 
When activating a player action other than the block actions, safe paths will be computed for all the reachable squares. 
Each path will have a probability of success associated to it, or a number of block dice if the path ends with a block action 
during a blitz move. Similarly, the catch roll target is given for handoff moves.

# Enabling Pathfinding Assistance

The pathfinding-assisted move actions are disabled by default in the botbowl gym environments but can be enabled by calling:

```python
env_conf = EnvConf(..., pathfinding=True)
```

In examples/a2c_examples.py, simply set the ```pathfinding_enabled = True``` in the variables at the top.

This will add a new feature layer to the observation space and it will also affect the values in the existing _move positions_ layer.

![Pathfinding feature layers](img/gym-pathfinding-layers.png?raw=true "Feature layers")

# Training

An interesting research question is whether a RL agent will benefit from pathfinding assisted move actions or if 
it's easier to learn by moving one square at a time. Intuitively, it should be easier to score for a randomly initialized policy
if it can sample all reachable squares rather than moving on square at a time, simply because it's more likely to have moved further by the end of the player activation. 

A drawback of having pathfinding Assistance enabled is that pathfinding is computational expensive. We would thus expect the 
environment to be slower while the agent may spent fewer actions every episode.

Simply run examples/a2c_example.py with ```env.config.pathfinding_enabled = True``` to train the agent.

# Results
As an experiment, we have trained agents for the 1-player, 3-player, and 5-player variants in botbowl, with and without pathfinding assistance enabled to test the difference.

![botbowl-1 with pathfinding](img/botbowl-1-pf.png?raw=true "botbowl-1 with pathfinding")

![botbowl-3 with pathfinding](img/botbowl-3-pf.png?raw=true "botbowl-3 with pathfinding")

![botbowl-5 with pathfinding](img/botbowl-5-pf.png?raw=true "botbowl-5 with pathfinding")

Interestingly, our agents with pathfinding assistance enabled did were not more data efficient than the agents without.
Maybe our intuition was wrong, or maybe we were missing important information in the feature layers, or something else.

We challenge you to find better ways to use the pathfinding assistance in RL.

# Training Time

**ToDo: re-run training with compiled pathfinding. Should be many times faster.** 

We forgot to time the experiments, but the effect of having pathfinding-assistance enabled were roughly this:

- botbowl-1-v3 (1-player): A few times slower
- botbowl-3-v3 (3-players): ~5 times slower
- botbowl-5-v3 (5-players): ~10 times slower

It took two weeks to run the botbowl-5-v3 experiment to the end with pathfinding-assistance enabled. 
This is obviously a big step back, but there we found a big opportunity for improvement. 
Since the A2C training algorithm takes synchronous steps in all its parallel environments, and only some of them will be at a point where it is computing paths, several envrionment processes are idle for a long period of time (perhaps something like 100 ms.).
We also noticed that the CPU usage were only 10-30% on each core on our training instance. 
An asynchronous variant, such as A3C or something more modern, could improve the training speed a lot!

Maybe one day we will post a tutorial where we try this out but for now, we leave this as a challenge for you.

# Running on the full game
If you manage to train an agent on the full game using pathfinding and want to play against it, or submit it to a competition, then use the A2CAgent as described in [**Reinforcement Learning III: The Full Board**](a2c-full.md). 
It is important that you set exclude_pathfinding_moves = False in the A2CAgent constructor if you want the agent to use the pathfinding-assisted moves.