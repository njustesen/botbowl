![Bot Bowl III](img/botbowl-iii.png?raw=true "Bot Bowl III")

Bot Bowl III was the third AI competition in Blood Bowl and and happened (virtually) at the [IEEE Conference of Games 2021](https://ieee-cog.org/2021/index.html). 

**See the results** further down on this page or watch [the announcement](https://youtu.be/A_7tgr3r5IA) from the Conference of Games.

# Format
Bot Bowl III will have the same format as last year while the framework offers a few new features that will empower bot developers. 

Each bot will play 10 matches of Blood Bowl against each other and the winner will determined based on the following point system:
3 points for winning, 1 point for a draw, 0 points for a loss. If the bot crashes and are unable to recover, the game will 
continue and the bot will simply timeout and not do anything during their turn.

Each match will using the following configurations:

- The BB2016 ruleset
- Only a fixed human teams
- Time limit of 2 minutes per turn and 15 seconds per decision in the opponent's turn
- Pathfinding-assisted move actions are enabled (read more about this in the New Features section below)

Group submissions are allowed but each person can only be part of one bot submission.

# Prizes
Bot Bowl III is funded by [IEEE Computational Intelligence Society](https://cis.ieee.org/) and [modl.ai](http://www.modl.ai) and has a prize pool of $1,500.

**IEEE CIS Sponsored Prizes:**
- $500 for the winner
- $300 for the second place
- $200 for the third place

**modl.ai Sponsored Prize:**
- $500 for the most impressive machine learning bot.

The machine learning prize sponsored by modl.ai of $500 will be given to the most impressive bot that relies on machine learning or search algorithms. A jury of AI Researchers from modl.ai will decide the winner of this prize based on the competition results, the implementation, and the submitted descriptions of the bots.

# New Features

## Pathfinding-assisted Move Actions
In previous Bot Bowls, bots could only move to adjacent squares, step by step. In Bot Bowl III, however, the botbowl framework 
will provide bots with available move actions to all reachable squares together with the safest and shortest path to reach it.
Bots are then able to perform move actions to squares that are further away than one step, and the framework will perform all 
the intermediate steps automatically. 

**Available actions with pathfinding-assistance:**

![pathfinding](img/pathfinding.png?raw=true "Pathfinding-assisted actions")

*The user interface is, for demonstration purposes, showing here the safest path to blitz the opponent safety lineman. All green squares are reachable and each of them has a pre-computed path.*

**Available actions without pathfinding-assistance:**

![no-athfinding](img/no-pathfinding.png?raw=true "Normal move actions")

*Previously, bots could only move players one square at the time*

During a Blitz action, the set of available actions will also include pathfinding-assisted move actions to opponent players, where it 
will optimize the number of block rolls as well as the risk of moving. Similarly, pathfinding-assisted move actions are given 
for handoff and foul actions.

Check out our new tutorial on how to use [pathfinding-assistance in reinforcement learning](a2c-pathfinding.md). 

## Forward Model
Previously, it was difficult to get a fast forward model up and running in botbowl due to the reliance on the slow ```copy.deepcopy()``` function. Thanks to amazing work by Mattias Bermell, botbowl now has a built-in forward model that is reasonably fast. At least much faster that what we had before!

It works by tracking changes to non-immutable properties in the game state. Such changes can then be reverted to go back in time, e.g. to reset the state, where we had to completely reinstantiate the entire game object before.

Check out our tutorials on [how to use the forward model](forward-model.md) and [how to implement a search-based bot](search-based.md).

# Participants

The following bots were submitted to Bot Bowl III.

## Dryad
This bot uses a forward model and monte carlo tree search with a simple heuristic reward function to select actions. This bot is still work in progress and a proof of concept. It currently shows promising results on small games with 3 to 5 players, but more work is needed to make it effective on the full game. 

## GrodBot
A scripted bot that won Bot Bowl I and became 2nd at Bot Bowl II.

## Grootbot
A deep reinforcement learning bot trained using the tutorials for the 1v1 variant of Blood Bowl. This bot was, unfourtunately, not able to compete in Bot Bowl III as it remains incompatable with the 11v11 varaint that is being used. 

## Goteboy
Goteboy has been trained in the [SEED RL framework](https://github.com/google-research/seed_rl) using the [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX) training algorithm with exception for the recurrent layer to reduce the memory requirments. Goteboy’s pre-training consists of a progressive curriculum learning similar to that of Gotebot that won the ML prize at Bot Bowl II. This bot is still work in progress but will hopefully return stronger next year.

## MimicBot
A deep-learning based actor-critic model with channel-wise attention that was trained by first imitating a scriptes bot and then improved with reinforcement-learning. Some scripted rules were added ontop of the neural network.

# Results

Congratulations to MimicBot by Nicola Pezzotti for winning Bot Bowl III and the Machine Learning prize!

The second place goes to GrodBot by Peter Moore, and the 3rd Place to Dryad by Jonas Busk.

![Bot Bowl III Results](img/bot-bowl-iii-results.png?raw=true "Bot Bowl III Results")

The replays can be downloaded [here](https://drive.google.com/file/d/1Oz7nrBRTCwHIiqJDYF7m6j9kvMdpdSDl/view?usp=sharing) and are compatible with [botbowl v0.3.1](https://github.com/njustesen/botbowl/releases/tag/v0.3.1).

