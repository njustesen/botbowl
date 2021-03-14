![Bot Bowl III](img/botbowl-iii.png?raw=true "Bot Bowl III")

Bot Bowl III the third AI competition in Blood Bowl and is set to happen (virtually) at the [IEEE Conference of Games 2021](https://ieee-cog.org/2021/index.html). 
Submit your bot by **July 15** to participate in the competition and have chance at winning one of the prestigious prizes.

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
Bot Bowl III is funded by [IEEE Computational Intelligence Society](https://cis.ieee.org/) and [modl.ai](www.modl.ai) and has a prize pool of $1,500.

**IEEE CIS Sponsored Prizes:**
- $500 for the winner
- $300 for the second place
- $200 for the third place

**modl.ai Sponsored Prize:**
- $500 for the most impressive machine learning bot.

The machine learning prize sponsored by modl.ai $500 will be given to the most impressive bot that relies on machine learning 
or search algorithms. A jury of AI Researchers from modl.ai will decide the winner of this prize based on the competition results, the implementation, and the submitted descriptions of the bots.

# New Features

## Pathfinding-assisted Move Actions
In previous Bot Bowls, bots could only move to adjacent squares, step by step. In Bot Bowl III, however, the FFAI framework 
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
In previous years, it was difficult to get a fast forward model up and running in FFAI due to the reliance on copy.deepcopy(). 
Thanks to amazing work by Mattias Bermell, FFAI now has a built-in forward model that is reasonably fast. 
It works by saving changes that happened to the game state and then it can revert those changes again instead of recreating the entire game object.

Check out our tutorials on [how to use the forward model](forward-model.md) and [how to implement a Flat Monte-Carlo search](a2c-pathfinding.md).

# Get Started
If you are completely new to bot development for Blood Bowl, there are plenty of detailed [tutorials](tutorials.md) to get started.
Please also join the [Fantasy Football AI Discord server](https://discord.gg/MTXMuae) for news, discussions, and assistance if you get stuck.

# Submit to Bot Bowl III
Submission details can be seen [here](submit.md).