![Bot Bowl III](img/botbowl-iii.png?raw=true "Bot Bowl III")


Bot Bowl III the third AI competition in Blood Bowl and is set to happen (virtually) at the IEEE Conference of Games 2021. 
Submit your bot by July 15 to participate in the competition and have chance at winning one of the prestigious prizes.

# Format
Bot Bowl III will have the same format as last year while the framework offers a few new features that will change the 
competition slightly. 

Each bot will play 10 matches of Blood Bowl against each other and the winner will determined based on the following point system:
3 points for winning, 1 point for a draw, 0 points for a loss. If the bot crashes and are unable to recover, the game will 
continue and the bot will simply timeout and not do anything during their turn.

Each match will using the following configurations:

- The BB2016 ruleset
- Only a fixed human teams
- Time limit of 2 minutes per turn and 15 seconds per decision in the opponent's turn
- Pathfinding-assisted move actions are enabled (read more about this in the New Features section below)

# Prizes
The winner of Bot Bowl III will receive $500, sponsored by IEEE.
Additionally, a machine learning prize of $500 sponsered by modl.ai will be given to the highest ranked bot that relies mainly on machine learning 
or search algorithms. If in doubt whether your bot would qualify for this prize, feel free to ask questions in the Discord channel.

# New Features

## Pathfinder-assisted Move Actions
In previous Bot Bowls, you could only move to adjacent squares, step by step. In Bot Bowl III, however, the FFAI framework 
will provide bots with available move actions to all reachable squares together with the safest and shortest path to reach it.
Bots are then able to perform move actions to squares that are further away than one step, and the framework will perform all 
the intermediate steps automatically. 

During a Blitz action, the set of available actions will also include pathfinder-assisted move actions to opponent players, where it 
will optimize the number of block rolls as well as the risk of moving. Similarly, pathfinder-assisted move actions are given 
for handoff and foul actions.


## Forward Model


# Get Started

Tutorials
Discord
