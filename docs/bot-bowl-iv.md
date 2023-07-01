![Bot Bowl IV](img/botbowl-iv.png?raw=true "Bot Bowl IV")

Bot Bowl IV is the fourth AI competition in Blood Bowl and happened (virtually) at the [IEEE Conference of Games 2022](https://ieee-cog.org/2022/). 

# Format
Bot Bowl IV had the same format as the previous year but with a few minor changes. 
The framework also offered a few new features and improvements that will empower bot developers. 
Group submissions were allowed but each person could only be part of one bot submission.
Each group must have a team leader that will receive any prize money they win and the lead will be responsible of sharing it with the group.

## Round-robin
In a round-robing tournament, each bot plays 10 matches against each other and the tournament winner will be determined based on the following point system: 3 points for winning, 1 point for a draw, 0 points for a loss. 
In case of a tie, the bot with the highest score, and then inflicted casualties would win.

## Match Rules
Each match used the following [configurations](https://github.com/njustesen/botbowl/blob/main/botbowl/data/config/bot-bowl.json):

- The BB2016 ruleset
- Time limit of 2 minutes per turn and 15 seconds per decision in the opponent's turn
- Pathfinding-assistance enabled
- Only a [fixed human team](https://github.com/njustesen/botbowl/blob/main/botbowl/data/teams/11/human.json) is available

If a bot crashes and is unable to recover, the game will continue and the bot will simply timeout and not do anything during their turn.

# Prizes

Bot Bowl IV had a prize pool of $1000 with the following two prizes:

- $500 is awarded to the 1st place winner
- $500 is awarded to the author(s) of the most innovative machine learning bot. This prize is sponsored by [modl.ai](modl.ai) and the winner will be selected by a jury consisting of Julian Togelius and Sebastian Risi.

# New Features

## Faster Pathfinding
The pathfinding module in the bot bowl framework was improved significantly by compiling the python code to cython and through some other optimizations.
The algorithm is estimated to be 20x faster compared to the previous year!
This makes it easier for reinforcement learning bots to learn the game with pathfinding enabled.
Check out the tutorial [pathfinding-assistance in reinforcement learning](a2c-pathfinding.md) to learn more.

## Forward Model
A forward model was introduced in Bot Bowl III. This year, the model is faster, has more features, and is more robust.
We were thus happy to see a few MCTS bots this year.

Check out the tutorial [how to use the forward model](forward-model.md) and [how to implement a search-based bot](search-based.md) to learn more.

# Results

Check out the presentation of the results on [YouTube](https://www.youtube.com/watch?v=iizeI546zmI).

![Bot Bowl IV Results](img/bot-bowl-iv-results.png?raw=true "Bot Bowl IV Results")

Congratulations to Drefsante by Frederic Bair for winning Bot Bowl IV and MotBot by Niaamo for winning the ML Prize!

The replays and detailed results can be downloaded [here](https://drive.google.com/drive/folders/1q6AMkfrbN7wEFroaFn7Mpys2U0eirhjY?usp=sharing) and are compatible with [botbowl v1.0.2](https://github.com/njustesen/botbowl/releases/tag/1.0.2).
