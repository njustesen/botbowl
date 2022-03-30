![Bot Bowl IV](img/botbowl-iv.png?raw=true "Bot Bowl IV")

Bot Bowl IV is the fourth AI competition in Blood Bowl and will happen (virtually) at the [IEEE Conference of Games 2022](https://ieee-cog.org/2022/). 

# Format
Bot Bowl IV will have the same format as last year but with a few minor changes. 
The framework also offers a few new features and improvements that will empower bot developers. 
Group submissions are allowed but each person can only be part of one bot submission.

## Qualifiers
Each bot will play 10 matches of against the random baseline. 
Bots that are able ot outperform the baseline will proceed to the round-robin tournament.

## Round-robin
In the round-robing tournament, each bot will play 10 matches against each other and the tournament winner will be determined based on the following point system: 3 points for winning, 1 point for a draw, 0 points for a loss. 
In case of a tie, the bot with the highest score, and then inflicted casualties will win.

## Match Rules
Each match will use the following [configurations](../botbowl/data/config/bot-bowl.json):

- The BB2016 ruleset
- Time limit of 2 minutes per turn and 15 seconds per decision in the opponent's turn
- Pathfinding-assistance enabled
- Only a [fixed human team](../botbowl/data/teams/11/human.json) is available

If a bot crashes and is unable to recover, the game will continue and the bot will simply timeout and not do anything during their turn.

# Prizes
Will be announced soon!

# New Features

## Faster Pathfinding
The pathfinding module in the bot bowl framework has been improved significantly by compiling the python code to cython and through some other optimizations.
The algorithm is estimated to be 20x faster compared to last year!
This will make it easier for reinforcement learning bots to learn the game with pathfinding enabled.
Check out the tutorial [pathfinding-assistance in reinforcement learning](a2c-pathfinding.md) to learn more.

We would like to thank Mattias Bermell for his awesome work on pathfinding module!

## Forward Model
A forward model was introduced in Bot Bowl III. This year, the model is faster, has more features, and is more robust.
We hope to see a few search-based bots in Bot Bowl IV!

We would like to thank Mattias Bermell for his awesome work on the forward model!

Check out the tutorial [how to use the forward model](forward-model.md) and [how to implement a search-based bot](search-based.md) to learn more.

# Important Dates

*March 7th, 2022:* New version of the bot bowl framework [released](https://github.com/njustesen/botbowl/releases/tag/1.0.0) and on main.

*May 15th, 2022:* We aim to freeze the code base until the competition and will make _final_ release. If very critical bugs are found, we will only fix them after coordinating with everyone on the [Bot Bowl Discord server](https://discord.gg/MTXMuae).

*July 15th, 2022:* Submission Deadline: [How to submit](submit.md).

*August 21-24, 2022:* Results announced at the [IEEE Conference of Games 2022](https://ieee-cog.org/2022/).
