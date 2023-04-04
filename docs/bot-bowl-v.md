![Bot Bowl V](img/bot-bowl-v.png?raw=true "Bot Bowl V")

Bot Bowl V is the fifth AI competition in Blood Bowl. The results will be announced at the [IEEE Conference of Games 2023](https://2023.ieee-cog.org/). 

# Submission Deadline

The submission deadline for Bot Bowl V is August 1st anywhere on earth.
In the first week after the submission deadline (August 1st - 8th), authors will receive an email with results of 10 matches played against the random baseline. 
This is to confirm that we were able to set up and run the bot correctly. 
If we run into issues running you bot, please be prepared to assist us in this week.

# Sign up already to receive news and updates!

TBA

# Format
Bot Bowl V has the same format as the previous years but we aim to have a few new tutorials ready for you. 
Group submissions are allowed but each person can only be part of one bot submission.
Each group must have a team leader that will receive any prize money they win and the lead will be responsible of sharing it with the group.

## Round-robin
In a round-robing tournament, each bot plays 10 matches against each other and the tournament winner will be determined based on the following point system: 3 points for winning, 1 point for a draw, 0 points for a loss. 
In case of a tie, the bot with the highest score, and then inflicted casualties will win.

## Match Rules
Each match uses the following [configurations](https://github.com/njustesen/botbowl/blob/main/botbowl/data/config/bot-bowl.json):

- The BB2016 ruleset
- Time limit of 2 minutes per turn and 15 seconds per decision in the opponent's turn
- Pathfinding-assistance enabled
- Only a [fixed human team](https://github.com/njustesen/botbowl/blob/main/botbowl/data/teams/11/human.json) is available

Bots will run for entire 10-match sequence against it's opponent. This allows it to adapt without having to save anything to the file system in-between matches.
However, if a bot crashes and is unable to recover, the matche sequence will continue and the bot will simply timeout and not do anything during their turns.

# Prizes

Bot Bowl V had a prize pool of $1000 with the following two prizes:

- $500 is awarded to the 1st place winner
- $500 is awarded to the author(s) of the most innovative machine learning bot. This prize is sponsored by [modl.ai](modl.ai) and the winner will be selected by a jury consisting of Julian Togelius and Sebastian Risi.

# New Features

## Fast Forward Model in Rust

TBA

