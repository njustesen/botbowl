![Bot Bowl V](img/bot-bowl-v.png?raw=true "Bot Bowl V")

Bot Bowl V is the fifth AI competition in Blood Bowl and the results was announced at the [IEEE Conference of Games 2023](https://2023.ieee-cog.org/). 

# Format
Bot Bowl V had the same format as the previous years. 
Group submissions were allowed but each person can only be part of one bot submission.

## Round-robin
In a round-robin tournament, each bot played 10 matches against each other and the tournament winner was determined based on the following point system: 3 points for winning, 1 point for a draw, 0 points for a loss. 
In case of a tie, the bot with the highest score difference, and then inflicted casualties would win.

## Match Rules
Each match used the following [configurations](https://github.com/njustesen/botbowl/blob/main/botbowl/data/config/bot-bowl.json):

- The BB2016 ruleset
- Time limit of 2 minutes per turn and 15 seconds per decision in the opponent's turn
- Pathfinding-assistance enabled
- Only a [fixed human team](https://github.com/njustesen/botbowl/blob/main/botbowl/data/teams/11/human.json) is available

# Prizes

We had the following two prizes:

- $250 1st Place Prize sponsored by modl.ai.
- $250 Machine Learning Prize sponsored by modl.ai for the most interesting bot that uses machine learning and/or search-based techniques.

# Results

Watch the presentation of the results and highlights on [YouTube](https://www.youtube.com/embed/ZWTwlQS3_So?si=T84qEagOSISH3ur8).

<img width="1313" alt="Screenshot 2023-08-25 at 08 32 15" src="https://github.com/njustesen/botbowl/assets/1433421/09ed0a4f-6ac6-49ca-9adf-fb67d8781cc3">

Congratulations to Drefsante by Frederic Bair for winning Bot Bowl V and Treeman by Jonas Busk for winning the ML Prize!

The [replays](https://drive.google.com/drive/folders/1VoJ6wJbDgj0p4L3KOZRVggqTMtRGQbL1?usp=sharing) and per-match [results](https://drive.google.com/drive/folders/173OAty0tduxtDKKqf18Pf_FiW2ciBs7u?usp=sharing) are compatible with [botbowl v1.1.0](https://github.com/njustesen/botbowl/releases/tag/1.1.0).
