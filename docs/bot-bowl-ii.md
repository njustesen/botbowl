![Bot Bowl II](img/botbowl-ii.png?raw=true "Bot Bowl II")

**See the results** further down on this page.

Bot Bowl II is an AI competition in Blood Bowl using the botbowl framework. The traditional board size of 26×15 squares with 11 players on each side is used. Participants are, however, limited to use a prefixed human team. In future competitions, we plan to allow all teams and the option to customized rosters.

Bot Bowl II had six bot submissions and we added the Random agent baseline to the competition. The competition consisted of a round-robin tournament where all bots played every other bot 10 times. The winner is the one with most wins, then draws, then TDs, then CAS.

Bot Bowl II had two prizes:

1. $500 to the 1st place winner.
2. $500 to the highest ranked machine learning bot*.

* To qualify as a machine learning bot, the core part of the decision-making must be learned. A machine learning bot can use search and even some scripted behaviors, as long as the main decision-making of the bot is learned. 

Participants **CAN** win both prizes, if they win the competition with a machine learning bot. 

## Format
The format of Bot Bowl II is similar to last year's competition. Only a [prefixed Human team](https://github.com/njustesen/botbowl/blob/master/botbowl/data/teams/11/human.json) can be used. 
None of the pre and post-game procedures are used. Rosters are thus reset without injuries and improvement rolls.

In contrast to last year there will be no final. 

## Rules
- Bots must be submitted no later than August 1st 2020
- Each participant can only be part of one bot submission
- Several participants can submit a bot together as a team
- Time limits used:
    - Turn: 120s
    - Opponent decisions: 5s
    - Initialization: 20s
    - Game End: 10s
- We use [this configuration file](https://github.com/njustesen/botbowl/blob/master/botbowl/data/config/bot-bowl-ii.json) for botbowl
- You are allowed to submit an extension of an existing bot that was developed by someone else, as long as it is a significant improvement. We reserve the right to disqualify 
bots that play similarly to the existing bot.

## Computational Resources
The philosophy of Bot Bowl is to limit the computational resources at run-time to that of a regular laptop. We of course encourage participants to use as much computation as they can afford to train/evolve their bots before the competition. Don't expect any GPUs. 

## Participants

### Giaobot by Zhang Pengpeng, Gao Huiru, Xu Zhiwei
Giaobot is a machine learning bot trained with IMPALA using an LSTM policy, and self-play.

### Grodbot MKII by Peter Moore
Grodbot MKII is an improved version of last year's winner, including bug fixes, less screening and more emphasis on chasing the ball.

### Gotebot by Mattias Bermell
Gotebot is a machine learning bot trained using A2C, self-play, and curriculum learning. Ontop of the learned behavior, it has to simple scripted rules: Never do GFI and never end the turn with unused players left.

### Minigrod by RogueLichen
Minigrod is a scripted bot based on Grodbot (the winner of Bot Bowl I) it applies dynamic formations and more focus on safe actions.

### Sapling by Jonas Busk
Sapling is a search-based bot that uses Monte Carlo Tree Search with a scripted heuristic function.

### Scripted Plus Bot by Mark Christiansen
Scripted Plus Bot is a scripted bot based on the one from the [tutorials](https://njustesen.github.io/botbowl/tutorials). It has dynamic formations, prioritizes blitzing players near the ball, and has logic for crowd surfing (pushing players of the board).

## Results

|                       | Minigrod | GB MKII  | Scripted PB | Gotebot | Sapling   | Random | Giaobot | W/D/L    | TD      | CAS     |
|-----------------------|:--------:|:--------:|:-----------:|:-------:|:---------:|:------:|:-------:|:--------:|:-------:|:-------:|
| Minigrod              | -        | 2/5/3    | 8/2/0       | 7/3/0   | 10/0/0    | 10/0/0 | 10/0/0  | 47/10/3  | 151/12  | 121/48  | 
| GrodBot<br/>MKII      | 3/5/2    | -        | 4/2/4       | 10/0/0  | 10/0/0    | 10/0/0 | 10/0/0  | 47/7/6   | 144/13  | 70/65   |
| Scripted<br/>Plus Bot | 0/2/8    | 4/2/4    | -           | 8/2/0   | 10/0/0    | 10/0/0 | 9/1/0   | 41/7/12  | 97/25   | 93/50   |
| Gotebot               | 0/3/7    | 0/0/10   | 0/2/8       | -       | 9/1/0     | 7/3/0  | 10/0/0  | 26/9/25  | 49/45   | 56/61   |
| Sapling               | 0/0/10   | 0/0/10   | 0/0/10      | 0/1/9   | -         | 0/10/0 | 1/9/0   | 1/20/39  | 1/104   | 38/65   |
| _Random_              | 0/0/10   | 0/0/10   | 0/0/10      | 0/3/7   | 0/10/0    | -      | 0/10/0  | 0/23/37  | 0/116   | 18/48   |
| Giaobot               | 0/0/10   | 0/0/10   | 0/1/9       | 0/0/10  | 0/9/1     | 0/10/0 | -       | 0/20/40  | 0/127   | 12/71   |

## Winners
Congratulations to Minigrod (RogueLichen) for winning Bot Bowl II and to Gotebot for winning the ML prize. The authors of both bots will each recieve $500.

## Highlights
Watch the highlight video here: [https://www.youtube.com/watch?v=qajXQhrBuV0](https://www.youtube.com/watch?v=qajXQhrBuV0).
