![Bot Bowl II](img/botbowl-ii.png?raw=true "Bot Bowl II")

Submit your bot before August 1st 2020 to participate in Bot Bowl II.
The results of  will be announced at the [Conference of Games (CoG) 2020 in Kyoto, Japan](http://ieee-cog.org/2020/) with a prize pool of $1000.
A submission link will appear on this page when we get closer to the deadline.

## Prizes

Bot Bowl II will have a prize pool of $1000.

- **1st:** $500
- **2nd:** $300
- **3rd:** $200

## Format
The format of Bot Bowl II will be similar to last year's competition.

Only a [prefixed Human team](https://github.com/njustesen/ffai/blob/master/ffai/data/teams/11/human.json) can be used. 
None of the pre and post-game procedures are used. Rosters are thus reset without injuries and improvement rolls.

Each bot will play 10 games against each other and in contrast to last year there will be no final. A bot 
that takes random actions will be added to the competition as a baseline.

## Rules
- Bots must be submitted no later than August 1st 2020
- Each participant can only be part of one bot submission
- Several participants can submit a bot together as a team
- Time limits used:
    - Turn: 120s
    - Opponent decisions: 5s
    - Initialization: 20s
    - Game End: 10s
    - Disqualification after delay of: 5s
- We use [this configuration file](https://github.com/njustesen/ffai/blob/master/ffai/data/config/bot-bowl-ii.json) for FFAI
- You are allowed to submit an extension of an existing bot that was developed by someone else, as long as it is a significant improvement. We reserve the right to disqualify 
bots that play similarly to the existing bot.

## Docker Containers
As a new initiative this year, bots will run in docker containers. Participants can submit their bot as a regular python 
package or optionally as a docker image. 

To facilitate docker containers, we are working on a new competition module that use the HTTP protocol to communicate with agents.
We aim to publish the module before June. It should, however, not influence your bot development as it will simply be a wrapper around the existing 
bot interface in FFAI.

Using docker has several advantages: 

1. It will be harder to cheat because the game and the agents run in separate processes. 
2. Participants can optionally setup their own docker images if they have particular requirements.
3. Any programming language can potentially be used. This will require quite a lot of work, so we don't expect to see bots developed in other languages than python.
4. Files can be written and read withing the docker container if needed. 

## Computational Resources
The philosophy of Bot Bowl is to limit the computational resources at run-time to that of a regular laptop. We of course encourage participants 
to use as much computation as they can afford to train/evolve their bots before the competition. 
The exact instance specification will be specified at a later date but don't expect any GPUs. 

## How to Submit
We will soon share templates for both a pure python submission and a docker submission. If your bot has requirements that are not 
installable through pip, you must make a docker submission.

## Questions
If you have questions about the competition, please join our [Discord server](https://discord.gg/MTXMuae).