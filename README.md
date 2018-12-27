# FFAI: Fantasy Football AI
A highly-extensible modular multi-purpose AI framework for digital fantasy-football board-games.
FFAI is still under development and will be updated heavily the next six months.

## Features
* Rule implementation of the Living Rulebook 5 with the following limitations:
  * Only skills for the Human and Orc teams have been implemented and tested
  * No big guys
  * No league or tournament play
  * No star player points or level up
  * No inducements
  * No timers; Players have unlimited time each turn
  * Only premade teams
* A web interface supporting:
  * Hot-seat 
  * Online play
  * Spectators
  * Human vs. bot
* An AI interface, that allows you to implement and test your own bots
* Implementation of the Open AI Gym interface, that allows you to train machine learning algorithms
* Custom pitches (we call these _arenas_). FFAI comes with arenas of four different sizes.
* Rule configurations are possible from a configuration file, including:
  * Arena (which .txt file to load describing the arena)
  * Ruleset (which .xml file to load containing rules for rosters etc.)
  * Setup restrictions
  * Number of turns
  * Kick-off table enabled/disabled
  * Which scatter dice to use
  * ...
* Premade formations to ease the setup phase. Custom made fomrations can easily be implemented. 
* Games can be saved and loaded

## Plans for Future Releases
* More documentation
* AI tournament module
* Dungeon arenas rules
* Support for all skills and teams in LRB6
* League mode
* Integration with OBBLM

## Installation
Clone the repository and make sure python 3.6 or newer is installed, together with pip.
Go to the cloned directory and run the following to install the dependencies: 
```
pip install -e .
```
Or
```
pip3 install -e .
```
Depending on your setup.

## Run FFAI's Web Server
```
python ffai/web/server.py
```
Go to: http://127.0.0.1:5000/

The home page lists active games. For each active game, you can click on a team to play it. If a team is disabled it is a bot and cannot be selected. Click hot-seat to play human vs. human on the same machine.

## Create a Bot
To make you own bot you must implement the Agent class and it's three methods: new_game, act, and end_game which are called by the framework. The act method must return an instance of the Action class. 

Take a look at our example here.

Be aware, that you should not modify instances that comes from the framework such as Square instances as these are shared with the GameState instance in FFAI. In future releases, we plan to release an AI tournament module that will clone the instances before they are handed to the bots.

## Gym Interface
FFAI implements the Open AI Gym interace for easy integration of machine learning algorithms. 

### Observations
Observations are split in two parts, one that consists of (spatial) two-dimensional feature leayers and a non-spatial vector of normalized values (e.g. turn number, half, scores etc.).



### Action Types
Actions consists of 44 action types and an optional x and y coordinate.


Take a look at our example here.

### Setup


### Run


### Disclaminers and Copyrighted Art

### Custom Feature layers


### 
