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

## Future Plans
* AI tournament module
* Dungeon arenas rules
* Support for all skills and teams in LRB6
* League mode
* Integration with OBBLM

## Installation


## Create a Bot


## Gym Interface


### Setup


### Run


### Custom Feature layers


### 
