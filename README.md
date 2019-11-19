# FFAI: Fantasy Football AI
FFAI is a python package that includes a framework for playing and developing bots for digital fantasy-football board-games.
For an in-depth description of the framework, challenge of applying AI to fantasy football, and some preliminaray results see our paper: [A New Board Game Challenge and Competition for AI](https://njustesen.files.wordpress.com/2019/07/justesen2019blood.pdf).

![FFAI](screenshots/ffai.png?raw=true "FFAI")

Please cite our paper if you use FFAI in your publications.
```
@inproceedings{justesen2019blood,
  title={Blood Bowl: A New Board Game Challenge and Competition for AI},
  author={Justesen, Niels and Moore, Peter David and Uth, Lasse M{\o}ller and Togelius, Julian and Jakobsen, Christopher and Risi, Sebastian}
  booktitle={2019 IEEE Conference on Games (COG)},
  year={2019},
  organization={IEEE}
}
```

## Features
* Rule implementation of the Living Rulebook 5 and BB2016 with the following limitations:
  * Only skills for the Human and Orc teams have been implemented and tested
  * No big guys
  * No league or tournament play
  * No star player points or level up
  * No inducements
* A web interface supporting:
  * Hot-seat
  * Online play
  * Spectators
  * Human vs. bot
* An AI interface that allows you to implement and test your own bots
  * OpenAI Gym interface
  * Path finding utilities
  * Examples to get started
* Implementation of the Open AI Gym interface, that allows you to train machine learning algorithms
* Custom pitches (we call these _arenas_). FFAI comes with arenas of four different sizes.
* Rule configurations are possible from a configuration file, including:
  * Arena (which .txt file to load describing the arena)
  * Ruleset (which .xml file to load containing rules for rosters etc.)
  * Setup restrictions
  * Number of turns
  * Kick-off table enabled/disabled
  * Which scatter dice to use
  * Time management
  * ...
* Premade formations to ease the setup phase. Custom made formations can easily be implemented.
* Games can be saved and loaded

## Plans for Future Releases
* Support for dungeon arenas
* Support for all skills and teams in LRB6
* League mode
* Integration with OBBLM

## Installation
[Make sure python 3.6 or newer is installed, together with pip.](https://www.makeuseof.com/tag/install-pip-for-python/)
```
pip install git+https://github.com/njustesen/ffai
```

## Run FFAI's Web Server
```
python -c "import ffai.web.server as server;server.start_server(debug=True, use_reloader=False, port=5000)"
```
Or download the examples folder and run:
```
python examples/server_example.py
```
Go to: http://127.0.0.1:5000/

The main page lists active games. For each active game you can click on a team to play it. If a team is disabled it is controlled by a bot and cannot be selected. Click hot-seat to play human vs. human on the same machine.

## Disclaminers and Copyrighted Art
FFAI is not affiliated with or endorsed by any company and/or trademark. FFAI is an open research framework and the authors have no commercial interests in this project. The web interface in FFAI currently uses a small set of icons from the Fantasy Football Client. These icons are not included in the license of FFAI. If you are the author of these icons and don't want us to use them in this project, please contact us at njustesen at gmail dot com, and we will replace them ASAP. The team icons are from FUMBBL are used. The license described in [LICENSE](LICENSE) only covers the source code - not any of the graphics files.

## Get Involved
Do you want implement a bot for FFAI or perhaps help us test, develop, and/or organize AI competitions? Join our Discord server using this link: [FFAI Discord Server](https://discord.gg/MTXMuae).
