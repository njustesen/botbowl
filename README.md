# botbowl
botbowl is a python package that includes a framework for playing and developing bots for Blood Bowl. Our aim is to develop a bot that can beat the best humans at the game. 

![botbowl](docs/img/botbowl_webUI.png?raw=true "botbowl")

Please cite our paper if you use botbowl in your publications.
```
@inproceedings{justesen2019blood,
  title={Blood Bowl: A New Board Game Challenge and Competition for AI},
  author={Justesen, Niels and Moore, Peter David and Uth, Lasse M{\o}ller and Togelius, Julian and Jakobsen, Christopher and Risi, Sebastian}
  booktitle={2019 IEEE Conference on Games (COG)},
  year={2019},
  organization={IEEE}
}
```

Read more about the challenges and achievements in [Papers.md](docs/papers.md)

## Installation
[Make sure python 3.6 or newer is installed, together with pip.](https://www.makeuseof.com/tag/install-pip-for-python/)
Then run:
```
pip install git+https://github.com/njustesen/botbowl
```
If python detects a C++ compiler in your system it compile the pathfinding algorithm which makes it faster. If you used `git clone` you have to manually compile. Here's a more [detailed guide](docs/installation.md) on how to set up the Bot Bowl framework.

## Tutorials
Head over to our [tutorials](docs/tutorials.md) to learn about how to use the Bot Bowl framework.

### Run botbowl's Web Server
```
python -c "import botbowl.web.server as server;server.start_server(debug=True, use_reloader=False, port=5000)"
```
Or download the examples folder and run:
```
python examples/server_example.py
```
Go to: http://127.0.0.1:1234/

The main page lists active games. For each active game you can click on a team to play it. If a team is disabled it is controlled by a bot and cannot be selected. Click hot-seat to play human vs. human on the same machine.

## Bot Bowl - the compitition
Bot Bowl is an AI competition using the Bot Bowl framework. Go read all about the results of [Bot Bowl III](docs/bot-bowl-iii.md). Not to be confused with 'botbowl' which refers to the module/repo. 

## Get involved 
We'd love you to join us in creating really good bots for Blood Bowl! There are mainly two ways. First is developing your own bot to compete in the competition. Head over to the [tutorials](docs/tutorials.md) to get started creating your own awesome bot. The second way is contributing to the project itself with development. Head over to [Development.md](docs/development.md) to learn more.  

Join the [Bot Bowl Discord server](https://discord.gg/MTXMuae) for questions, discussions and the latest news! 

## Disclaimers and Copyrighted Art
Bot Bowl is not affiliated with or endorsed by any company and/or trademark. Bot Bowl is an open research framework and the authors have no commercial interests in this project. The web interface in Bot Bowl currently uses a small set of icons from the Fantasy Football Client. These icons are not included in the license of Bot Bowl. If you are the author of these icons and don't want us to use them in this project, please contact us at njustesen at gmail dot com, and we will replace them ASAP. The team icons are from FUMBBL and are used with permission. The license described in [LICENSE](LICENSE) only covers the source code - not any of the graphics files.
