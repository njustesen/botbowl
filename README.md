# FFAI: Fantasy Football AI
FFAI is a python package that includes a framework for playing and developing bots for digital fantasy-football board games.
For an in-depth description of the framework, challenge of applying AI to fantasy football, and some preliminaray results see our paper: [Blood Bowl: The Next Board Game Challenge for AI](https://njustesen.github.io/njustesen/publications/justesen2018blood.pdf).

![FFAI](docs/img/ffai.png?raw=true "FFAI")

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

## Bot Bowl
Bot Bowl is an AI competition using the FFAI framework. Go read all about the upcoming [Bot Bowl III](docs/bot-bowl-iii.md).

## Installation
[Make sure python 3.6 or newer is installed, together with pip.](https://www.makeuseof.com/tag/install-pip-for-python/)
Then run:
```
pip install git+https://github.com/njustesen/ffai
```
Here's a more [detailed guide](docs/installation.md) on how to set up FFAI.

## Tutorials
Head over to our [tutorials](docs/tutorials.md) to learn about how to use FFAI.

## Run FFAI's Web Server
```
python -c "import ffai.web.server as server;server.start_server(debug=True, use_reloader=False, port=5000)"
```
Or download the examples folder and run:
```
python examples/server_example.py
```
Go to: http://127.0.0.1:1234/

The main page lists active games. For each active game you can click on a team to play it. If a team is disabled it is controlled by a bot and cannot be selected. Click hot-seat to play human vs. human on the same machine.

## Disclaimers and Copyrighted Art
FFAI is not affiliated with or endorsed by any company and/or trademark. FFAI is an open research framework and the authors have no commercial interests in this project. The web interface in FFAI currently uses a small set of icons from the Fantasy Football Client. These icons are not included in the license of FFAI. If you are the author of these icons and don't want us to use them in this project, please contact us at njustesen at gmail dot com, and we will replace them ASAP. The team icons are from FUMBBL and are used with permission. The license described in [LICENSE](LICENSE) only covers the source code - not any of the graphics files.

## Get Involved
Do you want implement a bot for FFAI or perhaps help us test, develop, and/or organize AI competitions? Join the [FFAI Discord server](https://discord.gg/MTXMuae).
