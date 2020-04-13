"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains functions to communicate with a game host to manage games.
"""

from ffai.web.host import *
from ffai.core.game import *
from ffai.core.load import *
from ffai.ai.registry import list_bots

# Create a game in-memory host
host = InMemoryHost()


def new_game(away_team_name, home_team_name, away_agent=None, home_agent=None, config_name="web.json", board_size=11):
    assert away_agent is not None
    assert home_agent is not None
    config = load_config(config_name)
    # config.competition_mode = True
    ruleset = load_rule_set(config.ruleset, all_rules=False)
    home = load_team_by_name(home_team_name, ruleset, board_size=board_size)
    away = load_team_by_name(away_team_name, ruleset, board_size=board_size)
    game_id = str(uuid.uuid1())
    game = Game(game_id, home, away, home_agent, away_agent, config)
    game.init()
    host.add_game(game)
    print("Game created with id ", game.game_id)
    return game


def step(game_id, action):
    game = host.get_game(game_id)
    game.step(action)
    return game


def save_game_exists(name):
    for save in host.get_saved_games():
        if save[1] == name.lower():
            return True
    return False


def save_game(game_id, name, team_id):
    name = name.replace("/", "").replace(".", "").lower()
    host.save_game(game_id, name, team_id)


def get_game(game_id):
    game = host.get_game(game_id)
    if game is not None and game.actor is not None and game.actor.human:
        game.refresh()
    return game


def get_replay(replay_id):
    return Replay(replay_id=replay_id, load=True)


def get_replay_ids():
    return host.get_replay_ids()


def load_game(name):
    return host.load_game(name)


def get_games():
    return host.get_games()


def get_saved_games():
    return host.get_saved_games()


def get_teams(ruleset, board_size=11):
    return load_all_teams(ruleset, board_size=board_size)


def get_bots():
    bots = list_bots()
    return bots