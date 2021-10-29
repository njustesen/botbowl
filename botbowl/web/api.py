"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains functions to communicate with a game host to manage games.
"""

from botbowl.web.host import *
from botbowl.core.game import *
from botbowl.core.load import *
from botbowl.ai.registry import list_bots
from copy import deepcopy

# Create a game in-memory host
host = InMemoryHost()
replay_cache = {}
step_cache = {}


def new_game(away_team_name, home_team_name, away_agent=None, home_agent=None, config_name="web.json", board_size=11):
    assert away_agent is not None
    assert home_agent is not None
    config = load_config(config_name)
    #config.competition_mode = True
    ruleset = load_rule_set(config.ruleset, all_rules=False)
    home = load_team_by_name(home_team_name, ruleset, board_size=board_size)
    away = load_team_by_name(away_team_name, ruleset, board_size=board_size)
    game_id = str(uuid.uuid1())
    game = Game(game_id, home, away, home_agent, away_agent, config, record=False)
    game.init()
    host.add_game(game)
    print("Game created with id ", game.game_id)
    return game


def step(game_id, action):
    game = host.get_game(game_id)
    if not game.is_started():
        action_choice = game.get_available_actions()[0]
        agent = game.get_team_agent(action_choice.team)
        other_agent = game.get_other_agent(agent)
        if not agent.human and other_agent.human:
            a = Action(ActionType.START_GAME)
            game.step(a)
            return game
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
    if replay_id in replay_cache:
        replay = replay_cache[replay_id]
    else:
        replay = Replay(replay_id=replay_id, load=True)
        steps = {}
        for idx, step in replay.steps.items():
            steps[idx] = step
        step_cache[replay_id] = steps
        replay.steps = {}
        replay_cache[replay_id] = replay
    steps = {}
    for i, step in step_cache[replay_id].items():
        if len(steps) >= 100:
            break
        steps[i] = step
    replay.steps = steps
    return replay


def get_replay_steps(replay_id, from_idx, num_steps):
    if replay_id not in replay_cache:
        return {}
    steps = {}
    c = 0
    for i, step in step_cache[replay_id].items():
        if c < from_idx:
            c += 1
            continue
        if len(steps) >= num_steps:
            break
        steps[i] = step
        c += 1
    return steps


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