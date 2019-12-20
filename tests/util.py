from ffai.core.load import *
from ffai.core.game import *
from ffai.ai.bots.random_bot import *


def get_game_turn(seed=0, empty=False):
    config = load_config("ff-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.KICK))
    game.step(Action(ActionType.SETUP_FORMATION_ZONE))
    game.step(Action(ActionType.END_SETUP))
    game.step(Action(ActionType.SETUP_FORMATION_WEDGE))
    game.step(Action(ActionType.END_SETUP))
    random_agent = RandomBot("home")
    while type(game.get_procedure()) is not Turn or game.is_quick_snap() or game.is_blitz():
        action = random_agent.act(game)
        game.step(action)
    if empty:
        game.clear_board()
    return game


def get_game_kickoff(seed=0):
    config = load_config("ff-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.KICK))
    game.step(Action(ActionType.SETUP_FORMATION_ZONE))
    game.step(Action(ActionType.END_SETUP))
    game.step(Action(ActionType.SETUP_FORMATION_WEDGE))
    game.step(Action(ActionType.END_SETUP))
    return game


def get_game_setup(home_team, seed=0):
    config = load_config("ff-11")
    config.kick_off_table = False
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    if game.actor == game.home_agent:
        if home_team:
            game.step(Action(ActionType.KICK))
        else:
            game.step(Action(ActionType.RECEIVE))
    if game.actor == game.away_agent:
        if home_team:
            game.step(Action(ActionType.RECEIVE))
        else:
            game.step(Action(ActionType.KICK))
    return game


def get_game_coin_toss(seed=0):
    config = load_config("ff-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    return game


def get_game_fans(seed=0):
    config = load_config("ff-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    return game


def get_game_weather_table(seed=0):
    config = load_config("ff-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("orc", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    #game.init()
    return game


def get_block_players(game, team):
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    attacker = None
    defender = None
    for p in players:
        if p.team != team:
            continue
        attacker = p
        adjacent = game.get_adjacent_opponents(attacker)
        if len(adjacent) > 0:
            defender = adjacent[0]
            break
    return attacker, defender
