import pytest
from botbowl.core.load import *
from botbowl.core.game import Game


def test_forced_action():
    config = load_config("gym-11")
    config.time_limits.turn = 0.01
    config.time_limits.secondary = 0.01
    config.competition_mode = True
    config.fast_mode = True
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()
    game.step(Action(ActionType.START_GAME))
    for i in range(100):
        time.sleep(config.time_limits.turn * 1.5)
        game.refresh()
    assert game.state.game_over


def test_forced_setup():
    config = load_config("gym-11")
    config.time_limits.turn = 0.1
    config.competition_mode = True
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.KICK))
    time.sleep(config.time_limits.turn * 1.5)
    game.refresh()
    assert len(game.get_players_on_pitch()) == 11
    time.sleep(config.time_limits.turn * 1.5)
    game.refresh()
    assert len(game.get_players_on_pitch()) == 22
