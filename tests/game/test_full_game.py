from botbowl.ai.registry import make_bot
from botbowl.core.game import *


def test_team():
    config = load_config("gym-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    away_agent = make_bot("random")
    home_agent = make_bot("random")
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()
    assert game.state.game_over
    assert game.end_time is not None
