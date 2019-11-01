import pytest
from ffai.core.game import *


def get_game(seed=0):
    config = get_config("ff-11")
    ruleset = get_rule_set(config.ruleset)
    home = get_team_by_filename("human", ruleset)
    away = get_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.HEADS))
    game.step(Action(ActionType.KICK))
    game.step(Action(ActionType.SETUP_FORMATION_ZONE))
    game.step(Action(ActionType.SETUP_FORMATION_WEDGE))
    return game


#def test_blitz():
#    D6.fix_roll(5)
#    D6.fix_roll(5)


#if __name__ == "__main__":
#    test_blitz()