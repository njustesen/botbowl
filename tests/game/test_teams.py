import pytest
from ffai.ai.registry import register_bot, make_bot
from ffai.core.game import *
from ffai.ai.bots.random_bot import RandomBot

races = [
    "amazon",
    "chaos",
    "chaos-dwarf",
    "elven-union",
    "high-elf",
    "human",
    "lizardmen",
    "orc",
    "skaven"
]


@pytest.mark.parametrize("race", races)
def test_team(race):
    config = load_config("ff-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename(race, ruleset)
    away = load_team_by_filename(race, ruleset)
    away_agent = make_bot("random")
    home_agent = make_bot("random")
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()
    assert game.state.game_over
    assert game.end_time is not None
