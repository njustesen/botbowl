import pytest
from ffai.ai.registry import register_bot, make_bot
from ffai.core.game import *
from ffai.ai.bots.random_bot import RandomBot

races = [
    "amazon",
    "chaos",
    "chaos-dwarf",
    "elf",
    "high-elf",
    "human",
    "lizardmen",
    "orc",
    "skaven"
]


@pytest.mark.parametrize("race", races)
def test_team(race):
    # Load configurations, rules, arena and teams
    config = get_config("ff-11")
    ruleset = get_rule_set(config.ruleset)
    home = get_team_by_filename(race, ruleset)
    away = get_team_by_filename(race, ruleset)
    away_agent = make_bot("random")
    home_agent = make_bot("random")
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()
    assert game.state.game_over
    assert game.end_time is not None
