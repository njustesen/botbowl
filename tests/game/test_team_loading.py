import pytest
from botbowl.core.game import *

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
    config = load_config("gym-11")
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename(race, ruleset)
    away = load_team_by_filename(race, ruleset)
    assert len(home.players) >= 11
    assert len(away.players) >= 11

