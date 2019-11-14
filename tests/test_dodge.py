import pytest
from ffai.core.game import *
from unittest.mock import *


@patch("ffai.core.game.Game")
def test_default_dodge_modifier(mock_game):
    mock_game.num_tackle_zones_at.return_value = 0 # no tackle zones in target square

    role = Role("Lineman", "orc", 6,3,3,9, [], 50000, None)
    player = Player("1", role, "test", 1, "orc")

    pos = Square(11,12)
    # need to test static method to ensure calculation for lookahead requests
    modifier = Dodge.dodge_modifiers(mock_game, player, pos)

    assert modifier == 1


@patch("ffai.core.game.Game")
def test_prehensile_tail_modifier(mock_game):
    # no tackle zones in target square
    mock_game.num_tackle_zones_at.return_value = 0

    # set up dodging player
    role = Role("Lineman", "orc", 6,3,3,9, [], 50000, None)
    player = Player("1", role, "test", 1, "orc")

    # set up a detractors in square moving from
    tackle_zones = 0
    tacklers = []
    diving_tacklers = []
    shadowers = []
    tentaclers = []

    tail_player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.PREHENSILE_TAIL])
    prehensile_tailers = [tail_player]

    tz_details = TackleZoneDetails(tackle_zones, tacklers, prehensile_tailers, diving_tacklers, shadowers, tentaclers)
    mock_game.get_tackle_zones_detailed.return_value = tz_details

    pos = Square(11, 12)
    modifier = Dodge.dodge_modifiers(mock_game, player, pos)

    assert modifier == 0
