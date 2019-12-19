from tests.util import *
import pytest


@pytest.mark.parametrize("data", [
    ((1, 0), PassDistance.QUICK_PASS),
    ((3, 0), PassDistance.QUICK_PASS),
    ((4, 0), PassDistance.SHORT_PASS),
    ((6, 0), PassDistance.SHORT_PASS),
    ((7, 0), PassDistance.LONG_PASS),
    ((10, 0), PassDistance.LONG_PASS),
    ((11, 0), PassDistance.LONG_BOMB),
    ((13, 0), PassDistance.LONG_BOMB),
    ((14, 0), PassDistance.HAIL_MARY),
    ((1, 1), PassDistance.QUICK_PASS),
    ((2, 2), PassDistance.QUICK_PASS),
    ((3, 3), PassDistance.SHORT_PASS),
    ((4, 4), PassDistance.SHORT_PASS),
    ((5, 5), PassDistance.LONG_PASS),
    ((7, 7), PassDistance.LONG_PASS),
    ((8, 8), PassDistance.LONG_BOMB),
    ((9, 9), PassDistance.LONG_BOMB),
    ((10, 10), PassDistance.HAIL_MARY)
])
def test_pass_distances_and_modifiers(data):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    offset = data[0]
    asserted_pass_range = data[1]
    passer = team.players[0]
    passer.role.skills = []
    catcher = team.players[1]
    game.put(passer, Square(1, 1))
    game.state.weather = WeatherType.NICE
    catcher_position = Square(passer.position.x + offset[0], passer.position.y + offset[1])
    game.put(catcher, catcher_position)
    pass_distance = game.get_pass_distance(passer.position, catcher_position)
    assert pass_distance == asserted_pass_range
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    if pass_distance == PassDistance.QUICK_PASS:
        assert pass_mods == 1
    elif pass_distance == PassDistance.SHORT_PASS:
        assert pass_mods == 0
    elif pass_distance == PassDistance.LONG_PASS:
        assert pass_mods == -1
    elif pass_distance == PassDistance.LONG_BOMB:
        assert pass_mods == -2
    # Weather
    game.state.weather = WeatherType.VERY_SUNNY
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    if pass_distance == PassDistance.QUICK_PASS:
        assert pass_mods == 1 - 1
    elif pass_distance == PassDistance.SHORT_PASS:
        assert pass_mods == 0 - 1
    elif pass_distance == PassDistance.LONG_PASS:
        assert pass_mods == -1 - 1
    elif pass_distance == PassDistance.LONG_BOMB:
        assert pass_mods == -2 - 1
    game.state.weather = WeatherType.NICE
    # Tackle zone modifier
    opp_player = game.get_opp_team(team).players[0]
    game.put(opp_player, Square(1, 2))
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    if pass_distance == PassDistance.QUICK_PASS:
        assert pass_mods == 0
    elif pass_distance == PassDistance.SHORT_PASS:
        assert pass_mods == -1
    elif pass_distance == PassDistance.LONG_PASS:
        assert pass_mods == -2
    elif pass_distance == PassDistance.LONG_BOMB:
        assert pass_mods == -3

