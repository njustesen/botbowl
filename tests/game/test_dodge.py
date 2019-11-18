import pytest
from ffai.core.game import *
from tests.util import *

'''
Not working..

def test_no_dodge():
    game = get_game_turn(empty=True)
    team = game.get_agent_team(game.actor)
    player = game.get_reserves(team)[0]
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player.role = role
    game.state.player_by_id[player.player_id] = player
    game.put(player, Square(11, 11))
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(11, 12)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert player.position == to
    assert player.state.up
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)


def test_dodge_no_modifier():
    game = get_game_turn()
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, game.state.home_team)
    game.put(player, Square(11, 11))
    opp_player = Player("2", role, "test", 1, game.state.away_team)
    game.put(opp_player, Square(12, 12))
    modifier = game.get_dodge_modifiers(player, Square(10, 10))
    assert modifier == 1


def test_dodge_one_modifier():
    game = get_game_turn()
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, game.state.home_team)
    game.put(player, Square(11, 11))
    opp_player = Player("2", role, "test", 1, game.state.away_team)
    game.put(opp_player, Square(12, 12))
    modifier = game.get_dodge_modifiers(player, Square(12, 11))
    assert modifier == 0


def test_dodge_two_modifier():
    game = get_game_turn()
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, game.state.home_team)
    game.put(player, Square(11, 11))
    opp_player = Player("2", role, "test", 1, game.state.away_team)
    game.put(opp_player, Square(12, 12))
    opp_player = Player("3", role, "test", 2, game.state.away_team)
    game.put(opp_player, Square(11, 12))
    pos = Square(12, 11)
    modifier = game.get_dodge_modifiers(player, pos)
    assert modifier == -1


def test_prehensile_tail_modifier():
    game = get_game_turn()
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, "orc", game.state.home_team)
    tail_player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.PREHENSILE_TAIL])
    game.put(player, Square(5, 5))
    game.put(tail_player, Square(6, 6))
    modifier = game.get_dodge_modifiers(game, player, Square(4, 4))
    assert modifier == 0


@pytest.mark.parametrize("ag", list(range(11)))
def test_dodge_fail_one(ag):
    game = get_game_turn()
    role = Role("Lineman", "orc", 6, 3, ag, 9, [], 50000, False)
    current_team = game.get_agent_team(game.actor)
    player = Player("1", role, "test", 1, current_team)
    game.put(player, Square(11, 11))
    opp_player = Player("2", role, "test", 1, game.state.away_team)
    game.put(opp_player, Square(12, 12))
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(11, 12)
    D6.fix_result(1)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert player.position == to
    assert not player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)

'''