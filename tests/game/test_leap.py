import pytest
from botbowl.core.game import *
from tests.util import *


def test_leap_one_square_success():
    game = get_game_turn()
    game.clear_board()

    team = game.get_agent_team(game.actor)

    player = team.players[0]
    player.extra_skills = [Skill.LEAP]
    player.role.ag = 3

    game.put(player, Square(1, 1))

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    from_pos = Square(player.position.x, player.position.y)
    to_pos = Square(from_pos.x + 1, from_pos.y)

    D6.fix(4)
    game.step(Action(ActionType.LEAP, player=player, position=to_pos))

    assert player.position == to_pos
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LEAP)
    assert player.state.moves == from_pos.distance(to_pos) == 1


def test_leap_two_squares_success():
    game = get_game_turn()
    game.clear_board()

    team = game.get_agent_team(game.actor)

    player = team.players[0]
    player.extra_skills = [Skill.LEAP]
    player.role.ag = 3

    game.put(player, Square(1, 1))

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 2, player.position.y)

    D6.fix(4)
    game.step(Action(ActionType.LEAP, player=player, position=to))

    assert player.position == to
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LEAP)
    assert player.state.moves == 2


def test_leap_two_squares_fail():
    game = get_game_turn()
    game.clear_board()

    team = game.get_agent_team(game.actor)

    player = team.players[0]
    player.extra_skills = [Skill.LEAP]
    player.role.ag = 3

    game.put(player, Square(1, 1))

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 2, player.position.y)

    D6.fix(3)
    game.step(Action(ActionType.LEAP, player=player, position=to))
    game.step(Action(ActionType.DONT_USE_REROLL))

    assert player.position == to
    assert not player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_LEAP)
    assert not game.state.active_player == player


def test_leap_very_long_legs_success():
    game = get_game_turn()
    game.clear_board()

    team = game.get_agent_team(game.actor)

    player = team.players[0]
    player.extra_skills = [Skill.LEAP, Skill.VERY_LONG_LEGS]
    player.role.ag = 3

    game.put(player, Square(1, 1))

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 2, player.position.y)

    D6.fix(3)
    game.step(Action(ActionType.LEAP, player=player, position=to))

    assert player.position == to
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LEAP)
    assert player.state.moves == 2


def test_leap_tackle_zone():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    player = players[1]
    player.extra_skills = [Skill.LEAP]
    player.role.ag = 3

    opp_team = game.get_opp_team(team)
    opp_player = opp_team.players[0]
    game.put(opp_player, Square(player.position.x + 1, player.position.y))
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 2, player.position.y)
    D6.fix(4)
    game.step(Action(ActionType.LEAP, player=player, position=to))

    assert player.position == to
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LEAP)
    assert player.state.moves == 2


def test_two_leaps():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    player = players[1]
    player.extra_skills = [Skill.LEAP]
    player.role.ag = 3

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 2, player.position.y)
    D6.fix(4)
    game.step(Action(ActionType.LEAP, player=player, position=to))
    assert player.position == to
    assert player.state.moves == 2
    to = Square(player.position.x + 2, player.position.y)

    with pytest.raises(InvalidActionError):
        game.step(Action(ActionType.LEAP, player=player, position=to))
