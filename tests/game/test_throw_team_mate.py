from tests.util import *
import pytest


def test_failed_pickup():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(1)  # Cause fumble
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=Square(5, 5)))
    assert game.has_report_of_type(OutcomeType.FUMBLE)
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
    assert right_stuff.position == right_stuff_position
    assert right_stuff.state.up


def test_successfull_land():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert passer.state.used
    assert not right_stuff.state.used
    assert Square(target_square.x + 3, target_square.y)
    assert right_stuff.state.up


def test_failed_landing():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(1)  # Land
    D6.fix(1)  # Armor roll
    D6.fix(1)  # Armor roll
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.FAILED_LAND)
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
    assert Square(target_square.x + 3, target_square.y)
    assert not right_stuff.state.up


def test_successful_landing():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
    assert Square(target_square.x + 3, target_square.y)
    assert right_stuff.state.up


def test_successful_landing_endzone():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    x_endzone = game.get_opp_endzone_x(team)
    y = 6
    endzone = Square(x_endzone, y)
    if x_endzone == 1:
        passer_square = Square(x_endzone + 5, y)
        right_stuff_square = Square(x_endzone + 6, y)
        target_square = Square(endzone.x + 1, endzone.y)
    else:
        passer_square = Square(x_endzone - 5, y)
        right_stuff_square = Square(x_endzone - 6, y)
        target_square = Square(endzone.x - 1, endzone.y)
    game.put(passer, passer_square)
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    game.put(right_stuff, right_stuff_square)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(4)  # Backward scatter
    D8.fix(5)  # Forward scatter
    if x_endzone == 1:
        D8.fix(4)  # Backward scatter
    else:
        D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert game.has_report_of_type(OutcomeType.TOUCHDOWN)


