from tests.util import *
import pytest


def test_failed_always_hungry_fail_escape():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE, Skill.ALWAYS_HUNGRY]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(1)  # Hungry
    D6.fix(1)  # Escape
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    game.step(Action(ActionType.DONT_USE_REROLL))
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert game.has_report_of_type(OutcomeType.FAILED_ESCAPE_BEING_EATEN)
    assert game.has_report_of_type(OutcomeType.EATEN_DURING_ALWAYS_HUNGRY)
    assert CasualtyEffect.DEAD in right_stuff.state.injuries_gained
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used


def test_failed_always_hungry_escaped():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE, Skill.ALWAYS_HUNGRY]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(1)  # Hungry
    D6.fix(2)  # Escape
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_ESCAPE_BEING_EATEN)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
