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




