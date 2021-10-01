from tests.util import *
import pytest


def test_stab_success_kill():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0
    attacker, defender = get_block_players(game, team)
    attacker.role.skills = [Skill.STAB]
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    game.step(Action(ActionType.STAB, position=defender.position))
    assert game.has_report_of_type(OutcomeType.DEAD)


def test_stab_blitz():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0
    attacker, defender = get_block_players(game, team)
    attacker.role.skills = [Skill.STAB]
    game.step(Action(ActionType.START_BLITZ, player=attacker))
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    game.step(Action(ActionType.STAB, position=defender.position))
    assert game.has_report_of_type(OutcomeType.DEAD)
    assert game.has_report_of_type(OutcomeType.END_PLAYER_TURN)
