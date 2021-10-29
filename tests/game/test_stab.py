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
    game.config.pathfinding_enabled = True
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    stabber = team.players[0]
    stabber.role.skills = [Skill.STAB]
    defender = other_team.players[0]
    defender.role.skills = []
    game.put(stabber, Square(1, 1))
    game.put(defender, Square(3, 3))
    game.step(Action(ActionType.START_BLITZ, player=stabber))
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    D6.fix(6)
    game.step(Action(ActionType.STAB, position=defender.position))
    assert game.has_report_of_type(OutcomeType.DEAD)
    assert game.has_report_of_type(OutcomeType.END_PLAYER_TURN)
