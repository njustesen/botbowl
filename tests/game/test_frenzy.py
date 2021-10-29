import pytest
from botbowl.core.game import *
from tests.util import *


def test_frenzy_block():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    attacker = None
    defender = None
    for p in players:
        if p.team != team:
            continue
        attacker = p
        adjacent = game.get_adjacent_opponents(attacker)
        if len(adjacent) > 0:
            attacker.extra_skills.append(Skill.FRENZY)
            defender = adjacent[0]
            break
    defender_pos = Square(defender.position.x, defender.position.y)
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_PUSH))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    assert attacker.position == defender_pos
    defender_pos = Square(defender.position.x, defender.position.y)
    assert game.state.active_player is attacker
    game.step(Action(ActionType.SELECT_PUSH))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    assert attacker.position == defender_pos
    assert game.state.active_player is not attacker


def test_frenzy_blitz():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    attacker = None
    defender = None
    for p in players:
        if p.team != team:
            continue
        attacker = p
        adjacent = game.get_adjacent_opponents(attacker)
        if len(adjacent) > 0:
            attacker.extra_skills.append(Skill.FRENZY)
            defender = adjacent[0]
            break
    defender_pos = Square(defender.position.x, defender.position.y)
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    game.step(Action(ActionType.START_BLITZ, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_PUSH))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    assert attacker.position == defender_pos
    defender_pos = Square(defender.position.x, defender.position.y)
    assert game.state.active_player is attacker
    game.step(Action(ActionType.SELECT_PUSH))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    assert attacker.position == defender_pos
    assert game.state.active_player is attacker


def test_frenzy_knocked_down():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    attacker = None
    defender = None
    for p in players:
        if p.team != team:
            continue
        attacker = p
        adjacent = game.get_adjacent_opponents(attacker)
        if len(adjacent) > 0:
            attacker.extra_skills.append(Skill.FRENZY)
            defender = adjacent[0]
            break
    defender_pos = Square(defender.position.x, defender.position.y)
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_DEFENDER_DOWN))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    assert attacker.position == defender_pos
    assert game.state.active_player is not attacker
