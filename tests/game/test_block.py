from tests.util import *
import pytest


def test_block_uphill():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = 0 - attacker.get_st() - 1 # set up uphill block
    defender_pos = Square(defender.position.x, defender.position.y)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    actions = game.get_available_actions()
    assert len(actions) == 3  # 3 die block uphill, no reroll
    game.step(Action(ActionType.SELECT_ATTACKER_DOWN))
    assert not attacker.state.up


def test_block_uphill_reroll_refused():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 1

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = 0 - attacker.get_st() - 1 # set up uphill block
    defender_pos = Square(defender.position.x, defender.position.y)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    actions = game.get_available_actions()
    assert len(actions) == 5  # 3 die block uphill, with reroll
    # dice action choices are disabled - it's a reroll choice
    assert not actions[0].disabled
    assert not actions[1].disabled
    assert actions[2].disabled
    assert actions[3].disabled
    assert actions[4].disabled

    game.step(Action(ActionType.DONT_USE_REROLL))
    actions = game.get_available_actions()
    assert len(actions) == 3  # 3 die block uphill, no reroll choices
    # dice action choices are no longer disabled
    assert not actions[0].disabled
    assert not actions[1].disabled
    assert not actions[2].disabled

    game.step(Action(ActionType.SELECT_ATTACKER_DOWN))
    assert not attacker.state.up

def test_block_uphill_reroll_used():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 1

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = 0 - attacker.get_st() - 1 # set up uphill block
    defender_pos = Square(defender.position.x, defender.position.y)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    actions = game.get_available_actions()
    assert len(actions) == 5  # 3 die block uphill, with reroll
    # dice action choices are disabled - it's a reroll choice
    assert not actions[0].disabled
    assert not actions[1].disabled
    assert actions[2].disabled
    assert actions[3].disabled
    assert actions[4].disabled

    game.step(Action(ActionType.USE_REROLL))
    actions = game.get_available_actions()
    assert len(actions) == 3  # 3 die block uphill, no reroll choices
    # dice action choices are no longer disabled
    assert not actions[0].disabled
    assert not actions[1].disabled
    assert not actions[2].disabled

    game.step(Action(ActionType.SELECT_BOTH_DOWN))
    assert attacker.state.up == attacker.has_skill(Skill.BLOCK)
    assert defender.state.up == defender.has_skill(Skill.BLOCK)

def test_block():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    defender_pos = Square(defender.position.x, defender.position.y)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_ATTACKER_DOWN))
    assert not attacker.state.up


def test_block_reroll():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 1

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.USE_REROLL))
    game.step(Action(ActionType.SELECT_ATTACKER_DOWN))
    assert not attacker.state.up


def test_block_ignore_reroll():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 1

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    #defender.extra_skills.append(Skill.STAND_FIRM)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_DEFENDER_DOWN))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    game.step(Action(ActionType.FOLLOW_UP, position=game.get_available_actions()[0].positions[0]))
    assert attacker.state.up
    assert not defender.state.up


def test_available_actions_on_block_reroll():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 1

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.DEFENDER_DOWN)
    BBDie.fix(BBDieResult.DEFENDER_STUMBLES)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    BBDie.fix(BBDieResult.ATTACKER_DOWN)
    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    actions = game.get_available_actions()
    assert len(actions) == 3
    assert actions[0].action_type == ActionType.USE_REROLL
    assert actions[1].action_type == ActionType.SELECT_DEFENDER_DOWN
    assert actions[2].action_type == ActionType.SELECT_DEFENDER_STUMBLES
    game.step(Action(ActionType.USE_REROLL))
    actions = game.get_available_actions()
    assert len(actions) == 2


