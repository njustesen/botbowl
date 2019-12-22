from tests.util import *


def test_casualty():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    attacker.extra_skills.append(Skill.BLOCK)
    defender_pos = Square(defender.position.x, defender.position.y)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix_result(BBDieResult.BOTH_DOWN)
    BBDie.fix_result(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix_result(5)
    D6.fix_result(5)
    # fix the injury roll to casualty
    D6.fix_result(5)
    D6.fix_result(5)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert defender.state.casualty_effect

    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_REGENERATION)
    assert not game.has_report_of_type(OutcomeType.FAILED_REGENERATION)


def test_casualty_regeneration_success():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    attacker.extra_skills.append(Skill.BLOCK)
    defender_pos = Square(defender.position.x, defender.position.y)
    defender.extra_skills.append(Skill.REGENERATION)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix_result(BBDieResult.BOTH_DOWN)
    BBDie.fix_result(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix_result(5)
    D6.fix_result(5)
    # fix the injury roll to casualty
    D6.fix_result(5)
    D6.fix_result(5)
    # add a value for casualty effect
    D6.fix_result(3)
    # fix the regeneration roll
    D6.fix_result(4)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_REGENERATION)
    assert defender in game.get_reserves(defender.team)
    assert not defender.state.casualty_effect


def test_casualty_regeneration_fail():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    attacker.extra_skills.append(Skill.BLOCK)
    defender_pos = Square(defender.position.x, defender.position.y)
    defender.extra_skills.append(Skill.REGENERATION)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix_result(BBDieResult.BOTH_DOWN)
    BBDie.fix_result(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix_result(5)
    D6.fix_result(5)
    # fix the injury roll to casualty
    D6.fix_result(5)
    D6.fix_result(5)
    # add a value for casualty effect
    D6.fix_result(4)
    # fix the regeneration roll
    D6.fix_result(3)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert game.has_report_of_type(OutcomeType.FAILED_REGENERATION)
    assert defender not in game.get_reserves(defender.team)
    assert defender.state.casualty_effect


