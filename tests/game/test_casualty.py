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
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix(5)
    D6.fix(5)
    # fix the injury roll to casualty
    D6.fix(5)
    D6.fix(5)
    # fix the casualty roll #1 (Gouged Eye / MNG)
    D6.fix(4)
    D8.fix(3)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert defender.state.injuries_gained[0] is CasualtyEffect.MNG

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
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix(5)
    D6.fix(5)
    # fix the injury roll to casualty
    D6.fix(5)
    D6.fix(5)
    # add a value for casualty effect
    D6.fix(3)
    # fix the regeneration roll
    D6.fix(4)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_REGENERATION)
    assert defender in game.get_reserves(defender.team)
    assert not defender.state.injuries_gained


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
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix(5)
    D6.fix(5)
    # fix the injury roll to casualty
    D6.fix(5)
    D6.fix(5)
    # add a value for casualty effect
    D6.fix(4)
    # fix the regeneration roll
    D6.fix(3)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert game.has_report_of_type(OutcomeType.FAILED_REGENERATION)
    assert defender not in game.get_reserves(defender.team)
    assert defender.state.injuries_gained


def test_casualty_with_decay():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    attacker.extra_skills.append(Skill.BLOCK)
    defender_pos = Square(defender.position.x, defender.position.y)
    defender.extra_skills.append(Skill.DECAY)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix(5)
    D6.fix(5)
    # fix the injury roll to casualty
    D6.fix(5)
    D6.fix(5)
    # fix the casualty roll #1 (Gouged Eye / MNG)
    D6.fix(4)
    D8.fix(3)
    # fix the casualty roll #2 (BH / none)
    D6.fix(3)
    D8.fix(1)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert defender.state.injuries_gained[0] is CasualtyEffect.MNG
    assert game.has_report_of_type(OutcomeType.MISS_NEXT_GAME)
    assert len(defender.state.injuries_gained) == 1
    assert game.has_report_of_type(OutcomeType.BADLY_HURT)


def test_casualty_with_decay_mng_twice_is_just_one():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    attacker.extra_skills.append(Skill.BLOCK)
    defender_pos = Square(defender.position.x, defender.position.y)
    defender.extra_skills.append(Skill.DECAY)
    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix(5)
    D6.fix(5)
    # fix the injury roll to casualty
    D6.fix(5)
    D6.fix(5)
    # fix the casualty roll #1 (Gouged Eye / MNG)
    D6.fix(4)
    D8.fix(3)
    # fix the casualty roll #2 (BH / none)
    D6.fix(4)
    D8.fix(4)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert defender.state.injuries_gained[0] is CasualtyEffect.MNG
    assert game.has_report_of_type(OutcomeType.MISS_NEXT_GAME)
    assert len(defender.state.injuries_gained) == 1


def test_casualty_regeneration_success():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    attacker.extra_skills.append(Skill.BLOCK)
    defender_pos = Square(defender.position.x, defender.position.y)
    defender.extra_skills.append(Skill.REGENERATION)
    defender.extra_skills.append(Skill.DECAY)

    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix(5)
    D6.fix(5)
    # fix the injury roll to casualty
    D6.fix(5)
    D6.fix(5)
    # add a value for casualty effect
    D6.fix(3)
    # fix the regeneration roll
    D6.fix(4)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_REGENERATION)
    assert defender in game.get_reserves(defender.team)
    assert len(defender.state.injuries_gained) == 0


def test_casualty_regeneration_failure():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    attacker.extra_skills.append(Skill.BLOCK)
    defender_pos = Square(defender.position.x, defender.position.y)
    defender.extra_skills.append(Skill.REGENERATION)
    defender.extra_skills.append(Skill.DECAY)

    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.BOTH_DOWN)
    BBDie.fix(BBDieResult.BOTH_DOWN)
    D6.FixedRolls.clear()
    # fix the armour roll
    D6.fix(5)
    D6.fix(5)
    # fix the injury roll to casualty
    D6.fix(5)
    D6.fix(5)
    # add a value for casualty effect - BH
    D6.fix(3)
    # fix the regeneration roll
    D6.fix(2)
    # add a value for casualty effect #2 - DEAD
    D6.fix(6)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert game.has_report_of_type(OutcomeType.FAILED_REGENERATION)
    assert not defender in game.get_reserves(defender.team)
    assert game.has_report_of_type(OutcomeType.BADLY_HURT)
    assert game.has_report_of_type(OutcomeType.DEAD)
    assert len(defender.state.injuries_gained) == 1
