from tests.util import *


def test_dodge_reroll_success():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    assert not player.has_skill(Skill.DODGE)
    # allow a team reroll
    game.state.teams[0].state.rerolls = 1

    opponents = game.get_players_on_pitch(game.get_opp_team(current_team))
    game.put(player, Square(11, 11))

    opp_player = opponents[1]
    game.put(opp_player, Square(12, 12))
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(11, 12)
    assert game.get_player_at(to) is None
    assert len(D6.FixedRolls) == 0
    D6.fix(1)  # fail first dodge
    D6.fix(4)  # pass on re-roll

    game.step(Action(ActionType.MOVE, player=player, position=to))
    game.step(Action(ActionType.USE_REROLL))
    assert player.position == to
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)


def test_bonehead_reroll_success():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    player.extra_skills = [Skill.BONE_HEAD]

    game.state.teams[0].state.rerolls = 1
    game.put(player, Square(11, 11))

    D6.fix(1)  # fail first bonehead
    D6.fix(4)  # pass on re-roll
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))  # should bonehead and present reroll choice

    # check that in a reroll context the game domain context is still Bonehead
    proc = game.get_procedure()
    assert isinstance(proc, Reroll)
    # but that the top of the stack is a re-roll proc
    assert isinstance(proc.context, Bonehead)

    game.step(Action(ActionType.USE_REROLL))  # use reroll

    assert not player.state.bone_headed
    assert game.has_report_of_type(OutcomeType.FAILED_BONE_HEAD)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_BONE_HEAD)


def test_gfi_reroll_success():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    player.role.skills = []
    player.extra_skills = []
    assert not player.has_skill(Skill.SURE_FEET)
    player.state.moves = player.get_ma()

    game.state.teams[0].state.rerolls = 1
    game.put(player, Square(5, 5))

    D6.fix(1)  # fail first gfi
    D6.fix(4)  # pass on re-roll
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 1, player.position.y)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    game.step(Action(ActionType.USE_REROLL))  # use reroll

    assert player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_GFI)
    assert game.has_report_of_type(OutcomeType.REROLL_USED)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_GFI)


def test_gfi_reroll_fail():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    player.role.skills = []
    player.extra_skills = []
    assert not player.has_skill(Skill.SURE_FEET)
    player.state.moves = player.get_ma()

    game.state.teams[0].state.rerolls = 1
    game.put(player, Square(5, 5))

    # Armor roll
    D6.fix(1)
    D6.fix(1)

    D6.fix(1)  # fail first gfi
    D6.fix(1)  # FAIL re-roll
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 1, player.position.y)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    game.step(Action(ActionType.USE_REROLL))  # use reroll

    assert not player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_GFI)
    assert game.has_report_of_type(OutcomeType.REROLL_USED)
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_GFI)


def test_bonehead_loner_reroll_success():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    player.role.skills = []
    player.extra_skills = [Skill.BONE_HEAD, Skill.LONER]

    # make sure we don't get stuck waiting for re-roll actions
    game.state.teams[0].state.rerolls = 1
    game.put(player, Square(11, 11))

    D6.fix(1)  # fail first bonehead
    D6.fix(4)  # pass loner
    D6.fix(4)  # pass on re-roll
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))  # should bonehead and present reroll choice
    game.step(Action(ActionType.USE_REROLL))  # use reroll

    assert not player.state.bone_headed
    assert game.has_report_of_type(OutcomeType.FAILED_BONE_HEAD)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LONER)
    assert game.has_report_of_type(OutcomeType.REROLL_USED)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_BONE_HEAD)


def test_bonehead_loner_reroll_fail():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    player.role.skills = []
    player.extra_skills = [Skill.BONE_HEAD, Skill.LONER]

    # make sure we don't get stuck waiting for re-roll actions
    game.state.teams[0].state.rerolls = 1
    game.put(player, Square(11, 11))

    D6.fix(1)  # fail first bonehead
    D6.fix(3)  # fail loner
    D6.fix(6)  # pass on re-roll - shouldn't be used
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))  # should bonehead and present reroll choice
    game.step(Action(ActionType.USE_REROLL))  # use reroll - should fail loner test

    assert player.state.bone_headed
    assert game.has_report_of_type(OutcomeType.FAILED_BONE_HEAD)
    assert game.has_report_of_type(OutcomeType.FAILED_LONER)
    assert game.has_report_of_type(OutcomeType.REROLL_USED)  # reroll was wasted
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_BONE_HEAD)  # no bonehead success


