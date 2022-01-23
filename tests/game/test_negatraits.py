import pytest
from tests.util import *


@pytest.mark.parametrize("trait", [[Skill.BONE_HEAD, Bonehead], [Skill.REALLY_STUPID, ReallyStupid], [Skill.WILD_ANIMAL, WildAnimal]])
def test_negatrait_pass_allows_player_action(trait):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [trait[0]]

    D6.FixedRolls.clear()
    D6.fix(6)  # pass trait test

    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has not ended
    assert game.state.active_player is player

    # check the player state
    if trait[0] is Skill.BONE_HEAD:
        assert not player.state.bone_headed
    elif trait[0] is Skill.REALLY_STUPID:
        assert not player.state.really_stupid

    # check the player can continue move
    to = game.get_adjacent_squares(player.position, occupied=False)[0]
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert player.position == to


@pytest.mark.parametrize("trait", [[Skill.BONE_HEAD, Bonehead], [Skill.REALLY_STUPID, ReallyStupid], [Skill.WILD_ANIMAL, WildAnimal]])
def test_negatrait_fail_ends_turn(trait):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [trait[0]]

    D6.FixedRolls.clear()
    D6.fix(1)  # fail trait test

    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has ended
    assert game.state.active_player is not player

    # check the player state
    if trait[0] is Skill.BONE_HEAD:
        assert player.state.bone_headed
    elif trait[0] is Skill.REALLY_STUPID:
        assert player.state.really_stupid


def test_take_root_fail_does_not_end_block_action():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    attacker, defender = get_block_players(game, team)
    attacker.extra_skills = [Skill.TAKE_ROOT]

    D6.FixedRolls.clear()
    D6.fix(1)  # fail take root test

    game.step(Action(ActionType.START_BLOCK, player=attacker))

    # check the player turn has not ended
    assert game.state.active_player is attacker


def test_take_root_ends_move_turn():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.TAKE_ROOT]

    D6.FixedRolls.clear()
    D6.fix(1)  # fail trait test

    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has ended
    assert game.state.active_player is not player


def test_taken_root_players_can_stand_up():
    game = get_game_turn()
    game.config.pathfinding_enabled = True
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.TAKE_ROOT]
    player.state.up = False

    D6.FixedRolls.clear()
    D6.fix(1)  # fail take root

    game.step(Action(ActionType.START_MOVE, player=player))

    # It's still that players turn
    assert game.state.active_player is player

    D6.fix(4)  # succeed stand up roll

    game.step(Action(ActionType.STAND_UP, player=player))

    assert player.state.up
    assert player.state.taken_root

    # Nothing more to do
    assert game.state.active_player is not player


def test_take_root_fail_reduces_ma_and_prevents_movement():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.TAKE_ROOT]

    D6.FixedRolls.clear()
    D6.fix(1)  # fail trait test

    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player ma is now 0
    assert player.get_ma() == 0
    for action in game.get_available_actions():
        assert not action.action_type == ActionType.MOVE


# no wild animal test as wild animal has no state impact
@pytest.mark.parametrize("trait", [Skill.BONE_HEAD, Skill.REALLY_STUPID])
def test_negatrait_success_resets_player_state(trait):
        game = get_game_turn()
        team = game.get_agent_team(game.actor)
        team.state.rerolls = 0  # ensure no reroll prompt

        players = game.get_players_on_pitch(team)
        player = players[1]
        player.extra_skills = [trait]

        if trait is Skill.BONE_HEAD:
            player.state.bone_headed = True
        elif trait is Skill.REALLY_STUPID:
            player.state.really_stupid = True

        D6.FixedRolls.clear()
        D6.fix(6)  # pass trait test

        game.step(Action(ActionType.START_MOVE, player=player))

        # check the player turn has not ended
        assert game.state.active_player is player

        # check the player state
        if trait is Skill.BONE_HEAD:
            assert not player.state.bone_headed
        elif trait is Skill.REALLY_STUPID:
            assert not player.state.really_stupid


@pytest.mark.parametrize("dice_value", [1,2,3])
def test_really_stupid_fails_without_support(dice_value):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.REALLY_STUPID]

    game.put(player, Square(5, 5))
    adjacent = game.get_adjacent_teammates(player)
    assert len(adjacent) == 0

    D6.FixedRolls.clear()
    D6.fix(dice_value)  # fail trait test

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has ended
    assert game.state.active_player is not player

    # check the player state
    assert player.state.really_stupid


@pytest.mark.parametrize("dice_value", [2,3])
def test_really_stupid_passes_with_support(dice_value):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.REALLY_STUPID]

    team_mate = players[2]
    assert not team_mate.has_skill(Skill.REALLY_STUPID)
    game.put(player, Square(5, 5))
    game.put(team_mate, Square(5, 6))

    adjacent = game.get_adjacent_teammates(player)
    assert len(adjacent) == 1

    D6.FixedRolls.clear()
    D6.fix(dice_value)  # pass trait test if supported

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has ended
    assert game.state.active_player is player

    # check the player state
    assert not player.state.really_stupid


@pytest.mark.parametrize("dice_value", [2,3])
def test_really_stupid_fails_if_support_is_really_stupid(dice_value):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.REALLY_STUPID]

    team_mate = players[2]
    team_mate.extra_skills.append(Skill.REALLY_STUPID)
    game.put(player, Square(5, 5))
    game.put(team_mate, Square(5,6))

    adjacent = game.get_adjacent_teammates(player)
    assert len(adjacent) == 1

    D6.FixedRolls.clear()
    D6.fix(dice_value)  # fail trait test if supported by really stupid player

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has ended
    assert game.state.active_player is not player

    # check the player state
    assert player.state.really_stupid        # check state


@pytest.mark.parametrize("action_type", [ActionType.START_MOVE, ActionType.START_FOUL, ActionType.START_HANDOFF, ActionType.START_PASS])
def test_wild_animal_fails_without_block_or_blitz(action_type):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.WILD_ANIMAL]

    D6.FixedRolls.clear()
    D6.fix(2)  # fails without block/blitz

    game.step(Action(action_type, player=player))

    # check the player turn has ended
    assert game.state.active_player is not player
    assert game.has_report_of_type(OutcomeType.FAILED_WILD_ANIMAL)


@pytest.mark.parametrize("action_type", [ActionType.START_BLITZ, ActionType.START_BLOCK])
def test_wild_animal_passes_when_block_or_blitz(action_type):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    attacker, defender = get_block_players(game, team)  # need adjacent players here.

    attacker.extra_skills = [Skill.WILD_ANIMAL]

    D6.FixedRolls.clear()
    D6.fix(2)  # fails without block/blitz

    game.step(Action(action_type, player=attacker))

    # check the player turn has ended
    assert not game.has_report_of_type(OutcomeType.FAILED_WILD_ANIMAL)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_WILD_ANIMAL)


def test_take_root_doesnt_trigger_if_rooted():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.TAKE_ROOT]
    player.state.taken_root = True

    D6.FixedRolls.clear()
    D6.fix(2)  # pass take root if it happens

    game.step(Action(ActionType.START_MOVE, player=player))

    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_TAKE_ROOT)


def test_rooted_players_cannot_start_a_move_or_blitz():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    player, defender = get_block_players(game, team)  # need adjacent players here.

    player.extra_skills = [Skill.TAKE_ROOT]
    player.state.taken_root = True

    # need to end turn here as available actions were set before taken_root happened.
    game.step(Action(ActionType.END_TURN))
    game.step(Action(ActionType.END_TURN))

    actions = game.get_available_actions()

    for action in actions:
        if action.action_type is ActionType.START_MOVE:
            assert player not in action.players
        if action.action_type is ActionType.START_BLITZ:
            assert player not in action.players
        if action.action_type is ActionType.START_BLOCK:
            assert player in action.players
        if action.action_type is ActionType.START_PASS:
            if game.get_ball_carrier() == player:
                assert player in action.players
        if action.action_type is ActionType.START_HANDOFF:
            if game.get_ball_carrier() == player:
                assert player in action.players
        if action.action_type is ActionType.START_FOUL:
            if len(game.get_adjacent_opponents(player, down=True, standing=False)) > 0:
                assert player in action.players

def test_take_root_not_removed_on_end_turn():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.TAKE_ROOT]
    player.state.taken_root = True

    game.step(Action(ActionType.END_TURN))

    assert player.state.taken_root


def test_take_root_removed_on_touchdown():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.TAKE_ROOT]
    player.state.taken_root = True

    scoring_player = players[2]
    game.move(scoring_player, Square(2, 5))
    game.get_ball().move_to(scoring_player.position)
    game.get_ball().is_carried = True
    assert not game.arena.is_in_opp_endzone(scoring_player.position, scoring_player.team == game.state.home_team)

    to = Square(1, 5)
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=scoring_player))
    game.step(Action(ActionType.MOVE, player=scoring_player, position=to))

    assert game.has_report_of_type(OutcomeType.TOUCHDOWN)
    assert not player.state.taken_root


def test_take_root_removed_on_new_half():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[1]
    player.extra_skills = [Skill.TAKE_ROOT]
    player.state.taken_root = True

    i = 0
    while game.state.half == 1 and i < 18:
        game.step(Action(ActionType.END_TURN))
        i += 1

    assert game.has_report_of_type(OutcomeType.END_OF_FIRST_HALF)
    assert not player.state.taken_root


def test_take_root_removed_on_knockdown():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0  # ensure no reroll prompt

    attacker, defender = get_block_players(game, team)

    defender.extra_skills = [Skill.TAKE_ROOT]
    defender.state.taken_root = True
    assert not defender.has_skill(Skill.BLOCK)
    attacker.extra_skills = [Skill.BLOCK]

    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.BOTH_DOWN)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_BOTH_DOWN))

    assert not defender.state.up
    assert game.has_report_of_type(OutcomeType.KNOCKED_DOWN)
    assert not defender.state.taken_root

def test_taken_root_players_may_not_follow_up():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_skills = [Skill.TAKE_ROOT]
    attacker.state.taken_root = True
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.

    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.DEFENDER_DOWN)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_DEFENDER_DOWN))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    for action in game.get_available_actions():
        assert action.action_type is not ActionType.FOLLOW_UP


def test_taken_root_players_may_not_follow_up_push():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    attacker.extra_skills = [Skill.TAKE_ROOT]
    attacker.state.taken_root = True
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.

    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.PUSH)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_PUSH))
    game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    for action in game.get_available_actions():
        assert action.action_type is not ActionType.FOLLOW_UP


def test_taken_root_players_may_not_be_pushed():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 0

    attacker, defender = get_block_players(game, team)
    defender.extra_skills = [Skill.TAKE_ROOT]
    defender.state.taken_root = True
    attacker.extra_st = defender.get_st() - attacker.get_st() + 1  # make this a 2 die block.
    def_pos = defender.position

    # it's a 2 dice block
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)

    game.step(Action(ActionType.START_BLOCK, player=attacker))
    game.step(Action(ActionType.BLOCK, position=defender.position))
    game.step(Action(ActionType.SELECT_PUSH))
    for action in game.get_available_actions():
        assert action.action_type is not ActionType.PUSH
    # game.step(Action(ActionType.PUSH, position=game.get_available_actions()[0].positions[0]))
    for action in game.get_available_actions():
        assert action.action_type is not ActionType.FOLLOW_UP

    assert defender.position is def_pos

