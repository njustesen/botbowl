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
    D6.fix_result(6)  # pass trait test

    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has not ended
    assert game.state.active_player is player

    # check the player state
    if trait[0] is Skill.BONE_HEAD:
        assert not player.state.bone_headed
    elif trait[0] is Skill.REALLY_STUPID:
        assert not player.state.really_stupid

    # check the player can continue move
    to = Square(player.position.x, player.position.y + 1)
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
    D6.fix_result(1)  # fail trait test

    game.step(Action(ActionType.START_MOVE, player=player))

    # check the player turn has ended
    assert game.state.active_player is not player

    # check the player state
    if trait[0] is Skill.BONE_HEAD:
        assert player.state.bone_headed
    elif trait[0] is Skill.REALLY_STUPID:
        assert player.state.really_stupid


# no wild animal test as wild animal has no state impact
@pytest.mark.parametrize("trait", [[Skill.BONE_HEAD, Bonehead], [Skill.REALLY_STUPID, ReallyStupid]])
def test_negatrait_success_resets_player_state(trait):
        game = get_game_turn()
        team = game.get_agent_team(game.actor)
        team.state.rerolls = 0  # ensure no reroll prompt

        players = game.get_players_on_pitch(team)
        player = players[1]
        player.extra_skills = [trait[0]]

        if trait[0] is Skill.BONE_HEAD:
            player.state.bone_headed = True
        elif trait[0] is Skill.REALLY_STUPID:
            player.state.really_stupid = True

        D6.FixedRolls.clear()
        D6.fix_result(6)  # pass trait test

        game.step(Action(ActionType.START_MOVE, player=player))

        # check the player turn has not ended
        assert game.state.active_player is player

        # check the player state
        if trait[0] is Skill.BONE_HEAD:
            assert not player.state.bone_headed
        elif trait[0] is Skill.REALLY_STUPID:
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
    D6.fix_result(dice_value)  # fail trait test

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
    game.put(team_mate, Square(5,6))

    adjacent = game.get_adjacent_teammates(player)
    assert len(adjacent) == 1

    D6.FixedRolls.clear()
    D6.fix_result(dice_value)  # pass trait test if supported

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
    D6.fix_result(dice_value)  # fail trait test if supported by really stupid player

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
    D6.fix_result(2)  # fails without block/blitz

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
    D6.fix_result(2)  # fails without block/blitz

    game.step(Action(action_type, player=attacker))

    # check the player turn has ended
    assert not game.has_report_of_type(OutcomeType.FAILED_WILD_ANIMAL)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_WILD_ANIMAL)



