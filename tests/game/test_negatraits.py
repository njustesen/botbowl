import pytest
from ffai.core.game import *
from unittest.mock import *
import numpy as np


@pytest.mark.parametrize("trait", [[Skill.BONE_HEAD, Bonehead], [Skill.REALLY_STUPID, ReallyStupid], [Skill.WILD_ANIMAL, WildAnimal]])
@patch("ffai.core.game.Game")
def test_negatrait_pass_allows_player_action(mock_game, trait):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[trait[0]])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()

        assert isinstance(proc, trait[1])

        D6.FixedRolls.clear()
        D6.fix_result(4)  # fix a pass
        result = proc.step(None)
        assert result is True  # trait is done
        # get rid of trait
        stack.pop()
        proc = stack.peek()
        assert isinstance(proc, PlayerAction)
        assert proc.done is False


@pytest.mark.parametrize("trait", [[Skill.BONE_HEAD, Bonehead], [Skill.REALLY_STUPID, ReallyStupid], [Skill.WILD_ANIMAL, WildAnimal]])
@patch("ffai.core.game.Game")
def test_negatrait_fail_ends_turn(mock_game, trait):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[trait[0]])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, trait[1])

        D6.FixedRolls.clear()
        D6.fix_result(1)
        result = proc.step(None)
        assert result is True # bonehead is done without reroll
        # get rid of bonehead
        proc = stack.peek()
        assert isinstance(proc, EndPlayerTurn)
        assert proc.done is False
        # check state
        if trait[0] is Skill.BONE_HEAD:
            assert player.state.bone_headed
        elif trait[0] is Skill.REALLY_STUPID:
            assert player.state.really_stupid

# no wild animal test as wild animal has no state impact
@pytest.mark.parametrize("trait", [[Skill.BONE_HEAD, Bonehead], [Skill.REALLY_STUPID, ReallyStupid]])
@patch("ffai.core.game.Game")
def test_negatrait_success_resets_player_state(mock_game, trait):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        # create boneheaded player
        player = Player("1", role, "test", 1, "orc", extra_skills=[trait[0]])

        if trait[0] is Skill.BONE_HEAD:
            player.state.bone_headed = True
        elif trait[0] is Skill.REALLY_STUPID:
            player.state.really_stupid = True

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, trait[1])

        D6.FixedRolls.clear()
        D6.fix_result(4)
        result = proc.step(None)
        assert result is True  # negatrait is done without reroll

        # check state
        if trait[0] is Skill.BONE_HEAD:
            assert not player.state.bone_headed
        elif trait[0] is Skill.REALLY_STUPID:
            assert not player.state.really_stupid

@pytest.mark.parametrize("dice_value", [1,2,3])
@patch("ffai.core.game.Game")
def test_really_stupid_fails_without_support(mock_game, dice_value):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.REALLY_STUPID])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, ReallyStupid)

        D6.FixedRolls.clear()
        D6.fix_result(dice_value)
        result = proc.step(None)
        assert result is True  # trait is done without reroll

        proc = stack.peek()
        assert isinstance(proc, EndPlayerTurn)
        assert proc.done is False
        # check state
        assert player.state.really_stupid

@pytest.mark.parametrize("dice_value", [2,3])
@patch("ffai.core.game.Game")
def test_really_stupid_passes_with_support(mock_game, dice_value):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False
    role = Role("Blitzer", "orc", 6, 3, 3, 9, [], 50000, None)
    team_mate = Player("3", role, "test", 1, "orc")

    mock_game.get_adjacent_teammates.return_value = [team_mate]

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.REALLY_STUPID])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, ReallyStupid)

        D6.FixedRolls.clear()
        D6.fix_result(dice_value)
        result = proc.step(None)
        assert result is True  # negatrait is done without reroll

        # check state
        assert not player.state.really_stupid


@pytest.mark.parametrize("dice_value", [2,3])
@patch("ffai.core.game.Game")
def test_really_stupid_fails_if_support_is_really_stupid(mock_game, dice_value):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False
    role = Role("Blitzer", "orc", 6, 3, 3, 9, [], 50000, None)
    team_mate = Player("3", role, "test", 1, "orc", extra_skills=[Skill.REALLY_STUPID])

    mock_game.get_adjacent_teammates.return_value = [team_mate]

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.REALLY_STUPID])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, ReallyStupid)

        D6.FixedRolls.clear()
        D6.fix_result(dice_value)
        result = proc.step(None)
        assert result is True  # trait is done without reroll

        proc = stack.peek()
        assert isinstance(proc, EndPlayerTurn)
        assert proc.done is False
        # check state
        assert player.state.really_stupid


@pytest.mark.parametrize("action_type", [ActionType.START_MOVE, ActionType.START_FOUL, ActionType.START_HANDOFF, ActionType.START_PASS])
@patch("ffai.core.game.Game")
def test_wild_animal_fails_without_block_or_blitz(mock_game, action_type):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.WILD_ANIMAL])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(action_type, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, WildAnimal)

        D6.FixedRolls.clear()
        D6.fix_result(2)  # Fails without a block/blitz
        result = proc.step(None)
        assert result is True  # trait is done without reroll

        proc = stack.peek()
        assert isinstance(proc, EndPlayerTurn)
        assert proc.done is False
        # check state
        assert player.state.wild_animal



@pytest.mark.parametrize("action_type", [ActionType.START_BLITZ, ActionType.START_BLOCK] )
@patch("ffai.core.game.Game")
def test_wild_animal_passes_when_block_or_blitz(mock_game, action_type):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.WILD_ANIMAL])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(action_type, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, WildAnimal)

        D6.FixedRolls.clear()
        D6.fix_result(2)  # fails without block/blitz
        result = proc.step(None)
        assert result is True  # negatrait is done without reroll

        # check state
        assert not player.state.wild_animal

