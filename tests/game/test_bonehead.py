import pytest
from ffai.core.game import *
from unittest.mock import *
import numpy as np


@patch("ffai.core.game.Game")
def test_bonehead_pass_allows_player_action(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.BONE_HEAD])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, Bonehead)

        D6.FixedRolls.clear()
        D6.fix_result(3)
        result = proc.step(None)
        assert result is True # bonehead is done
        # get rid of bonehead
        stack.pop()
        proc = stack.peek()
        assert isinstance(proc, PlayerAction)
        assert proc.done is False


@patch("ffai.core.game.Game")
def test_bonehead_fail_ends_turn(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.BONE_HEAD])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, Bonehead)

        D6.FixedRolls.clear()
        D6.fix_result(1)
        result = proc.step(None)
        assert result is True # bonehead is done without reroll
        # get rid of bonehead
        proc = stack.peek()
        assert isinstance(proc, EndPlayerTurn)
        assert proc.done is False
        assert player.state.bone_headed



@patch("ffai.core.game.Game")
def test_bonehead_success_resets_player_state(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    mock_game.can_use_reroll.return_value = False

    with patch("ffai.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        # create boneheaded player
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.BONE_HEAD])
        player.state.bone_headed = True

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)
        proc = stack.peek()
        assert isinstance(proc, Bonehead)

        D6.FixedRolls.clear()
        D6.fix_result(2)
        result = proc.step(None)
        assert result is True # bonehead is done without reroll

        assert player.state.bone_headed is False
