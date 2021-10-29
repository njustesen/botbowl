import pytest
from botbowl.core.game import *
from unittest.mock import *
import numpy as np

'''
TODO: Re-write to not use mocks and fix issues.
@patch("botbowl.core.game.Game")
def test_turn_start_player_action_default(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)

        proc = stack.pop()
        assert isinstance(proc, PlayerAction)


@patch("botbowl.core.game.Game")
def test_turn_start_player_action_with_bonehead(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack

    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.BONE_HEAD])

        team = Team("humans", "dudes", Race("orc", None, 10, False, None))
        # test subject
        turn = Turn(mock_game, team, None, None)
        turn.started = True
        action = Action(ActionType.START_MOVE, player=player)
        turn.step(action)

        # uppermost is bonehead
        proc = stack.pop()
        assert isinstance(proc, Bonehead)
        # next is the intended player action
        proc = stack.pop()
        assert isinstance(proc, PlayerAction)
'''