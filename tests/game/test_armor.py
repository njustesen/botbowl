import pytest
from botbowl.core.game import *
from unittest.mock import *
import numpy as np

@patch("botbowl.core.game.Game")
def test_armour_with_mighty_blow(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 4+5 = 9 -> not broken without MB
        D6.FixedRolls.clear()
        D6.FixedRolls.append(4)
        D6.FixedRolls.append(5)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")
        blocker = Player("1", role, "test", 1, "orc", extra_skills=[Skill.MIGHTY_BLOW])
        arm = Armor(mock_game, player, inflictor=blocker)
        arm.step(action=None)
        
        # mighty blow makes armour broken (10)
        proc = stack.peek()
        assert isinstance(proc, Injury)
        assert proc.mighty_blow_used == True # indicate MB can't be used in Injury roll

@patch("botbowl.core.game.Game")
def test_armour_broken_with_mighty_blow_unused(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 5+5 = 10 -> broken without MB
        D6.FixedRolls.clear()
        D6.FixedRolls.append(5)
        D6.FixedRolls.append(6)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")
        blocker = Player("1", role, "test", 1, "orc", extra_skills=[Skill.MIGHTY_BLOW])
        arm = Armor(mock_game, player, inflictor=blocker)
        arm.step(action=None)
        
        # armour broken (10)
        proc = stack.peek()
        assert isinstance(proc, Injury)
        assert proc.mighty_blow_used == False # indicate MB can be used in Injury roll

@patch("botbowl.core.game.Game")
def test_armour_no_break(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 4+5 = 9 -> not broken without MB
        D6.FixedRolls.clear()
        D6.FixedRolls.append(4)
        D6.FixedRolls.append(5)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")
        blocker = Player("1", role, "test", 1, "orc")
        arm = Armor(mock_game, player, inflictor=blocker)
        arm.step(action=None)
        
        # NO mighty blow so proc is still Armor for unbroken
        proc = stack.peek()
        assert isinstance(proc, Armor)

