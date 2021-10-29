import pytest
from botbowl.core.game import *
from unittest.mock import *


@patch("botbowl.core.game.Game")
def test_injury_default_stunned(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 3+4 = 7 => Stunned
        D6.FixedRolls.clear()
        D6.FixedRolls.append(3)
        D6.FixedRolls.append(4)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")
        blocker = Player("1", role, "test", 1, "orc")
        injury = Injury(mock_game, player, inflictor=blocker)
        injury.step(action=None)
        
        # NO mighty blow so proc is still Injury for stunned result
        proc = stack.peek()
        assert isinstance(proc, Injury)

@patch("botbowl.core.game.Game")
def test_injury_default_casualty(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 3+6 = 9 => KO
        D6.FixedRolls.clear()
        D6.FixedRolls.append(3)
        D6.FixedRolls.append(6)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")
        blocker = Player("1", role, "test", 1, "orc")
        injury = Injury(mock_game, player, inflictor=blocker)
        injury.step(action=None)
        
        # NO mighty blow so proc is still Injury for stunned result
        proc = stack.peek()
        assert isinstance(proc, KnockOut)


@patch("botbowl.core.game.Game")
def test_injury_with_mighty_blow(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 3+4 = 7 => Stunned
        D6.FixedRolls.clear()
        D6.FixedRolls.append(3)
        D6.FixedRolls.append(4)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")
        blocker = Player("1", role, "test", 1, "orc", extra_skills=[Skill.MIGHTY_BLOW])
        injury = Injury(mock_game, player, inflictor=blocker)
        injury.step(action=None)
        
        # mighty blow should make it a KO
        proc = stack.peek()
        assert isinstance(proc, KnockOut)


@patch("botbowl.core.game.Game")
def test_injury_with_mighty_blow_used(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 3+4 = 7 => Stunned
        D6.FixedRolls.clear()
        D6.FixedRolls.append(3)
        D6.FixedRolls.append(4)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc")
        blocker = Player("1", role, "test", 1, "orc", extra_skills=[Skill.MIGHTY_BLOW])
        injury = Injury(mock_game, player, inflictor=blocker, mighty_blow_used=True)
        injury.step(action=None)
        
        # can't use mighty blow again, so should still be Injury for Stunned result
        proc = stack.peek()
        assert isinstance(proc, Injury)


@patch("botbowl.core.game.Game")
def test_injury_stunned_with_stunty(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack    
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value=stack

        # fix the dice rolls - 3+4 = 7 => KO for stunty
        D6.FixedRolls.clear()
        D6.FixedRolls.append(3)
        D6.FixedRolls.append(4)

        role = Role("Blitzer", "orc", 6,3,3,9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.STUNTY])
        blocker = Player("1", role, "test", 1, "orc")
        injury = Injury(mock_game, player, inflictor=blocker)
        injury.step(action=None)
        
        # stunty should make it a KO
        proc = stack.peek()
        assert isinstance(proc, KnockOut)


@patch("botbowl.core.game.Game")
def test_injury_ko_with_stunty(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        # fix the dice rolls - 3+6 = 9 => CAS for stunty
        D6.FixedRolls.clear()
        D6.FixedRolls.append(3)
        D6.FixedRolls.append(6)

        role = Role("Blitzer", "orc", 6, 3, 3, 9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.STUNTY])
        blocker = Player("1", role, "test", 1, "orc")
        injury = Injury(mock_game, player, inflictor=blocker)
        injury.step(action=None)

        # stunty should make it a Casualty
        proc = stack.peek()
        assert isinstance(proc, Casualty)


@patch("botbowl.core.game.Game")
def test_injury_ko_with_thick_skull(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        # fix the dice rolls - 4+4 = 8 => stunned for Thick skull
        D6.FixedRolls.clear()
        D6.FixedRolls.append(4)
        D6.FixedRolls.append(4)

        role = Role("Blitzer", "orc", 6, 3, 3, 9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.THICK_SKULL])
        blocker = Player("1", role, "test", 1, "orc")
        injury = Injury(mock_game, player, inflictor=blocker)
        injury.step(action=None)

        # thick skull should make it Stunned => Injury
        proc = stack.peek()
        assert isinstance(proc, Injury)


@patch("botbowl.core.game.Game")
def test_injury_cas_with_thick_skull(mock_game):
    # patch the mock game proc stack
    stack = Stack()
    mock_game.state.stack = stack
    with patch("botbowl.core.util.Stack", new_callable=PropertyMock) as a:
        a.return_value = stack

        # fix the dice rolls - 5+5 = 10 => no effect for Thick skull
        D6.FixedRolls.clear()
        D6.FixedRolls.append(5)
        D6.FixedRolls.append(5)

        role = Role("Blitzer", "orc", 6, 3, 3, 9, [], 50000, None)
        player = Player("1", role, "test", 1, "orc", extra_skills=[Skill.THICK_SKULL])
        blocker = Player("1", role, "test", 1, "orc")
        injury = Injury(mock_game, player, inflictor=blocker)
        injury.step(action=None)

        # thick skull has no effect - it's a casualty
        proc = stack.peek()
        assert isinstance(proc, Casualty)

