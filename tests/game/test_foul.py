from tests.util import *
import pytest
import math


def test_foul_fail():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    d = 3
    D6.fix(d)
    D6.fix(defender.get_av() - d)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_NOT_BROKEN)
    assert not game.has_report_of_type(OutcomeType.PLAYER_EJECTED)


def test_foul_success_stunned():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    d = 6
    D6.fix(d)
    D6.fix(target - d)

    # Injury
    D6.fix(1)
    D6.fix(2)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    assert not game.has_report_of_type(OutcomeType.PLAYER_EJECTED)


def test_foul_success_ko():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    d = 6
    D6.fix(d)
    D6.fix(target - d)

    # Injury
    D6.fix(4)
    D6.fix(5)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.KNOCKED_OUT)
    assert not game.has_report_of_type(OutcomeType.PLAYER_EJECTED)


def test_foul_success_cas():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    d = 6
    D6.fix(d)
    D6.fix(target - d)

    # Injury
    D6.fix(5)
    D6.fix(6)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.CASUALTY)
    assert not game.has_report_of_type(OutcomeType.PLAYER_EJECTED)


def test_foul_success_assist():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    assister = team.players[1]
    game.put(assister, Square(6, 6))

    # Armor
    target = defender.get_av() + 1
    d = 6
    D6.fix(d)
    D6.fix(defender.get_av() - d)

    # Injury
    D6.fix(1)
    D6.fix(2)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert not game.has_report_of_type(OutcomeType.PLAYER_EJECTED)


def test_foul_fail_ejected():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(1)
    D6.fix(1)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_NOT_BROKEN)
    assert game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler not in game.get_players_on_pitch(team)
    assert fouler.state.ejected


def test_foul_success_stunned_ejected_armor():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(6)
    D6.fix(6)

    # Injury
    D6.fix(1)
    D6.fix(2)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    assert game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler not in game.get_players_on_pitch(team)
    assert fouler.state.ejected


def test_foul_success_stunned_ejected_injury():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(5)
    D6.fix(6)

    # Injury
    D6.fix(2)
    D6.fix(2)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    assert game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler not in game.get_players_on_pitch(team)
    assert fouler.state.ejected


def test_foul_success_stunned_ejected_armor_bribe_success():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.bribes = 1
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(6)
    D6.fix(6)

    # Injury
    D6.fix(1)
    D6.fix(2)

    # Bribe
    D6.fix(2)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    game.step(Action(ActionType.USE_BRIBE))
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_BRIBE)
    assert not game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler in game.get_players_on_pitch(team)
    assert not fouler.state.ejected


def test_foul_success_stunned_ejected_armor_bribe_fail_success():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.bribes = 2
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(6)
    D6.fix(6)

    # Injury
    D6.fix(1)
    D6.fix(2)

    # Bribes
    D6.fix(1)
    D6.fix(2)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    game.step(Action(ActionType.USE_BRIBE))
    assert game.has_report_of_type(OutcomeType.FAILED_BRIBE)
    game.step(Action(ActionType.USE_BRIBE))
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_BRIBE)
    assert not game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler in game.get_players_on_pitch(team)
    assert not fouler.state.ejected


def test_foul_success_stunned_ejected_armor_bribe_fail_dont_bribe():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.bribes = 2
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(6)
    D6.fix(6)

    # Injury
    D6.fix(1)
    D6.fix(2)

    # Bribe
    D6.fix(1)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    game.step(Action(ActionType.USE_BRIBE))
    assert game.has_report_of_type(OutcomeType.FAILED_BRIBE)
    game.step(Action(ActionType.DONT_USE_BRIBE))
    assert game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler not in game.get_players_on_pitch(team)
    assert fouler.state.ejected


def test_foul_success_stunned_ejected_armor_bribe_fail():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.bribes = 1
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(6)
    D6.fix(6)

    # Injury
    D6.fix(1)
    D6.fix(2)

    # Bribe
    D6.fix(1)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    game.step(Action(ActionType.USE_BRIBE))
    assert game.has_report_of_type(OutcomeType.FAILED_BRIBE)
    assert game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler not in game.get_players_on_pitch(team)
    assert fouler.state.ejected


def test_foul_success_stunned_ejected_armor_dont_bribe():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.bribes = 1
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(6)
    D6.fix(6)

    # Injury
    D6.fix(1)
    D6.fix(2)

    # Bribe
    D6.fix(1)

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert game.has_report_of_type(OutcomeType.ARMOR_BROKEN)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    game.step(Action(ActionType.DONT_USE_BRIBE))
    assert game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler not in game.get_players_on_pitch(team)
    assert fouler.state.ejected

    
def test_foul_fail_ejected_ball_carrier():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    fouler = team.players[2]
    defender = other_team.players[2]
    game.put(fouler, Square(5, 5))
    game.get_ball().move_to( Square(5, 5) ) 
    game.get_ball().is_carried = True 
    
    game.put(defender, Square(6, 5))
    defender.state.up = False

    # Armor
    target = defender.get_av() + 1
    D6.fix(6)
    D6.fix(6)
    D6.fix(5)
    D6.fix(6)
    

    game.step(Action(ActionType.START_FOUL, player=fouler))
    assert game.get_ball().position == Square(5, 5)
    assert game.has_report_of_type(OutcomeType.FOUL_ACTION_STARTED)
    game.step(Action(ActionType.FOUL, position=defender.position))
    assert game.has_report_of_type(OutcomeType.FOUL)
    assert not game.has_report_of_type(OutcomeType.ARMOR_NOT_BROKEN)
    assert game.has_report_of_type(OutcomeType.PLAYER_EJECTED)
    assert fouler not in game.get_players_on_pitch(team)
    assert fouler.state.ejected    
    
    assert game.has_report_of_type(OutcomeType.BALL_BOUNCED)
    assert game.get_ball().position !=   Square(5, 5)