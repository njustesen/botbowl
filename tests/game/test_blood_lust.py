from tests.util import *
import pytest


def test_blood_lust_roll_success():
    data = (ActionType.START_MOVE, OutcomeType.MOVE_ACTION_STARTED) 
 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(5, 5))
    game.put(defender, Square(6, 5))
    game.state.teams[0].state.rerolls = 0
    game.state.teams[1].state.rerolls = 0
    
    D6.fix(2)
    
    game.step(Action(data[0], player=player))
    assert not player.state.blood_lust 
    assert game.has_report_of_type(data[1])
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_BLOOD_LUST)
 
    assert not player.state.blood_lust 
    
    game.step(Action(ActionType.END_PLAYER_TURN)) 
    
    assert game.has_report_of_type(data[1] )
    assert player in game.get_players_on_pitch(player.team)



 
@pytest.mark.parametrize("data", [  (ActionType.START_FOUL, OutcomeType.FOUL_ACTION_STARTED), 
                                    (ActionType.START_MOVE, OutcomeType.MOVE_ACTION_STARTED), 
                                    (ActionType.START_PASS, OutcomeType.PASS_ACTION_STARTED),
                                    (ActionType.START_HANDOFF, OutcomeType.HANDOFF_ACTION_STARTED), 
                                    #(ActionType.START_BLOCK, OutcomeType.BLOCK_ACTION_STARTED),
                                    (ActionType.START_BLITZ, OutcomeType.BLITZ_ACTION_STARTED)]) 
def test_blood_lust_roll_fail_eject(data): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(5, 5))
    game.put(defender, Square(6, 5))
    game.state.teams[0].state.rerolls = 0
    game.state.teams[1].state.rerolls = 0
    
    D6.fix(1)
    
    game.step(Action(data[0], player=player))
    assert game.has_report_of_type(OutcomeType.FAILED_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_BLOOD_LUST)
    assert player.state.blood_lust 
    
    game.step(Action(ActionType.END_PLAYER_TURN)) 
    
    assert game.has_report_of_type(data[1] )
    assert game.has_report_of_type(OutcomeType.EJECTED_BY_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.EATEN_DURING_BLOOD_LUST) 
    
    assert player not in game.get_players_on_pitch(player.team)
    

    
    
def test_blood_lust_roll_fail_eat_thrall(): 

    data = (ActionType.START_MOVE, OutcomeType.MOVE_ACTION_STARTED) 

    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(5, 5))
    game.put(defender, Square(6, 5))
    game.state.teams[0].state.rerolls = 0
    game.state.teams[1].state.rerolls = 0
    
    thrall = team.players[3]
    game.put(thrall, Square(4, 4))
    assert not thrall.has_skill(Skill.BLOOD_LUST)
    
    
    D6.fix(1) #fail bloodlust
    D6.fix(1) #injury roll stun
    D6.fix(1) #injury roll stun
    
    
    game.step(Action(data[0], player=player))
    assert game.has_report_of_type(OutcomeType.FAILED_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_BLOOD_LUST)
    assert player.state.blood_lust 
    
    game.step(Action(ActionType.END_PLAYER_TURN)) 
    game.step(Action(ActionType.SELECT_PLAYER, position=Square(4, 4)) ) 
    
    
    #assert game.has_report_of_type(data[1] )
    assert not game.has_report_of_type(OutcomeType.EJECTED_BY_BLOOD_LUST)
    assert game.has_report_of_type(OutcomeType.EATEN_DURING_BLOOD_LUST) 
    
    assert player in game.get_players_on_pitch(player.team)
    
    
    
    
def test_blood_lust_fail_pass(): 

    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(5, 5))
    game.put(defender, Square(6, 5))
    game.state.teams[0].state.rerolls = 0
    game.state.teams[1].state.rerolls = 0
    
    catcher = team.players[3]
    catcher.extra_skills = [Skill.BLOOD_LUST]
    game.put(catcher, Square(4, 4))
    game.get_ball().move_to( Square(5, 5) )
    game.get_ball().is_carried = True 
    
    D6.fix(1) #fail bloodlust
    
    
    game.step(Action(ActionType.START_PASS, player=player))
    assert game.has_report_of_type(OutcomeType.FAILED_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_BLOOD_LUST)
    assert player.state.blood_lust 
    
    game.step(Action(ActionType.PASS, position = Square(4, 4) )) 
    
    assert game.has_report_of_type(OutcomeType.EJECTED_BY_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.EATEN_DURING_BLOOD_LUST) 
    
    assert player not in game.get_players_on_pitch(player.team)
    
    assert game.has_report_of_type(OutcomeType.BALL_BOUNCED)
    

def test_blood_lust_fail_handoff(): 

    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(5, 5))
    game.put(defender, Square(6, 5))
    game.state.teams[0].state.rerolls = 0
    game.state.teams[1].state.rerolls = 0
    
    catcher = team.players[3]
    catcher.extra_skills = [Skill.BLOOD_LUST]
    game.put(catcher, Square(4, 4))
    game.get_ball().move_to( Square(5, 5) )
    game.get_ball().is_carried = True 
    
    D6.fix(1) #fail bloodlust
    
    game.step(Action(ActionType.START_HANDOFF, player=player))
    assert game.has_report_of_type(OutcomeType.FAILED_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_BLOOD_LUST)
    assert player.state.blood_lust 
    
    game.step(Action(ActionType.HANDOFF, position=Square(4, 4)))
    
    assert game.has_report_of_type(OutcomeType.EJECTED_BY_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.EATEN_DURING_BLOOD_LUST) 
    
    assert player not in game.get_players_on_pitch(player.team)
    
    assert game.has_report_of_type(OutcomeType.BALL_BOUNCED)

def test_blood_lust_fail_touchdown(): 

    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(2, 2))
    game.state.teams[0].state.rerolls = 0
    game.state.teams[1].state.rerolls = 0
    
    game.get_ball().move_to( Square(2, 2) )
    game.get_ball().is_carried = True 
    
    D6.fix(1) #fail bloodlust
    
    game.step(Action(ActionType.START_MOVE, player=player))
    assert game.has_report_of_type(OutcomeType.FAILED_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_BLOOD_LUST)
    assert player.state.blood_lust 
    
    game.step(Action(ActionType.MOVE, position = Square(1, 2) )) 
    
    assert game.has_report_of_type(OutcomeType.EJECTED_BY_BLOOD_LUST)
    assert not game.has_report_of_type(OutcomeType.EATEN_DURING_BLOOD_LUST) 
    
    assert player not in game.get_players_on_pitch(player.team)
    
    assert game.has_report_of_type(OutcomeType.BALL_BOUNCED)
    assert not game.has_report_of_type(OutcomeType.TOUCHDOWN)        
    
    
    
    
    
    
    
    