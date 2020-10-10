from tests.util import *
import pytest

from pytest import set_trace 

def test_throw_TM_success():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    
    thrower = team.players[0]
    thrower.extra_skills = [Skill.THROW_TEAM_MATE] 
    #thrower.extra_skills.append(Skill.ALWAYS_HUNGRY]
    game.put(thrower, Square(1, 1))
    
    thrown_player = team.players[1]
    thrown_player.extra_skills = [Skill.RIGHT_STUFF]
    game.put(thrown_player, Square(1, 2))
    game.get_ball().move_to(thrown_player.position)
    game.get_ball().is_carried = True
    
    game.set_available_actions() 
    game.state.reports.clear() 
    
    D6.fix_result(6) #always hungry success
    D6.fix_result(6) # throw 
    D8.fix_result(5) # scatter
    D8.fix_result(5)
    D8.fix_result(5)
    
    D6.fix_result(6) # landing 
    
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert ActionType.SELECT_PLAYER in [a.action_type for a in game.get_available_actions()] 
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert not game.is_pass_available()
    
    game.step(Action(ActionType.PASS, position=Square(4,4) ))
    assert thrown_player.state.up 
    assert not thrown_player.state.used
    assert thrown_player.position == Square(7,4)
    assert game.get_ball().position == Square(7,4)
    
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert game.has_report_of_type(OutcomeType.TEAM_MATE_THROWN)
    assert game.has_report_of_type(OutcomeType.THROWN_TEAM_MATE_SCATTER)
    assert game.has_report_of_type(OutcomeType.RIGHT_STUFF_LANDING_SUCCESS)
    
    
@pytest.mark.parametrize("break_armor", [True, False]) 
def test_throw_TM_failed_landing(break_armor):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    
    thrower = team.players[0]
    thrower.extra_skills = [Skill.THROW_TEAM_MATE] 
    #thrower.extra_skills.append(Skill.ALWAYS_HUNGRY]
    game.put(thrower, Square(1, 1))
    
    thrown_player = team.players[1]
    thrown_player.extra_skills = [Skill.RIGHT_STUFF]
    game.put(thrown_player, Square(1, 2))
    game.get_ball().move_to(thrown_player.position)
    game.get_ball().is_carried = True

    game.set_available_actions() 
    game.state.reports.clear() 
    
    #D6.fix_result(2) #always hungry success
    D6.fix_result(6) # throw 
    
    D8.fix_result(5) # scatter 
    D8.fix_result(5)
    D8.fix_result(5)
    
    D6.fix_result(1) # landing 
    
    D8.fix_result(5) #ball bounce  
    
    if break_armor: 
        D6.fix_result(6) # armor 1
        D6.fix_result(6) # armor 2 
        D6.fix_result(1) # injury 1
        D6.fix_result(1) # injury 2 
    else: 
        D6.fix_result(1) # armor 1
        D6.fix_result(1) # armor 2 
        
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert ActionType.SELECT_PLAYER in [a.action_type for a in game.get_available_actions()] 
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert not game.is_pass_available()
    
    game.step(Action(ActionType.PASS, position=Square(4,4) ))
    
    assert thrown_player.position == Square(7,4)
    assert game.get_ball().position == Square(8,4)
    assert not thrown_player.state.up 
    if break_armor: 
        assert thrown_player.state.stunned
    else: 
        assert not thrown_player.state.used  
    
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert game.has_report_of_type(OutcomeType.TEAM_MATE_THROWN)
    assert game.has_report_of_type(OutcomeType.THROWN_TEAM_MATE_SCATTER)
    assert game.has_report_of_type(OutcomeType.RIGHT_STUFF_LANDING_FAILURE)
    
@pytest.mark.parametrize("with_ball", [True, False]) 
def test_throw_TM_land_on_multiple_players(with_ball):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    
    thrower = team.players[0]
    thrower.extra_skills = [Skill.THROW_TEAM_MATE] 
    #thrower.extra_skills.append(Skill.ALWAYS_HUNGRY]
    game.put(thrower, Square(1, 1))
    
    thrown_player = team.players[1]
    thrown_player.extra_skills = [Skill.RIGHT_STUFF]
    game.put(thrown_player, Square(1, 2))
    if with_ball:
        game.get_ball().move_to(thrown_player.position)
        game.get_ball().is_carried = True
    
    victim1 = game.state.teams[1].players[0]
    game.put(victim1, Square(7, 4))
    
    victim2 = game.state.teams[1].players[1]
    game.put(victim2, Square(8, 4))
    
    game.set_available_actions() 
    game.state.reports.clear() 
    
    D6.fix_result(6) # throw 
    
    D8.fix_result(5) # scatter
    D8.fix_result(5)
    D8.fix_result(5)
    D8.fix_result(5) 
    D8.fix_result(5)
    
    D6.fix_result(1) #armor 
    D6.fix_result(1) #armor
    D6.fix_result(1) #armor
    D6.fix_result(1) #armor
    
    D8.fix_result(5) # bounce 
    D8.fix_result(5) # bounce 
    D8.fix_result(5) # bounce 
    
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert ActionType.SELECT_PLAYER in [a.action_type for a in game.get_available_actions()] 
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert not game.is_pass_available()
    
    game.step(Action(ActionType.PASS, position=Square(4,4) ))
    assert not thrown_player.state.up 
    assert not victim1.state.up
    assert victim2.state.up
    
    assert thrown_player.position == Square(9,4)
    if with_ball: assert game.get_ball().position == Square(10,4)
    
    assert game.has_report_of_type(OutcomeType.THROWN_TEAM_MATE_HIT_PLAYER) 
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert game.has_report_of_type(OutcomeType.TEAM_MATE_THROWN)
    assert game.has_report_of_type(OutcomeType.THROWN_TEAM_MATE_SCATTER)
    #assert game.has_report_of_type(OutcomeType.RIGHT_STUFF_LANDING_SUCCESS)
    
    
    
def test_throw_TM_land_on_own_player(): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    
    thrower = team.players[0]
    thrower.extra_skills = [Skill.THROW_TEAM_MATE] 
    #thrower.extra_skills.append(Skill.ALWAYS_HUNGRY]
    game.put(thrower, Square(1, 1))
    
    thrown_player = team.players[1]
    thrown_player.extra_skills = [Skill.RIGHT_STUFF]
    game.put(thrown_player, Square(1, 2))
    
    victim1 = team.players[2]
    game.put(victim1, Square(7, 4))
        
    game.set_available_actions() 
    game.state.reports.clear() 
    
    D6.fix_result(6) # throw 
    
    D8.fix_result(5) # scatter
    D8.fix_result(5)
    D8.fix_result(5)
    D8.fix_result(5)
    D8.fix_result(5)
    
    
    D6.fix_result(1) #armor 
    D6.fix_result(1) #armor
    D6.fix_result(1) #armor
    D6.fix_result(1) #armor
    
    D8.fix_result(5) # bounce 
    D8.fix_result(5) # bounce 
    D8.fix_result(5) # bounce 
    
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert ActionType.SELECT_PLAYER in [a.action_type for a in game.get_available_actions()] 
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert not game.is_pass_available()
    
    game.step(Action(ActionType.PASS, position=Square(4,4) ))
    assert not thrown_player.state.up 
    assert not victim1.state.up
    
    assert thrown_player.position == Square(8,4)
    
    assert game.has_report_of_type(OutcomeType.THROWN_TEAM_MATE_HIT_PLAYER) 
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert game.has_report_of_type(OutcomeType.TEAM_MATE_THROWN)
    assert game.has_report_of_type(OutcomeType.THROWN_TEAM_MATE_SCATTER)
    #assert game.has_report_of_type(OutcomeType.RIGHT_STUFF_LANDING_SUCCESS)
    assert game.has_report_of_type(OutcomeType.TURNOVER)
    
def test_Always_Hungry(): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    
    thrower = team.players[0]
    thrower.extra_skills = [Skill.THROW_TEAM_MATE] 
    thrower.extra_skills.append(Skill.ALWAYS_HUNGRY) 
    game.put(thrower, Square(1, 1))
    
    thrown_player = team.players[1]
    thrown_player.extra_skills = [Skill.RIGHT_STUFF]
    game.put(thrown_player, Square(1, 2))
    game.get_ball().move_to(thrown_player.position)
    game.get_ball().is_carried = True
    
    game.set_available_actions() 
    game.state.reports.clear() 
    
    D6.fix_result(6) #always hungry success
    D6.fix_result(6) # throw 
    D8.fix_result(5) # scatter
    D8.fix_result(5)
    D8.fix_result(5)
    
    D6.fix_result(6) # landing 
    
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert ActionType.SELECT_PLAYER in [a.action_type for a in game.get_available_actions()] 
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert not game.is_pass_available()
    
    game.step(Action(ActionType.PASS, position=Square(4,4) ))
    assert thrown_player.state.up 
    assert not thrown_player.state.used
    
    #set_trace() 
    assert thrown_player.position == Square(7,4)
    assert game.get_ball().position == Square(7,4)
    
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert game.has_report_of_type(OutcomeType.TEAM_MATE_THROWN)
    assert game.has_report_of_type(OutcomeType.THROWN_TEAM_MATE_SCATTER)
    assert game.has_report_of_type(OutcomeType.RIGHT_STUFF_LANDING_SUCCESS)
    
    