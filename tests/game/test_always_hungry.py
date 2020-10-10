from tests.util import *
import pytest

from pytest import set_trace 

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
    
    