from tests.util import *
import pytest

from pytest import set_trace 

def test_Always_Hungry_success(): 
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
    
    game.set_available_actions() 
    game.state.reports.clear() 
    
    D6.fix_result(2) #always hungry success
    
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert ActionType.SELECT_PLAYER in [a.action_type for a in game.get_available_actions()] 
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert game.has_report_of_type(OutcomeType.ALWAYS_HUNGRY_SUCCESS)
    assert not game.has_report_of_type(OutcomeType.ALWAYS_HUNGRY_FAILURE)
    assert not game.has_report_of_type(OutcomeType.ESCAPED_BEING_EATEN)
    assert not game.has_report_of_type(OutcomeType.EATEN_BY_TEAM_MATE)
    
    
def test_Always_Hungry_fail_Escape_eaten_success(): 
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
    
    game.set_available_actions() 
    game.state.reports.clear() 
    
    D6.fix_result(1) #always hungry success
    D6.fix_result(2) # escape being eaten success 
    D6.fix_result(6) # right stuff landing 
    
    
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    assert ActionType.SELECT_PLAYER in [a.action_type for a in game.get_available_actions()] 
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert not game.has_report_of_type(OutcomeType.ALWAYS_HUNGRY_SUCCESS)
    assert game.has_report_of_type(OutcomeType.ALWAYS_HUNGRY_FAILURE)
    assert game.has_report_of_type(OutcomeType.ESCAPED_BEING_EATEN)
    assert not game.has_report_of_type(OutcomeType.EATEN_BY_TEAM_MATE)
    
    assert thrown_player.position == Square(1,2)

@pytest.mark.parametrize("with_ball", [True, False]) 
def test_Always_Hungry_fail_Escape_eaten_fail(with_ball): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    game.state.teams[0].state.apothecaries = 3
    
    
    thrower = team.players[0]
    thrower.extra_skills = [Skill.THROW_TEAM_MATE] 
    thrower.extra_skills.append(Skill.ALWAYS_HUNGRY) 
    game.put(thrower, Square(1, 1))
    
    thrown_player = team.players[1]
    thrown_player.extra_skills = [Skill.RIGHT_STUFF]
    game.put(thrown_player, Square(1, 2))
    if with_ball: 
        game.get_ball().move_to(thrown_player.position)
        game.get_ball().is_carried = True
    
    game.set_available_actions() 
    game.state.reports.clear() 
    
    D6.fix_result(1) #always hungry success
    D6.fix_result(1) # escape being eaten success 
    D8.fix_result(5) # injury roll which is actually not used 
    D8.fix_result(5) # ball bounce
    
    
    game.step(Action(ActionType.START_PASS, player=thrower))
    assert game.has_report_of_type(OutcomeType.PASS_ACTION_STARTED)
    
    game.step(Action(ActionType.SELECT_PLAYER, position=thrown_player.position ))
    assert not game.has_report_of_type(OutcomeType.ALWAYS_HUNGRY_SUCCESS)
    assert game.has_report_of_type(OutcomeType.ALWAYS_HUNGRY_FAILURE)
    assert not game.has_report_of_type(OutcomeType.ESCAPED_BEING_EATEN)
    assert game.has_report_of_type(OutcomeType.EATEN_BY_TEAM_MATE)
    
    assert game.has_report_of_type(OutcomeType.DEAD)
    assert thrown_player.position is None 
    assert not (ActionType.USE_APOTHECARY in [a.action_type for a in game.get_available_actions()] ) 
    
    
    if with_ball: 
        assert game.get_ball().position == Square(2,2)
