from tests.util import *
import pytest

from pytest import set_trace 

@pytest.mark.parametrize("nbr_of_teammates", [0,1,2,3]) 
def test_pass_roll_inaccurate(nbr_of_teammates): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 1

    player = team.players[0]
    player.role.skills = [Skill.TIMMMBER]
    player.role.ma = 2
    game.put(player, Square(2, 2) )
    player.state.up = False 
    
    if nbr_of_teammates > 0: 
        for i in range(nbr_of_teammates+1): 
            game.put(team.players[i+1], Square(1,i) )
        
    game.set_available_actions()
    game.state.reports.clear() 
    
    target = max(2, 4 - nbr_of_teammates) 
    
    D6.fix(target - 1)  # Fail first,
    D6.fix(target)  # re-roll success next

    game.step(Action(ActionType.START_MOVE, player=player))
    for action in game.get_available_actions(): 
        if action.action_type == ActionType.STAND_UP: 
            assert action.rolls == [[target]]
            break 
    
    game.step(Action(ActionType.STAND_UP, player=player))
    assert game.has_report_of_type(OutcomeType.FAILED_STAND_UP)
    
    game.step(Action(ActionType.USE_REROLL, player=player))
    assert game.has_report_of_type(OutcomeType.STAND_UP)
