from tests.util import *
import pytest

@pytest.mark.parametrize("armor_brake", [True, False]) 
@pytest.mark.parametrize("double_on_armor", [True, False])
@pytest.mark.parametrize("double_on_injury", [True, False]) 
@pytest.mark.parametrize("bribe_success", [True, False, None]) 

def test_foul_bribe(armor_brake, double_on_armor, double_on_injury, bribe_success):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    fouler = team.players[0]
    fouler.role.skills = []
    game.put(fouler, Square(1, 1))
    victim = game.get_opp_team(team).players[0]
    victim.state.up = False 
    game.put(victim, Square(1, 2))
    game.set_available_actions()
    game.state.reports.clear() 
    
    team.state.bribes = 1 if bribe_success is not None else 0 
    
    if armor_brake: 
        if double_on_armor:
            D6.fix_result(6)   
            D6.fix_result(6)   
        else: 
            D6.fix_result(5)
            D6.fix_result(6)
        
        if double_on_injury: 
            D6.fix_result(1)   
            D6.fix_result(1)
        else: 
            D6.fix_result(1)   
            D6.fix_result(2)
        
        
    else: #No armor brake 
        if double_on_armor:
            D6.fix_result(1)
            D6.fix_result(1)
        else: 
            D6.fix_result(1)
            D6.fix_result(2)
            
    if bribe_success: 
        D6.fix_result(2) 
    else: 
        D6.fix_result(1)
    
    game.step(Action(ActionType.START_FOUL, player=fouler))
    game.step(Action(ActionType.FOUL, position=victim.position, player=victim ))
    
    #No double roll - Referee did not spot the foul 
    if (not double_on_injury and not double_on_armor) or (not armor_brake and not double_on_armor): 
        assert fouler.position is not None  
        assert [r.outcome_type for r in game.state.reports].count(OutcomeType.PLAYER_EJECTED) == 0
        assert [r.outcome_type for r in game.state.reports].count(OutcomeType.TURNOVER) == 0
         
    elif bribe_success is not None : 
        if bribe_success: 
            assert fouler.position is not None  
            assert team.state.bribes == 0 
            
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.PLAYER_EJECTED) == 1
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_USED) == 1
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_SUCCESSFUL) == 1
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_FAILED) == 0
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.TURNOVER) == 0
        
        else: #bribe failed 
            assert fouler.position is None  
            assert team.state.bribes == 0 
        
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.PLAYER_EJECTED) == 2
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_USED) == 1
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_FAILED) == 1
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_SUCCESSFUL) == 0
            assert [r.outcome_type for r in game.state.reports].count(OutcomeType.TURNOVER) == 1
    else: #No bribe 
        assert fouler.position is None  
        assert [r.outcome_type for r in game.state.reports].count(OutcomeType.PLAYER_EJECTED) == 1
        assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_USED) == 0
        assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_FAILED) == 0
        assert [r.outcome_type for r in game.state.reports].count(OutcomeType.BRIBE_SUCCESSFUL) == 0
        assert [r.outcome_type for r in game.state.reports].count(OutcomeType.TURNOVER) == 1