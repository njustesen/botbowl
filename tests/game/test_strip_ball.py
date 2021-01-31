from tests.util import *
import pytest

@pytest.mark.parametrize("sure_hands", [True, False]) 
def test_strip_ball(sure_hands): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    
    blocker = team.players[0]
    blocker.extra_skills = [Skill.STRIP_BALL]
    game.put(blocker, Square(5, 6))
    
    victim = opp_team.players[0]
    victim.extra_skills = [] if not sure_hands else [Skill.SURE_HANDS]
    game.put(victim, Square(5, 5))
    game.get_ball().move_to(victim.position)
    game.get_ball().is_carried = True
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    D8.fix(2)
    
    game.step(Action(ActionType.START_BLOCK, player=blocker))
    game.step(Action(ActionType.BLOCK, position=victim.position ))
    game.step(Action(ActionType.SELECT_PUSH ))
    game.step(Action(ActionType.PUSH, position=Square(5,4)))
    game.step(Action(ActionType.FOLLOW_UP, position=Square(5,6))) 
    
    if not sure_hands: 
        assert game.has_report_of_type(OutcomeType.FUMBLE)
        assert game.get_ball().position == Square(5,3)
    else: 
        assert not game.has_report_of_type(OutcomeType.FUMBLE)
        assert game.get_ball().position == Square(5,4)
        assert game.get_ball().is_carried 

def test_strip_ball_taken_root(): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    
    blocker = team.players[0]
    blocker.extra_skills = [Skill.STRIP_BALL]
    game.put(blocker, Square(5, 6))
    
    victim = opp_team.players[0]
    victim.extra_skills = [Skill.TAKE_ROOT]
    victim.state.taken_root = True 
    game.put(victim, Square(5, 5))
    game.get_ball().move_to(victim.position)
    game.get_ball().is_carried = True
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    D8.fix(2)
    
    game.step(Action(ActionType.START_BLOCK, player=blocker))
    game.step(Action(ActionType.BLOCK, position=victim.position ))
    game.step(Action(ActionType.SELECT_PUSH ))
    #game.step(Action(ActionType.PUSH, position=Square(5,4)))
    #game.step(Action(ActionType.FOLLOW_UP, position=Square(5,6))) 
    
    assert game.has_report_of_type(OutcomeType.FUMBLE)
    assert game.get_ball().position == Square(5,4)
    assert victim.position == Square(5,5)
    
    
def test_strip_ball_stand_firm(): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    
    blocker = team.players[0]
    blocker.extra_skills = [Skill.STRIP_BALL]
    game.put(blocker, Square(5, 6))
    
    victim = opp_team.players[0]
    victim.extra_skills = [Skill.STAND_FIRM]
    game.put(victim, Square(5, 5))
    game.get_ball().move_to(victim.position)
    game.get_ball().is_carried = True
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    D8.fix(2)
    
    game.step(Action(ActionType.START_BLOCK, player=blocker))
    game.step(Action(ActionType.BLOCK, position=victim.position ))
    game.step(Action(ActionType.SELECT_PUSH ))
    game.step(Action(ActionType.USE_SKILL ))
    #game.step(Action(ActionType.PUSH, position=Square(5,4)))
    #game.step(Action(ActionType.FOLLOW_UP, position=Square(5,6))) 
    
    assert game.has_report_of_type(OutcomeType.FUMBLE)
    assert game.get_ball().position == Square(5,4)
    assert victim.position == Square(5,5)
    
        
def test_strip_ball_chain_push_sequence(): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    
    blocker = team.players[0]
    blocker.extra_skills = [Skill.STRIP_BALL]
    game.put(blocker, Square(5, 6))
    
    victim = opp_team.players[0]
    game.put(victim, Square(5, 5))
    game.get_ball().move_to(victim.position)
    game.get_ball().is_carried = True
    
    # Setup chain push
    game.put(opp_team.players[1], Square(5, 4))
    game.put(opp_team.players[2], Square(4, 4))
    game.put(opp_team.players[3], Square(6, 4))
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    BBDie.clear_fixes()
    BBDie.fix(BBDieResult.PUSH)
    BBDie.fix(BBDieResult.PUSH)
    D8.fix(7) # scatter to blocker
    D6.fix(6) # blocker's catch roll success
    
    game.step(Action(ActionType.START_BLOCK, player=blocker))
    game.step(Action(ActionType.BLOCK, position=victim.position ))
    game.step(Action(ActionType.SELECT_PUSH ))
    game.step(Action(ActionType.PUSH, position=Square(5,4)))
    game.step(Action(ActionType.PUSH, position=Square(5,3)))
    game.step(Action(ActionType.FOLLOW_UP, position=Square(5,5))) 
    
    assert game.has_report_of_type(OutcomeType.FUMBLE)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_CATCH)
    assert game.get_ball().position == Square(5,5)
    assert blocker.position == Square(5,5)

