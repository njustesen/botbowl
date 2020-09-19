from tests.util import *
import pytest

@pytest.mark.parametrize("data", [  (ActionType.START_FOUL, OutcomeType.FOUL_ACTION_STARTED), 
                                    (ActionType.START_MOVE, OutcomeType.MOVE_ACTION_STARTED), 
                                    (ActionType.START_PASS, OutcomeType.PASS_ACTION_STARTED),
                                    (ActionType.START_HANDOFF, OutcomeType.HANDOFF_ACTION_STARTED), 
                                    #(ActionType.START_BLOCK, OutcomeType.BLOCK_ACTION_STARTED),
                                    (ActionType.START_BLITZ, OutcomeType.BLITZ_ACTION_STARTED)]) 
def test_blood_lust_roll_success(data): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(5, 5))
    game.put(defender, Square(6, 5))
    

    D6.fix_result(2)
    
    game.step(Action(data[0], player=player))
    assert game.has_report_of_type(data[1] )
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_BLOOD_LUST)
    
@pytest.mark.parametrize("data", [  (ActionType.START_FOUL, OutcomeType.FOUL_ACTION_STARTED), 
                                    (ActionType.START_MOVE, OutcomeType.MOVE_ACTION_STARTED), 
                                    (ActionType.START_PASS, OutcomeType.PASS_ACTION_STARTED),
                                    (ActionType.START_HANDOFF, OutcomeType.HANDOFF_ACTION_STARTED), 
                                    #(ActionType.START_BLOCK, OutcomeType.BLOCK_ACTION_STARTED),
                                    (ActionType.START_BLITZ, OutcomeType.BLITZ_ACTION_STARTED)]) 
def test_blood_lust_roll_fail(data): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    other_team = game.get_opp_team(team)
    game.clear_board()
    player = team.players[2]
    player.extra_skills = [Skill.BLOOD_LUST]
    defender = other_team.players[2]
    game.put(player, Square(5, 5))
    game.put(defender, Square(6, 5))
    

    D6.fix_result(1)
    
    game.step(Action(data[0], player=player))
    assert game.has_report_of_type(data[1] )
    assert game.has_report_of_type(OutcomeType.FAILED_BLOOD_LUST)