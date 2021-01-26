import pytest
from tests.util import *



@pytest.mark.parametrize("reroll",  [True, False])
@pytest.mark.parametrize("success", [True, False])

def test_hypnotize(success, reroll):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    team.state.rerolls = 1*reroll  # ensure no reroll prompt

    players = game.get_players_on_pitch(team)
    player = players[0]
    player.extra_skills = [Skill.HYPNOTIC_GAZE]
    player.role.ag = 3 
    game.put(player, Square(2, 3))
    
    opponents = game.get_players_on_pitch(game.get_opp_team(team))
    opp_player = opponents[0] 
    opp_square = Square(2, 2)
    game.put(opp_player, opp_square)
    
    D6.FixedRolls.clear()
    
    if reroll: 
        D6.fix(1)
        
    if success: 
        D6.fix(6)
    else: 
        D6.fix(1)
        
        
    game.step(Action(ActionType.START_MOVE, player=player))
    
    actions = game.get_available_actions() 
    #pytest.set_trace() 
    
    game.step(Action(ActionType.HYPNOTIC_GAZE, position = opp_square))

    if reroll: 
        game.step(Action(ActionType.USE_REROLL, position = opp_square))
    
    # check that the player turn has ended
    assert player.state.used 
    
    if success: 
        assert opp_player.state.hypnotized 
        assert not opp_player.has_tackle_zone() 
        assert game.has_report_of_type(OutcomeType.SUCCESSFUL_HYPNOTIC_GAZE)
        if not reroll: 
            assert not game.has_report_of_type(OutcomeType.FAILED_HYPNOTIC_GAZE)
    else: 
        assert not opp_player.state.hypnotized 
        assert opp_player.has_tackle_zone() 
        assert game.has_report_of_type(OutcomeType.FAILED_HYPNOTIC_GAZE)
        assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_HYPNOTIC_GAZE)