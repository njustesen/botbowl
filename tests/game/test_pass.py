from tests.util import *
import pytest


@pytest.mark.parametrize("data", [
    ((1, 0), PassDistance.QUICK_PASS),
    ((3, 0), PassDistance.QUICK_PASS),
    ((4, 0), PassDistance.SHORT_PASS),
    ((6, 0), PassDistance.SHORT_PASS),
    ((7, 0), PassDistance.LONG_PASS),
    ((10, 0), PassDistance.LONG_PASS),
    ((11, 0), PassDistance.LONG_BOMB),
    ((13, 0), PassDistance.LONG_BOMB),
    ((14, 0), PassDistance.HAIL_MARY),
    ((1, 1), PassDistance.QUICK_PASS),
    ((2, 2), PassDistance.QUICK_PASS),
    ((3, 3), PassDistance.SHORT_PASS),
    ((4, 4), PassDistance.SHORT_PASS),
    ((5, 5), PassDistance.LONG_PASS),
    ((7, 7), PassDistance.LONG_PASS),
    ((8, 8), PassDistance.LONG_BOMB),
    ((9, 9), PassDistance.LONG_BOMB),
    ((10, 10), PassDistance.HAIL_MARY)
])
def test_pass_distances_and_modifiers(data):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    offset = data[0]
    asserted_pass_range = data[1]
    passer = team.players[0]
    passer.role.skills = []
    catcher = team.players[1]
    game.put(passer, Square(1, 1))
    game.state.weather = WeatherType.NICE
    catcher_position = Square(passer.position.x + offset[0], passer.position.y + offset[1])
    game.put(catcher, catcher_position)
    pass_distance = game.get_pass_distance(passer.position, catcher_position)
    assert pass_distance == asserted_pass_range
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    if pass_distance == PassDistance.QUICK_PASS:
        assert pass_mods == 1
    elif pass_distance == PassDistance.SHORT_PASS:
        assert pass_mods == 0
    elif pass_distance == PassDistance.LONG_PASS:
        assert pass_mods == -1
    elif pass_distance == PassDistance.LONG_BOMB:
        assert pass_mods == -2
    # Weather
    game.state.weather = WeatherType.VERY_SUNNY
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    if pass_distance == PassDistance.QUICK_PASS:
        assert pass_mods == 1 - 1
    elif pass_distance == PassDistance.SHORT_PASS:
        assert pass_mods == 0 - 1
    elif pass_distance == PassDistance.LONG_PASS:
        assert pass_mods == -1 - 1
    elif pass_distance == PassDistance.LONG_BOMB:
        assert pass_mods == -2 - 1
    game.state.weather = WeatherType.NICE
    # Tackle zone modifier
    opp_player = game.get_opp_team(team).players[0]
    game.put(opp_player, Square(1, 2))
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    if pass_distance == PassDistance.QUICK_PASS:
        assert pass_mods == 0
    elif pass_distance == PassDistance.SHORT_PASS:
        assert pass_mods == -1
    elif pass_distance == PassDistance.LONG_PASS:
        assert pass_mods == -2
    elif pass_distance == PassDistance.LONG_BOMB:
        assert pass_mods == -3


def test_accurate_pass():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 3
    catcher = team.players[1]
    game.put(passer, Square(1, 1))
    game.state.weather = WeatherType.NICE
    catcher_position = Square(passer.position.x + 7, passer.position.y + 0)
    game.put(catcher, catcher_position)
    pass_distance = game.get_pass_distance(passer.position, catcher.position)
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    passer.role.skills = [Skill.ACCURATE]
    accurate_pass_mods = game.get_pass_modifiers(passer, pass_distance)
    assert pass_mods + 1 == accurate_pass_mods


def test_strong_arm():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.ag = 3
    game.put(passer, Square(1, 1))
    game.state.weather = WeatherType.NICE
    for pass_distance in [PassDistance.QUICK_PASS, PassDistance.SHORT_PASS, PassDistance.LONG_PASS, PassDistance.LONG_BOMB, PassDistance.HAIL_MARY]:
        passer.role.skills = []
        pass_mods = game.get_pass_modifiers(passer, pass_distance)
        passer.role.skills = [Skill.STRONG_ARM]
        strong_arm_mods = game.get_pass_modifiers(passer, pass_distance)
        if pass_distance in [PassDistance.SHORT_PASS, PassDistance.LONG_PASS, PassDistance.LONG_BOMB]:
            assert pass_mods + 1 == strong_arm_mods
        else:
            assert pass_mods == strong_arm_mods


def test_pass_nerves_of_steel():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 3
    catcher = team.players[1]
    game.put(passer, Square(1, 1))
    opponent = game.get_opp_team(team).players[0]
    game.put(opponent, Square(1, 2))
    # one TZ on the thrower
    assert len(game.get_adjacent_opponents(passer, down=False, stunned=False)) == 1

    game.state.weather = WeatherType.NICE
    catcher_position = Square(passer.position.x + 7, passer.position.y + 0)
    game.put(catcher, catcher_position)
    pass_distance = game.get_pass_distance(passer.position, catcher.position)
    pass_mods = game.get_pass_modifiers(passer, pass_distance)
    passer.role.skills = [Skill.NERVES_OF_STEEL]
    nos_pass_mods = game.get_pass_modifiers(passer, pass_distance)
    # nos removes the 1 TZ impact
    assert pass_mods + 1 == nos_pass_mods

@pytest.mark.parametrize("pass_skill", [True, False]) 
def test_pass_roll_fumble(pass_skill): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    
    passer = team.players[0]
    passer.role.skills = [Skill.PASS] if pass_skill else [] 
    passer.role.ag = 3
    game.put(passer, Square(2, 2))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    
    catcher = team.players[1]
    catcher_position = Square(passer.position.x + 6, passer.position.y + 0)
    game.put(catcher, catcher_position)
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    D6.fix_result(1)  # Fumble pass
    D6.fix_result(6)  # Successful pass after skill re-roll
    
    if pass_skill: 
        assert passer.can_use_skill(Skill.PASS)
    
    game.step(Action(ActionType.START_PASS, player=passer))
    game.step(Action(ActionType.PASS, position=catcher_position ))
    
    if pass_skill:
        assert game.has_report_of_type(OutcomeType.FUMBLE)
        assert game.has_report_of_type(OutcomeType.ACCURATE_PASS)
        assert game.has_report_of_type(OutcomeType.SKILL_USED)
    else:
        assert game.has_report_of_type(OutcomeType.FUMBLE)
        assert game.has_report_of_type(OutcomeType.BALL_BOUNCED)
        assert game.has_report_of_type(OutcomeType.BALL_ON_GROUND)
        assert not game.has_report_of_type(OutcomeType.ACCURATE_PASS)
        assert not game.has_report_of_type(OutcomeType.SKILL_USED)
        
@pytest.mark.parametrize("pass_skill", [True, False]) 
def test_pass_roll_inaccurate(pass_skill): 
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    game.state.weather = WeatherType.NICE
    game.state.teams[0].state.rerolls = 0
    
    passer = team.players[0]
    passer.role.skills = [Skill.PASS] if pass_skill else [] 
    passer.role.ag = 3
    game.put(passer, Square(5, 5))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    
    catcher = team.players[1]
    catcher_position = Square(passer.position.x + 3, passer.position.y + 0)
    game.put(catcher, catcher_position)
    
    game.set_available_actions()
    game.state.reports.clear() 
    
    D6.fix_result(2)  # Inaccurate pass
    D6.fix_result(6)  # Successful pass after skill re-roll
    
    if pass_skill: 
        assert passer.can_use_skill(Skill.PASS)
    
    game.step(Action(ActionType.START_PASS, player=passer))
    game.step(Action(ActionType.PASS, position=catcher_position ))
    
    if pass_skill:
        assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)
        assert game.has_report_of_type(OutcomeType.ACCURATE_PASS)
        assert game.has_report_of_type(OutcomeType.SKILL_USED)
        assert not game.has_report_of_type(OutcomeType.FUMBLE)
    else:
        assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)
        assert game.has_report_of_type(OutcomeType.BALL_SCATTER)
        assert not game.has_report_of_type(OutcomeType.ACCURATE_PASS)
        assert not game.has_report_of_type(OutcomeType.FUMBLE)
        assert not game.has_report_of_type(OutcomeType.SKILL_USED)