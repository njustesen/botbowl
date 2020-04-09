from tests.util import *
import pytest


def test_interception_success():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    catcher = team.players[1]
    catcher.role.skills = []
    interceptor = opp_team.players[0]
    interceptor.role.skills = []
    interceptor.role.ag = 3
    game.put(passer, Square(1, 1))
    game.put(interceptor, Square(5, 1))
    game.put(catcher, Square(10, 1))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    game.state.weather = WeatherType.NICE
    game.set_available_actions()
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix_result(6)  # Interception
    game.step(Action(ActionType.PASS, position=catcher.position))
    game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))
    assert game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.get_ball_carrier() == interceptor


def test_interception_safe_throw_success():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    passer = team.players[0]
    passer.extra_skills = [Skill.SAFE_THROW]
    catcher = team.players[1]
    catcher.extra_skills = []
    interceptor = opp_team.players[0]
    interceptor.role.ag = 3
    interceptor.extra_skills = []
    game.put(passer, Square(1, 1))
    game.put(interceptor, Square(5, 1))
    game.put(catcher, Square(10, 1))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    game.state.weather = WeatherType.NICE
    game.set_available_actions()
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix_result(6)  # Interception
    D6.fix_result(4)  # Safe throw agility roll
    D6.fix_result(6)  # Pass
    D6.fix_result(6)  # Catch
    game.step(Action(ActionType.PASS, position=catcher.position))
    game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))
    assert game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.has_report_of_type(OutcomeType.SKILL_USED)
    assert game.has_report_of_type(OutcomeType.CATCH)
    assert game.get_ball_carrier() == catcher


def test_interception_safe_throw_fail():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    passer = team.players[0]
    passer.extra_skills = [Skill.SAFE_THROW]
    catcher = team.players[1]
    catcher.extra_skills = []
    interceptor = opp_team.players[0]
    interceptor.role.ag = 3
    interceptor.extra_skills = []
    game.put(passer, Square(1, 1))
    game.put(interceptor, Square(5, 1))
    game.put(catcher, Square(10, 1))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    game.state.weather = WeatherType.NICE
    game.set_available_actions()
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix_result(6)  # Interception
    D6.fix_result(3)  # Safe throw agility roll
    D6.fix_result(3)  # Safe throw agility re-roll
    game.step(Action(ActionType.PASS, position=catcher.position))
    game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))
    game.step(Action(ActionType.USE_REROLL))
    assert game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.has_report_of_type(OutcomeType.SKILL_USED)
    assert game.get_ball_carrier() == interceptor


def test_interception_safe_throw_very_long_legs_fail():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    passer = team.players[0]
    passer.extra_skills = [Skill.SAFE_THROW]
    catcher = team.players[1]
    catcher.extra_skills = []
    interceptor = opp_team.players[0]
    interceptor.role.ag = 3
    interceptor.extra_skills = [Skill.VERY_LONG_LEGS]
    game.put(passer, Square(1, 1))
    game.put(interceptor, Square(5, 1))
    game.put(catcher, Square(10, 1))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    game.state.weather = WeatherType.NICE
    game.set_available_actions()
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix_result(6)  # Interception
    game.step(Action(ActionType.PASS, position=catcher.position))
    game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))
    assert game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.has_report_of_type(OutcomeType.SKILL_USED)
    assert game.get_ball_carrier() == interceptor


def test_interception_fail():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    catcher = team.players[1]
    catcher.role.skills = []
    interceptor = opp_team.players[0]
    interceptor.role.skills = []
    interceptor.role.ag = 3
    game.put(passer, Square(1, 1))
    game.put(interceptor, Square(5, 1))
    game.put(catcher, Square(10, 1))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    game.state.weather = WeatherType.NICE
    game.set_available_actions()
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix_result(1)  # Interception
    D6.fix_result(6)  # Pass
    D6.fix_result(6)  # Catch
    game.step(Action(ActionType.PASS, position=catcher.position))
    game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))
    assert not game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.has_report_of_type(OutcomeType.CATCH)
    assert game.get_ball_carrier() == catcher


def test_interception_modifiers():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    opp_team = game.get_opp_team(team)
    game.clear_board()
    catcher = team.players[1]
    catcher.extra_skills = []
    interceptor = opp_team.players[0]
    interceptor.extra_skills = []
    interceptor.role.ag = 3
    game.put(interceptor, Square(5, 1))
    game.state.weather = WeatherType.NICE
    mod = game.get_catch_modifiers(interceptor, interception=True)
    assert mod == -2
    interceptor.extra_skills = [Skill.VERY_LONG_LEGS]
    mod = game.get_catch_modifiers(interceptor, interception=True)
    assert mod == -1
    interceptor.extra_skills = [Skill.EXTRA_ARMS]
    mod = game.get_catch_modifiers(interceptor, interception=True)
    assert mod == -1
