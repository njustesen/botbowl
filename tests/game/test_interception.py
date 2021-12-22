from tests.util import *
import pytest


def test_interception_success():
    game, (passer, catcher, interceptor) = get_custom_game_turn(player_positions=[(1, 1), (10, 1)],
                                                                opp_player_positions=[(5, 1)],
                                                                ball_position=(1, 1))

    game.step(Action(ActionType.START_PASS, player=passer))
    game.step(Action(ActionType.PASS, position=catcher.position))

    # fix interception roll
    with only_fixed_rolls(game, d6=[6]):
        game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))

    assert game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.get_ball_carrier() == interceptor


def test_interception_safe_throw_success():
    game, (passer, catcher, interceptor) = get_custom_game_turn(player_positions=[(1, 1), (10, 1)],
                                                                opp_player_positions=[(5, 1)],
                                                                ball_position=(1, 1))
    passer.extra_skills.append(Skill.SAFE_THROW)

    game.step(Action(ActionType.START_PASS, player=passer))
    game.step(Action(ActionType.PASS, position=catcher.position))

    # fix interception roll, safe throw roll, pass roll, catch roll
    with only_fixed_rolls(game, d6=[6, 4, 6, 6]):
        game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))

    assert game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.has_report_of_type(OutcomeType.SKILL_USED)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_CATCH)
    assert game.get_ball_carrier() == catcher


def test_interception_safe_throw_fail():
    game, (passer, catcher, interceptor) = get_custom_game_turn(player_positions=[(1, 1), (10, 1)],
                                                                opp_player_positions=[(5, 1)],
                                                                ball_position=(1, 1),
                                                                rerolls=1)
    passer.extra_skills.append(Skill.SAFE_THROW)

    game.step(Action(ActionType.START_PASS, player=passer))
    game.step(Action(ActionType.PASS, position=catcher.position))

    # fix interception roll, safe throw roll, safe throw reroll
    with only_fixed_rolls(game, d6=[6, 3, 3]):
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
    D6.fix(6)  # Interception
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
    D6.fix(1)  # Interception
    D6.fix(6)  # Pass
    D6.fix(6)  # Catch
    game.step(Action(ActionType.PASS, position=catcher.position))
    game.step(Action(ActionType.SELECT_PLAYER, player=interceptor))
    assert not game.has_report_of_type(OutcomeType.INTERCEPTION)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_CATCH)
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
