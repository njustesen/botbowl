from tests.util import *
import pytest


@pytest.mark.parametrize("weather", [WeatherType.SWELTERING_HEAT, WeatherType.VERY_SUNNY, WeatherType.NICE, WeatherType.POURING_RAIN, WeatherType.BLIZZARD])
def test_catch_modifiers(weather):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    catcher = team.players[1]
    catcher.role.skills = []
    game.put(passer, Square(1, 1))
    game.put(catcher, Square(10, 1))
    game.state.weather = weather
    weather_mod = -1 if weather == WeatherType.POURING_RAIN else 0
    for accurate in [True, False]:
        mods = game.get_catch_modifiers(catcher, accurate=accurate)
        if accurate:
            assert mods == 1 + weather_mod
        else:
            assert mods == 0 + weather_mod
    for handoff in [True, False]:
        mods = game.get_catch_modifiers(catcher, handoff=handoff)
        if handoff:
            assert mods == 1 + weather_mod
        else:
            assert mods == 0 + weather_mod
    # Tackle zone modifier
    opp_player = game.get_opp_team(team).players[0]
    game.put(opp_player, Square(10, 2))
    mods = game.get_catch_modifiers(catcher, accurate=True)
    assert mods == 0 + weather_mod
    mods = game.get_catch_modifiers(catcher, handoff=True)
    assert mods == 0 + weather_mod


def test_catch_team_reroll():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.state.weather = WeatherType.NICE
    game.clear_board()
    passer = team.players[0]
    catcher = team.players[1]
    catcher.role.skills = []
    game.put(passer, Square(1, 1))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    game.put(catcher, Square(10, 1))
    D6.fix_result(6)  # Second catch roll
    D6.fix_result(1)  # First catch roll
    D6.fix_result(6)  # Pass roll
    game.step(Action(ActionType.START_PASS, player=passer))
    game.step(Action(ActionType.PASS, position=catcher.position))
    assert game.has_report_of_type(OutcomeType.CATCH_FAILED)
    game.step(Action(ActionType.USE_REROLL))
    assert game.has_report_of_type(OutcomeType.REROLL_USED)
    assert game.has_report_of_type(OutcomeType.CATCH)


def test_catch_skill_reroll():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.state.weather = WeatherType.NICE
    game.clear_board()
    passer = team.players[0]
    catcher = team.players[1]
    catcher.role.skills = [Skill.CATCH]
    game.put(passer, Square(1, 1))
    game.get_ball().move_to(passer.position)
    game.get_ball().is_carried = True
    game.put(catcher, Square(10, 1))
    D6.fix_result(6)  # Second catch roll
    D6.fix_result(1)  # First catch roll
    D6.fix_result(6)  # Pass roll
    game.step(Action(ActionType.START_PASS, player=passer))
    game.step(Action(ActionType.PASS, position=catcher.position))
    assert game.has_report_of_type(OutcomeType.CATCH_FAILED)
    assert game.has_report_of_type(OutcomeType.SKILL_USED)
    assert game.has_report_of_type(OutcomeType.CATCH)

