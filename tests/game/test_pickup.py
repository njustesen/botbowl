from tests.util import *
import pytest


@pytest.mark.parametrize("weather", [WeatherType.SWELTERING_HEAT, WeatherType.VERY_SUNNY, WeatherType.NICE, WeatherType.POURING_RAIN, WeatherType.BLIZZARD])
def test_pickup_modifiers(weather):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    player = team.players[0]
    player.role.skills = []
    player.role.ag = 3
    game.put(player, Square(1, 1))
    game.get_ball().move_to(player.position)
    game.get_ball().is_carried = False
    game.state.weather = weather
    weather_mod = -1 if weather == WeatherType.POURING_RAIN else 0
    mods = game.get_pickup_modifiers(player, player.position)
    assert mods == 1 + weather_mod
    # Big hand mod
    player.role.skills = [Skill.BIG_HAND]
    mods = game.get_pickup_modifiers(player, player.position)
    assert mods == 1
    player.role.skills = []
    # Extra arms
    player.role.skills = [Skill.EXTRA_ARMS]
    mods = game.get_pickup_modifiers(player, player.position)
    assert mods == 2 + weather_mod
    player.role.skills = []
    # Tackle zone modifier
    opp_player = game.get_opp_team(team).players[0]
    game.put(opp_player, Square(2, 2))
    mods = game.get_pickup_modifiers(player, player.position)
    assert mods == 0 + weather_mod
    opp_player_2 = game.get_opp_team(team).players[1]
    game.put(opp_player_2, Square(1, 2))
    mods = game.get_pickup_modifiers(player, player.position)
    assert mods == - 1 + weather_mod
    # Big hand
    player.role.skills = [Skill.BIG_HAND]
    mods = game.get_pickup_modifiers(player, player.position)
    assert mods == 1


@pytest.mark.parametrize("sure_hands", [True, False])
def test_pickup_sure_hands(sure_hands):
    game, (player, ) = get_custom_game_turn(player_positions=[(1, 1)],
                                            ball_position=(2, 2))
    if sure_hands:
        player.extra_skills.append(Skill.SURE_HANDS)
    assert player.has_skill(Skill.SURE_HANDS) == sure_hands

    game.step(Action(ActionType.START_MOVE, player=player))
    D6.fix(1)  # Failed pickup
    D6.fix(6)  # Successful pickup
    game.step(Action(ActionType.MOVE, position=Square(2, 2)))

    if sure_hands:
        assert game.has_report_of_type(OutcomeType.FAILED_PICKUP)
        assert game.has_report_of_type(OutcomeType.SUCCESSFUL_PICKUP)
        assert game.has_report_of_type(OutcomeType.SKILL_USED)
    else:
        assert game.has_report_of_type(OutcomeType.FAILED_PICKUP)
        assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_PICKUP)
        assert not game.has_report_of_type(OutcomeType.SKILL_USED)



