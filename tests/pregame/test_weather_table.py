import pytest
from botbowl.core.game import *
from tests.util import *


weather_dice_rolls = []
for i in range(1, 7, 1):
    for j in range(1, 7, 1):
        weather_dice_rolls.append([i, j])


@pytest.mark.parametrize("dice_roll", weather_dice_rolls)
def test_weather_table(dice_roll):
    game = get_game_weather_table()
    proc = WeatherTable(game)
    proc.start()
    D6.fix(dice_roll[0])
    D6.fix(dice_roll[1])
    proc.step(None)
    if np.sum(dice_roll) == 2:
        assert game.state.weather == WeatherType.SWELTERING_HEAT
    elif np.sum(dice_roll) == 3:
        assert game.state.weather == WeatherType.VERY_SUNNY
    elif 4 <= np.sum(dice_roll) <= 10:
        assert game.state.weather == WeatherType.NICE
        assert not game.has_report_of_type(OutcomeType.GENTLE_GUST_IN_BOUNDS)
    elif np.sum(dice_roll) == 11:
        assert game.state.weather == WeatherType.POURING_RAIN
    elif np.sum(dice_roll) == 12:
        assert game.state.weather == WeatherType.BLIZZARD
