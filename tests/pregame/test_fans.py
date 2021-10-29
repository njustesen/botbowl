import pytest
from botbowl.core.game import *
from tests.util import *


fans_dice_rolls = []
for i in range(1, 7, 1):
    for j in range(1, 7, 1):
        fans_dice_rolls.append([i, j])


@pytest.mark.parametrize("dice_roll", fans_dice_rolls)
def test_spectators(dice_roll):
    game = get_game_fans()
    proc = Fans(game)
    proc.start()
    D6.fix(dice_roll[0])
    D6.fix(dice_roll[1])
    D6.fix(dice_roll[0])
    D6.fix(dice_roll[1])
    proc.step(None)
    assert game.has_report_of_type(OutcomeType.TEAM_SPECTATORS)
    for report in game.state.reports:
        if report.outcome_type == OutcomeType.TEAM_SPECTATORS:
            assert report.team == game.state.home_team or report.team == game.state.away_team
            assert report.n == (dice_roll[0] + dice_roll[1] + report.team.fan_factor) * 1000
    assert game.has_report_of_type(OutcomeType.SPECTATORS)
    for report in game.state.reports:
        if report.outcome_type == OutcomeType.SPECTATORS:
            assert report.n == (dice_roll[0] + dice_roll[1] + game.state.home_team.fan_factor) * 1000 + (dice_roll[0] + dice_roll[1] + game.state.away_team.fan_factor) * 1000


@pytest.mark.parametrize("dice_roll", fans_dice_rolls)
def test_fame(dice_roll):
    game = get_game_fans()
    proc = Fans(game)
    proc.start()
    # Home
    D6.fix(dice_roll[0])
    D6.fix(dice_roll[0])
    # Away
    D6.fix(dice_roll[1])
    D6.fix(dice_roll[1])
    proc.step(None)
    ff_home = game.state.home_team.fan_factor
    ff_away = game.state.away_team.fan_factor
    assert game.has_report_of_type(OutcomeType.FAME)
    home_fans = dice_roll[0] * 2 + ff_home
    away_fans = dice_roll[1] * 2 + ff_away
    for report in game.state.reports:
        if report.outcome_type == OutcomeType.FAME:
            if report.team == game.state.home_team:
                if home_fans == away_fans:
                    assert report.n == 0
                    assert report.team.state.fame == 0
                elif home_fans >= away_fans*2:
                    assert report.n == 2
                    assert report.team.state.fame == 2
                elif home_fans > away_fans:
                    assert report.n == 1
                    assert report.team.state.fame == 1
                elif home_fans < away_fans:
                    assert report.n == 0
                    assert report.team.state.fame == 0
            elif report.team == game.state.away_team:
                if away_fans == home_fans:
                    assert report.n == 0
                    assert report.team.state.fame == 0
                elif away_fans >= home_fans * 2:
                    assert report.n == 2
                    assert report.team.state.fame == 2
                elif away_fans > home_fans:
                    assert report.n == 1
                    assert report.team.state.fame == 1
                elif away_fans < home_fans:
                    assert report.n == 0
                    assert report.team.state.fame == 0
