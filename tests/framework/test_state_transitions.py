from tests.util import *
import pytest
from copy import deepcopy
from ffai.core.table import *

def test_game_state_revert():
    # Init
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()

    # Setup
    player = team.players[0]
    game.put(player, Square(2, 2))
    game.get_ball().move_to(player.position)
    game.get_ball().is_carried = True

    # Prepare test
    init_state = deepcopy(game.state)
    game.logging_enabled = True
    assert init_state == game.state

    # Do the thing that will be reverted
    game.state.weather = WeatherType.SWELTERING_HEAT
    player.state.stunned = True

    # Revert and assert
    assert not init_state == game.state
    game.revert_state()
    assert init_state == game.state

    # catcher = team.players[1]
    # catcher_position = Square(passer.position.x + 12, passer.position.y + 0)
    # game.put(catcher, catcher_position)
    #
    # game.set_available_actions()
    # game.state.reports.clear()
    #
    # D6.fix(2)  # Fumble pass
    #
    # game.step(Action(ActionType.START_PASS, player=passer))
    # game.step(Action(ActionType.PASS, position=catcher_position))
    #
    # assert game.has_report_of_type(OutcomeType.SKILL_USED)
    # assert not game.has_report_of_type(OutcomeType.FUMBLE)
    # assert game.get_ball_carrier() == passer