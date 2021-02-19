from tests.util import *
import pytest
from copy import deepcopy
from ffai.core.table import *
from ffai.core.util import compare_json

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
    game.state_log.enabled = True
    saved_step = game.state_log.current_step
    assert init_state.to_json() == game.state.to_json()

    # Do the thing that will be reverted
    game.state.weather = WeatherType.SWELTERING_HEAT
    player.state.stunned = True

    # Revert and assert
    assert not init_state.to_json() == game.state.to_json()
    game.state_log.step_backward(to_step=saved_step)

    assert not player.state.stunned
    assert game.state.weather != WeatherType.SWELTERING_HEAT



    for s in compare_json(init_state.to_json(), game.state.to_json()):
        print(s)

    assert init_state.to_json() == game.state.to_json()

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