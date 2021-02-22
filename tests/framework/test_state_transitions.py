from tests.util import *
import pytest
from copy import deepcopy
from ffai.core.table import *
from ffai.core.util import *


def test_logged_list():
    log = Logger()
    log.enabled = True

    l = LoggedList(["yolo"])
    l.set_logger(log)

    l.append("1337")
    assert l.__repr__() == "['yolo', '1337']"

    log.step_backward(clear_log=False)
    assert l.__repr__() == "['yolo']"

    log.step_forward(to_step=0)
    assert l.__repr__() == "['yolo', '1337']"

    log.next_step()
    assert l.pop() == "1337"
    assert l.__repr__() == "['yolo']"

    log.step_backward(to_step=1, clear_log=False)
    assert l.__repr__() == "['yolo', '1337']"

    log.step_forward(to_step=1)
    assert l.__repr__() == "['yolo']"

    l.append("123")
    log.next_step()
    assert l.__repr__() == "['yolo', '123']"

    l[0] = "swag"
    assert l.__repr__() == "['swag', '123']"

    log.step_backward(to_step=2, clear_log=False)
    assert l.__repr__() == "['yolo', '123']"

    log.step_forward(to_step=2)
    assert l.__repr__() == "['swag', '123']"


def test_logged_state():
    class MyState(LoggedState):
        def __init__(self, data, log):
            super().__init__()
            self.set_logger(log)
            self.data = data

    log = Logger()
    log.enabled = True

    ms = MyState("immutable", log)
    ms.data = "new immutable"
    log.step_backward()
    assert ms.data == "immutable"

    ms = MyState(["mutable", "object"], log)
    exception_caught = False
    try:
        ms.data = []  # Should raise an exception
    except AttributeError:
        exception_caught = True
    assert exception_caught


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
    game.set_available_actions()

    # Prepare test
    init_state = deepcopy(game.state)
    game.state_log.enabled = True
    saved_step = game.state_log.current_step
    assert len(game.state.compare(init_state)) == 0

    # Do the things that will be reverted
    game.state.weather = WeatherType.SWELTERING_HEAT
    player.state.spp_earned = 2
    game.step(Action(ActionType.START_MOVE, player=player))
    game.step(Action(ActionType.MOVE, position=Square(3, 3)))
    game.step(Action(ActionType.END_PLAYER_TURN))
    # game.step(Action(ActionType.END_TURN))  #  Not working yet.

    # Make sure the differences are found
    errors = game.state.compare(init_state)
    assert len(errors) > 0
    # print("\n\nThese were the differences:")
    # for e in errors:
    #    print(e)

    # Revert and assert
    game.state_log.step_backward(to_step=saved_step)
    assert player.state.spp_earned == 0
    assert game.state.weather != WeatherType.SWELTERING_HEAT

    errors = game.state.compare(init_state)
    if len(errors) > 0:
        print("\n\nThese differences were not reverted:")
        for error in errors:
            print(error)
        set_trace()
        assert len(errors) == 0



    # assert init_state.to_json() == game.state.to_json()

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
