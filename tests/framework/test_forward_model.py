from tests.util import *
import pytest
from copy import deepcopy
from ffai.core.table import *
from ffai.core.util import *
from ffai.core.forward_model import *

from ffai.ai.registry import make_bot


def assert_equal_game_states(g1, g2):
    errors = g1.state.compare(g2.state)
    if len(errors) > 0:
        print("\n\nThese differences were not reverted:")
        for error in errors:
            print(error)
        assert len(errors) == 0


def test_random_games():
    steps = 1000

    config = load_config("ff-11")
    config.fast_mode = False
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    away_agent = make_bot("random")
    home_agent = make_bot("random")
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()

    to_step = game.get_forward_model_current_step()

    game.enable_forward_model()
    game_unchanged = deepcopy(game)


    i = 0
    while not game.state.game_over and i < steps:
        game.step()
        i += 1

    game.revert_state(to_step)

    try:
        assert_equal_game_states(game, game_unchanged)
    except AssertionError as e:
        set_trace()
        raise e


def test_logged_list():
    log = Trajectory()
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

    class Cant_log_this:
        pass

    log = Trajectory()
    log.enabled = True

    ms = MyState("immutable", log)
    ms.data = "new immutable"
    log.step_backward()
    assert ms.data == "immutable"

    ms = MyState(["mutable", "object"], log)

    exception_caught = False
    try:
        ms.data = Cant_log_this()
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
    game.enable_forward_model()
    init_state = deepcopy(game.state)
    saved_step = game.get_forward_model_current_step()
    assert len(game.state.compare(init_state)) == 0

    # Do the things that will be reverted
    game.state.weather = WeatherType.SWELTERING_HEAT
    player.state.spp_earned = 2
    game.step(Action(ActionType.START_MOVE, player=player))
    game.step(Action(ActionType.MOVE, position=Square(3, 3)))
    game.step(Action(ActionType.END_PLAYER_TURN))
    game.step(Action(ActionType.END_TURN))
    game.step(Action(ActionType.END_TURN))
    game.step(Action(ActionType.START_PASS, player=player))
    game.step(Action(ActionType.PASS, position=Square(5, 5)))

    # Make sure the differences are found
    errors = game.state.compare(init_state)
    assert len(errors) > 0

    # Revert and assert
    game.revert_state(to_step=saved_step)
    errors = game.state.compare(init_state)
    if len(errors) > 0:
        print("\n\nThese differences were not reverted:")
        for error in errors:
            print(error)
        assert len(errors) == 0
