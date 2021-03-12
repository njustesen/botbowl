from tests.util import *
import pytest
from copy import deepcopy
from ffai.core.table import *
from ffai.core.util import *
from ffai.core.forward_model import *

from ffai.ai.registry import make_bot


def get_game(fast_mode=False):
    config = load_config("ff-11")
    config.fast_mode = fast_mode
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    away_agent = make_bot("random")
    home_agent = make_bot("random")
    game = Game(1, home, away, home_agent, away_agent, config)
    return game


def test_game_deep_copy():
    config = load_config("ff-3")
    config.fast_mode = True
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    agent = make_bot("random")
    human_agent = Agent("Gym Learner", human=True)
    game = Game(1, home, away, human_agent, human_agent, config)

    game.enable_forward_model()
    game.init()

    actions = 0

    while not game.state.game_over and actions < 20:
        actions += 1
        game_copy = deepcopy(game)
        assert_game_states(game, game_copy, equal=True)
        game.step(agent.act(game))

    assert actions == 20


def test_revert_multiple_times():
    game = get_game()
    game.init()

    game.enable_forward_model()
    step = game.get_forward_model_current_step()
    init_state = deepcopy(game)
    assert_game_states(game, init_state, equal=True)

    for i in range(1, 10):
        for _ in range(i):
            game.step()

        assert_game_states(game, init_state, equal=False)

        game.revert_state(to_step=step)
        assert_game_states(game, init_state, equal=True)


def assert_game_states(g1, g2, equal):
    errors = g1.state.compare(g2.state)
    if len(errors) > 0 and equal:
        print("\n\nThese differences were not reverted:")
        for error in errors:
            print(error)
        assert len(errors) == 0
    elif len(errors) == 0 and not equal:
        raise AssertionError("Expected not equal game states")


def test_random_games():
    steps = 1000

    for _ in range(4):
        game = get_game()
        game.init()

        to_step = game.get_forward_model_current_step()

        game.enable_forward_model()
        game_unchanged = deepcopy(game)

        i = 0
        while not game.state.game_over and i < steps:
            game.step()
            i += 1

        try:
            game.revert_state(to_step)
        except KeyError as error:
            set_trace()
            print("Got key error, investigate!")

        try:
            assert_game_states(game, game_unchanged, equal=True)
        except AssertionError as e:
            set_trace()
            raise e


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


def test_logged_set():
    traj = Trajectory()
    traj.enabled = True
    logged_set = LoggedSet(set())
    logged_set.set_logger(traj)

    logged_set.add(123)

    assert 123 in logged_set
    assert len(logged_set) == 1

    traj.next_step()
    logged_set.clear()
    assert len(logged_set) == 0

    traj.step_backward(1)
    assert 123 in logged_set
    assert len(logged_set) == 1

    traj.step_backward(0)

    assert len(logged_set) == 0
