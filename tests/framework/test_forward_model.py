from itertools import cycle

from tests.util import *
import pytest
from copy import deepcopy
from botbowl.core.forward_model import ReversibleSet
from botbowl.ai.registry import make_bot


def test_revert_and_forward():
    game = get_game(fast_mode=True, human_agents=True)

    for actions in range(1, 10):
        game_before_actions = deepcopy(game)
        assert_game_states(game, game_before_actions, equal=True)
        prev_step = game.get_step()

        for _ in range(actions):
            game.step(get_random_action(game))
            if game.state.game_over:
                return

        assert_game_states(game, game_before_actions, equal=False)

        game_after_actions = deepcopy(game)
        undone_steps = game.revert(prev_step)

        assert_game_states(game, game_before_actions, equal=True)

        game.forward(undone_steps)
        assert_game_states(game, game_after_actions, equal=True)

        game = game_after_actions


def test_logged_state():
    class MyState(Reversible):
        def __init__(self, data, log):
            super().__init__()
            self.set_trajectory(log)
            self.data = data

    class CantLogThis:
        pass

    trajectory = Trajectory()
    trajectory.enabled = True

    ms = MyState("OLD_STATE", trajectory)
    ms.data = "NEW STATE"

    steps = trajectory.revert(0)
    assert ms.data == "OLD_STATE"

    trajectory.step_forward(steps)
    assert ms.data == "NEW STATE"

    ms = MyState(["mutable", "object"], trajectory)

    with pytest.raises(AttributeError):
        ms.data = CantLogThis()


def test_logged_set():
    traj = Trajectory()
    traj.enabled = True
    logged_set = ReversibleSet(set())
    logged_set.set_trajectory(traj)

    logged_set.add(123)

    assert 123 in logged_set
    assert len(logged_set) == 1

    logged_set.clear()
    assert len(logged_set) == 0

    traj.revert(1)
    assert 123 in logged_set
    assert len(logged_set) == 1

    traj.revert(0)

    assert len(logged_set) == 0


def test_forward_model_revert_every_step():
    game = get_game(fast_mode=True, human_agents=True)
    game.config.pathfinding_enabled = False

    def avail_actions_str(game_):
        return "-".join([action_choice.action_type.name for action_choice in game_.state.available_actions])

    def stack_str(game_):
        return "-".join([type(proc).__name__ for proc in game_.state.stack.items])

    def position_str(game_):
        positions = [player.position for team in game_.state.teams for player in team.players] + \
                    [game_.get_ball_position()]
        return "-".join([f"{pos}" if pos is not None else "None" for pos in positions])

    cycled_counter = cycle(range(10))
    tmp_game = None

    while not game.state.game_over:
        tmp_step = game.get_step()
        tmp_available_actions_str = avail_actions_str(game)
        tmp_stack_str = stack_str(game)
        tmp_position_str = position_str(game)

        do_deepcopy = next(cycled_counter) == 0
        if do_deepcopy:
            tmp_game = deepcopy(game)

        game.step(get_random_action(game))
        game.revert(tmp_step)

        assert tmp_available_actions_str == avail_actions_str(game)
        assert tmp_stack_str == stack_str(game)
        assert tmp_position_str == position_str(game)
        if do_deepcopy:
            assert_game_states(game, tmp_game, equal=True)

        game.step(get_random_action(game))


def get_game(fast_mode=True, human_agents=True):
    config = load_config("bot-bowl-iii")
    config.fast_mode = fast_mode
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    away_agent = Agent("Human 1", human=True, agent_id=1) if human_agents else make_bot("random")
    home_agent = Agent("Human 2", human=True, agent_id=2) if human_agents else make_bot("random")
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()

    game.enable_forward_model()

    return game


def get_random_action(game):
    while True:
        action_choice = game.rnd.choice(game.state.available_actions)
        if action_choice.action_type != botbowl.ActionType.PLACE_PLAYER:
            break
    position = game.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
    player = game.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
    return botbowl.Action(action_choice.action_type, position=position, player=player)


def assert_game_states(g1, g2, equal):
    errors = g1.state.compare(g2.state)
    if len(errors) > 0 and equal:
        print("\n\nThese differences were not reverted:")
        for error in errors:
            print(error)
        assert len(errors) == 0
    elif len(errors) == 0 and not equal:
        raise AssertionError("Expected not equal game states")


def test_actor_correctly_reset():
    game = get_game()
    root = game.get_step()
    actor1 = game.actor

    while game.actor == actor1:
        game.step(get_random_action(game))

    game.revert(root)
    assert game.actor == actor1


def test_immutability():
    sq = Square(1, 2)
    sq2 = Square(3, 2)

    with pytest.raises(AttributeError):
        sq.x = 2
    assert sq.x == 1
