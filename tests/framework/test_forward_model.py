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
    config = load_config("ff-11")
    config.fast_mode = False
    ruleset = load_rule_set(config.ruleset)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)
    away_agent = make_bot("random")
    home_agent = make_bot("random")
    game = Game(1, home, away, home_agent, away_agent, config)

    game.init()
    game.enable_forward_model()

    for i in range(0, 100, 20):
        for _ in range(i):
            game.step()
        game_copy = deepcopy(game)

        assert game.state is not game_copy.state


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

    game = get_game()
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
        assert_game_states(game, game_unchanged, equal=True)
    except AssertionError as e:
        set_trace()
        raise e


def test_flat_MCTS():
    class MCTSNode:
        def __init__(self, action):
            self.action = action
            self.evaluations = []

        def visits(self):
            return len(self.evaluations)

        def visit(self, score):
            self.evaluations.append(score)

        def score(self):
            return np.average(self.evaluations)

    class FlatMCTSBot(ffai.Agent):

        def __init__(self, name, seed=None):
            super().__init__(name)
            self.my_team = None
            self.rnd = np.random.RandomState(seed)

        def new_game(self, game, team):
            self.my_team = team

        def act(self, game):

            nodes = []
            root = game.get_forward_model_current_step()

            for action_choice in game.get_available_actions():
                if action_choice.action_type == ffai.ActionType.PLACE_PLAYER:
                    continue
                for player in action_choice.players:
                    nodes.append(MCTSNode(Action(action_choice.action_type, player=player)))
                for position in action_choice.positions:
                    nodes.append(MCTSNode(Action(action_choice.action_type, position=position)))
                if len(action_choice.players) == len(action_choice.positions) == 0:
                    nodes.append(MCTSNode(Action(action_choice.action_type)))

            best_node = None
            for node in nodes:
                game.step(node.action)
                score = self._evaluate(game)
                node.visit(score)
                if best_node is None or node.score() > best_node.score():
                    best_node = node

                game.revert_state(root)

            return best_node.action

        def _evaluate(self, game):
            return 1

        def end_game(self, game):
            pass

    # Register the bot to the framework
    ffai.register_bot('mcts_bot', FlatMCTSBot)

    # Load configurations, rules, arena and teams
    config = ffai.load_config("bot-bowl-ii")
    config.competition_mode = False
    ruleset = ffai.load_rule_set(config.ruleset)
    arena = ffai.load_arena(config.arena)
    home = ffai.load_team_by_filename("human", ruleset)
    away = ffai.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False
    config.fast_mode = True

    # Play 10 games
    for i in range(10):
        human_player = Agent("Gym Learner", human=True)
        bot = ffai.make_bot("mcts_bot")

        game = ffai.Game(i, home, away, human_player, human_player, config, arena=arena, ruleset=ruleset)
        game.init()
        game.enable_forward_model()

        assert game.config.fast_mode
        for _ in range(100):
            if game.state.game_over:
                break

            while len(game.get_available_actions()) == 0:
                game.step()

            action = bot.act(game)
            game.step(action)


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
