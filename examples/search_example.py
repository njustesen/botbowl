import ffai
from ffai.core import Action, Agent
import numpy as np
from copy import deepcopy
import random
import time


class Node:
    def __init__(self, action):
        self.action = action
        self.evaluations = []

    def visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        return np.average(self.evaluations)


class SearchBot(ffai.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):

        print(f"Turn {self.my_team.state.turn}")

        nodes = []
        game_copy = deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True

        for action_choice in game_copy.get_available_actions():
            if action_choice.action_type == ffai.ActionType.PLACE_PLAYER:
                continue
            for player in action_choice.players:
                nodes.append(Node(Action(action_choice.action_type, player=player)))
            for position in action_choice.positions:
                nodes.append(Node(Action(action_choice.action_type, position=position)))
            if len(action_choice.players) == len(action_choice.positions) == 0:
                nodes.append(Node(Action(action_choice.action_type)))

        root = game_copy.get_forward_model_current_step()
        print("ROOT STACK: ", len(game_copy.state.stack.items))

        best_node = None
        print(f"Evaluating {len(nodes)} nodes")
        t = time.time()
        for node in nodes:
            game_copy.step(node.action)
            while not game.state.game_over and len(game.state.available_actions) == 0:
                game_copy.step()
            score = self._evaluate(game)
            node.visit(score)
            print(f"{node.action.action_type}: {node.score()}")
            if best_node is None or node.score() > best_node.score():
                best_node = node

            game_copy.revert_state(root)

        print(f"{best_node.action.action_type} selected in {time.time() - t} seconds")

        return best_node.action

    def _evaluate(self, game):
        return random.random()

    def end_game(self, game):
        pass


# Register the bot to the framework
ffai.register_bot('search-bot', SearchBot)

# Load configurations, rules, arena and teams
config = ffai.load_config("bot-bowl-ii")
ruleset = ffai.load_rule_set(config.ruleset)
arena = ffai.load_arena(config.arena)
home = ffai.load_team_by_filename("human", ruleset)
away = ffai.load_team_by_filename("human", ruleset)
config.competition_mode = False
config.debug_mode = False
config.fast_mode = True
config.pathfinding_enabled = False

# Play a game
bot_a = ffai.make_bot("search-bot")
bot_b = ffai.make_bot("search-bot")
game = ffai.Game(1, home, away, bot_a, bot_b, config, arena=arena, ruleset=ruleset)
print("Starting game")
game.init()
print("Game is over")
