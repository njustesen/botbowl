import botbowl
from botbowl.core import Action, Agent
import numpy as np
from copy import deepcopy
import random
import time


class Node:
    def __init__(self, action=None, parent=None):
        self.parent = parent
        self.children = []
        self.action = action
        self.evaluations = []

    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        return np.average(self.evaluations)


class SearchBot(botbowl.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        game_copy = deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True

        root_step = game_copy.get_step()
        root_node = Node()
        for action_choice in game_copy.get_available_actions():
            if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
                continue
            for player in action_choice.players:
                root_node.children.append(Node(Action(action_choice.action_type, player=player), parent=root_node))
            for position in action_choice.positions:
                root_node.children.append(Node(Action(action_choice.action_type, position=position), parent=root_node))
            if len(action_choice.players) == len(action_choice.positions) == 0:
                root_node.children.append(Node(Action(action_choice.action_type), parent=root_node))

        best_node = None
        print(f"Evaluating {len(root_node.children)} nodes")
        t = time.time()
        for node in root_node.children:
            game_copy.step(node.action)
            while not game.state.game_over and len(game.state.available_actions) == 0:
                game_copy.step()
            score = self._evaluate(game)
            node.visit(score)
            print(f"{node.action.action_type}: {node.score()}")
            if best_node is None or node.score() > best_node.score():
                best_node = node

            game_copy.revert(root_step)

        print(f"{best_node.action.action_type} selected in {time.time() - t} seconds")

        return best_node.action

    def _evaluate(self, game):
        return random.random()

    def end_game(self, game):
        pass


# Register the bot to the framework
botbowl.register_bot('search-bot', SearchBot)

# Load configurations, rules, arena and teams
config = botbowl.load_config("bot-bowl-iii")
ruleset = botbowl.load_rule_set(config.ruleset)
arena = botbowl.load_arena(config.arena)
home = botbowl.load_team_by_filename("human", ruleset)
away = botbowl.load_team_by_filename("human", ruleset)
config.competition_mode = False
config.debug_mode = False
config.fast_mode = True
config.pathfinding_enabled = False

# Play a game
bot_a = botbowl.make_bot("search-bot")
bot_b = botbowl.make_bot("search-bot")
game = botbowl.Game(1, home, away, bot_a, bot_b, config, arena=arena, ruleset=ruleset)
print("Starting game")
game.init()
print("Game is over")
