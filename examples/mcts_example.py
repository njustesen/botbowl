import botbowl
from botbowl.core import Action
import numpy as np
from copy import deepcopy
import time
from typing import List
from examples.hash_example import gamestate_hash


def ucb1(node, c=0.707):
    best_node = None
    best_score = None
    maximize = True
    if type(node) is ActionNode and node.opp():
        maximize = False
    for child in node.children:
        mean_score = child.score() if maximize else 1-child.score()
        ucb_score = mean_score + 2*c * np.sqrt((2 * np.log(node.num_visits())) / child.num_visits())
        if best_score is None or ucb_score > best_score:
            best_node = child
            best_score = ucb_score
    return best_node


def random_policy(game, team):
    while True:
        action_choice = np.random.choice(game.state.available_actions)
        if action_choice.action_type == botbowl.ActionType.END_SETUP and game.get_players_on_pitch(team) == 0:
            continue
        if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
            continue
        break
    position = np.random.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
    player = np.random.choice(action_choice.players) if len(action_choice.players) > 0 else None
    action = botbowl.Action(action_choice.action_type, position=position, player=player)
    return action


def most_visited(node):
    return max(node.children, key=lambda x: x.num_visits())


def simple_heuristic(game: botbowl.Game, agent:botbowl.Agent):
    own_team = game.get_agent_team(agent)
    opp_team = game.get_opp_team(own_team)
    own_score = own_team.state.score
    opp_score = opp_team.state.score
    own_kos = len(game.get_knocked_out(own_team))
    opp_kos = len(game.get_knocked_out(opp_team))
    own_cas = len(game.get_casualties(own_team))
    opp_cas = len(game.get_casualties(opp_team))
    own_stunned = len([p for p in game.get_players_on_pitch(own_team, up=False) if p.state.stunned])
    opp_stunned = len([p for p in game.get_players_on_pitch(opp_team, up=False) if p.state.stunned])
    own_down = len([p for p in game.get_players_on_pitch(own_team, up=False) if not p.state.stunned])
    opp_down = len([p for p in game.get_players_on_pitch(opp_team, up=False) if not p.state.stunned])
    own_ejected = len(game.get_dungeon(own_team))
    opp_ejected = len(game.get_dungeon(opp_team))
    own_has_ball = False
    opp_has_ball = False
    ball_carrier = game.get_ball_carrier()
    if ball_carrier is not None:
        own_has_ball = 1 if ball_carrier.team == own_team else 0
        opp_has_ball = 1 if ball_carrier.team == opp_team else 0
    own = own_score/10 + own_has_ball/20 - (own_cas + own_ejected)/30 - own_kos/50 - own_stunned/100 - own_down/200
    opp = opp_score/10 + opp_has_ball/20 - (opp_cas + opp_ejected)/30 - opp_kos/50 - opp_stunned/100 - opp_down/200
    if game.state.game_over:
        if game.get_winner() == agent:
            return 1
        elif game.get_winner() is None:
            return 0.5
        else:
            return -1
    return 0.5 + own - opp


class Node:

    def __init__(self):
        self.evaluations = []

    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        return np.average(self.evaluations)

    def print(self):
        raise NotImplementedError

    def opp(self):
        raise NotImplementedError


class ActionNode(Node):

    def __init__(self, game, hash_key, opp=False):
        super().__init__()
        self.hash_key = hash_key
        self.available_actions = self._extract_actions(game)
        self.children: List[ChangeNode] = []
        self._opp = opp
        self.terminal = game.state.game_over

    def opp(self):
        return self._opp

    def _extract_actions(self, game):
        actions = []
        for action_choice in game.get_available_actions():
            if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
                continue
            if action_choice.action_type == botbowl.ActionType.END_SETUP:
                continue
            if len(action_choice.players) > 0:
                for player in action_choice.players:
                    actions.append(Action(action_choice.action_type, position=None, player=player))
            elif len(action_choice.positions) > 0:
                for position in action_choice.positions:
                    actions.append(Action(action_choice.action_type, position=position))
            else:
                actions.append(Action(action_choice.action_type))
        return actions

    def is_fully_expanded(self):
        return len(self.children) == len(self.available_actions)

    def print(self, tabs=0):
        t = ''.join(['\t' for _ in range(tabs)])
        min_max = 'min' if self.opp() else 'max'
        if len(self.children) == 0:
            print(f"{t}<ActionNode p='{min_max}' visits={self.num_visits()} score={self.score()} actions={len(self.available_actions)}/>")
        else:
            print(f"{t}<ActionNode p='{min_max}' visits={self.num_visits()} score={self.score()} actions={len(self.available_actions)}>")
            for child in self.children:
                child.print(tabs+1)
        if len(self.children) > 0:
            print(f'{t}</ActionNode>')


class ChangeNode(Node):

    def __init__(self, parent, action):
        super().__init__()
        self.parent = parent
        self.action = action
        self.outcomes = {}  # hash: Node
        self.evaluations = []
        self.terminal = False

    def print(self, tabs=0):
        t = ''.join(['\t' for _ in range(tabs)])
        min_max = 'min' if self.parent.opp() else 'max'
        if len(self.outcomes) == 0:
            print(f"{t}<ChanceNode p='{min_max}' visits={self.num_visits()} score={self.score()} action='{self.action.to_json()}'/>")
        else:
            print(f"{t}<ChanceNode p='{min_max}' visits={self.num_visits()} score={self.score()} action='{self.action.to_json()}'>")
            for _, child in self.outcomes.items():
                child.print(tabs+1)
        if len(self.outcomes) > 0:
            print(f"{t}</ChanceNode>")

    def opp(self):
        return self.parent.opp()


class MCTS:

    def __init__(self, game, agent, tree_policy, action_policy, heuristic):
        self.game = game
        self.agent = agent
        self.tree_policy = tree_policy
        self.action_policy = action_policy
        self.heuristic = heuristic

    def run(self, seconds):
        t = time.time()
        hash_key = gamestate_hash(self.game)
        root = ActionNode(self.game, hash_key)
        root_score = self.heuristic(self.game, self.agent)
        step = self.game.get_step()
        while time.time() < t + seconds:
            tree_trajectory = self._select_and_expand(root)
            self._rollout()
            score = self.heuristic(self.game, self.agent)
            self._backpropagate(tree_trajectory, score)
            self.game.revert(step)
            # root.print()
        return root

    def _backpropagate(self, trajectory, score):
        for node in reversed(trajectory):
            node.visit(score)

    def _rollout(self):
        turns = self.game.state.home_team.state.turn + self.game.state.away_team.state.turn
        while not self.game.state.game_over and turns == self.game.state.home_team.state.turn + self.game.state.away_team.state.turn:
            action = self.action_policy(self.game, self.game.get_agent_team(self.agent))
            self.game.step(action)
            if "SETUP_" in action.action_type.name:
                self.game.step(Action(botbowl.ActionType.END_SETUP))

    def _select_and_expand(self, root):
        node = root
        trajectory = [root]
        while node.is_fully_expanded():
            if node.terminal:
                return trajectory
            best_child = self.tree_policy(node)
            self.game.step(best_child.action)
            if "SETUP_" in best_child.action.action_type.name:
                self.game.step(Action(botbowl.ActionType.END_SETUP))
            trajectory.append(best_child)
            hash_key = gamestate_hash(self.game)
            if hash_key not in best_child.outcomes:
                node = ActionNode(self.game, hash_key, self.game.actor == self.agent)
                best_child.outcomes[hash_key] = node
                trajectory.append(node)
                return trajectory
            else:
                node = best_child.outcomes[hash_key]
            trajectory.append(node)
        new_chance_node, new_node = self._expand(node)
        trajectory.append(new_chance_node)
        trajectory.append(new_node)
        return trajectory

    def _expand(self, node: ActionNode):
        next_action_idx = len(node.children)
        action = node.available_actions[next_action_idx]
        chance_node = ChangeNode(node, action)
        node.children.append(chance_node)
        self.game.step(action)
        if "SETUP_" in action.action_type.name:
            self.game.step(Action(botbowl.ActionType.END_SETUP))
        hash_key = gamestate_hash(self.game)
        node = ActionNode(self.game, hash_key, self.game.actor == self.agent)
        chance_node.outcomes[node.hash_key] = node
        return chance_node, node


class MCTSBot(botbowl.Agent):

    def __init__(self,
                 name,
                 tree_policy=ucb1,
                 action_policy=random_policy,
                 final_policy=most_visited,
                 heuristic=simple_heuristic,
                 seconds=5,
                 seed=None):
        super().__init__(name)
        self.my_team = None
        self.rng = np.random.RandomState(seed)
        self.tree_policy = tree_policy
        self.action_policy = action_policy
        self.final_policy = final_policy
        self.heuristic = heuristic
        self.seconds = seconds
        self.next_action = None

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
        game_copy = deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True
        mcts = MCTS(game_copy,
                    self,
                    tree_policy=self.tree_policy,
                    action_policy=self.action_policy,
                    heuristic=self.heuristic)
        root = mcts.run(self.seconds)
        # root.print()
        best_node = self.final_policy(root)
        # print(f"Found action {best_node.action.action_type} with {root.num_visits()} rollouts.")
        action = best_node.action
        if "SETUP_" in action.action_type.name:
            self.next_action = Action(botbowl.ActionType.END_SETUP)
        else:
            self.next_action = None
        return action

    def end_game(self, game):
        pass


# Register the bot to the framework
botbowl.register_bot('mcts', MCTSBot)

for i in [1, 3, 5, 7, 11]:
    print(f"Testing env {i}")
    # Load configurations, rules, arena and teams
    config = botbowl.load_config(f"gym-{i}")
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    for _ in range(10):
        home = botbowl.load_team_by_filename("human", ruleset)
        away = botbowl.load_team_by_filename("human", ruleset)
        config.competition_mode = False
        config.debug_mode = False
        config.fast_mode = True
        config.pathfinding_enabled = True

        # Play a game
        bot_a = botbowl.make_bot("mcts")
        bot_b = botbowl.make_bot("random")
        game = botbowl.Game(1, home, away, bot_a, bot_b, config, arena=arena, ruleset=ruleset)
        print("Starting game")
        game.init()
        print(f"{game.home_agent.name} score: {game.state.home_team.state.score}")
        print(f"{game.away_agent.name} score: {game.state.away_team.state.score}")
