"""
==========================
Author: Niels Justesen
Year: 2020
==========================
This module contains pathfinding functionalities for FFAI.
"""

from typing import Optional, List
from ffai.core.model import Player, Square
from ffai.core.table import Skill, WeatherType, Tile
from ffai.core.game import Game
import time
import copy
from functools import lru_cache
import numpy as np


class Path:

    def __init__(self, steps: List['Square'], prob: float):
        self.steps = steps
        self.prob = prob
        self.dodge_used_prob: float = 0
        self.sure_feet_used_prob: float = 0
        self.rr_used_prob: float = 0

    def __len__(self) -> int:
        return len(self.steps)

    def get_last_step(self) -> 'Square':
        return self.steps[-1]

    def is_empty(self) -> bool:
        return len(self) == 0


class Node:

    def __init__(self, position, parent=None, moves=0):
        self.parent: Optional[Node] = parent
        self.position = position
        self.costs = [0, 0, 0, 0, 0]
        if parent is not None:
            self.moves = parent.moves + moves
            self.prob = parent.prob
            self.dodge_used_prob = self.parent.dodge_used_prob
            self.sure_feet_used_prob = self.parent.sure_feet_used_prob
            self.rr_used_prob = self.parent.rr_used_prob
        else:
            self.moves: int = 0
            self.prob: float = 1  # Prob of success
            self.dodge_used_prob: float = 0
            self.sure_feet_used_prob: float = 0
            self.rr_used_prob: float = 0
        self.update_costs()

    def update_costs(self):
        self.costs[0] = 1-self.prob
        self.costs[1] = self.moves
        self.costs[2] = self.dodge_used_prob
        self.costs[3] = self.sure_feet_used_prob
        self.costs[4] = self.rr_used_prob

    def add_moves(self, moves):
        self.moves += moves
        self.costs[1] = self.moves

    def add_dodge_prob(self, p:float, dodge_skill=False, rr=False):
        can_use_dodge_p = 0 if not dodge_skill else 1 - self.dodge_used_prob
        assert can_use_dodge_p <= 1
        dodge_used_now_p = (1-p) * can_use_dodge_p
        assert dodge_used_now_p <= 1
        can_use_rr_p = 0 if not rr else (1 - can_use_dodge_p) * (1 - self.rr_used_prob)
        assert can_use_rr_p <= 1
        rr_used_now_p = (1-p) * can_use_rr_p
        assert rr_used_now_p < 1
        success_first = p
        success_skill = dodge_used_now_p * p
        success_reroll = rr_used_now_p * p
        success = success_first + success_skill + success_reroll
        if success >= 1:
            raise Exception(f"{success}: {success_first} + {success_skill} + {success_reroll}")
        self.prob *= success
        self.dodge_used_prob += success_skill
        self.rr_used_prob += success_reroll
        self.update_costs()

    def add_gfi_prob(self, p:float, sure_feet_skill=False, rr=False):
        can_use_sure_feet_p = 0 if not sure_feet_skill else 1 - self.sure_feet_used_prob
        assert can_use_sure_feet_p <= 1
        sure_feet_used_now_p = (1 - p) * can_use_sure_feet_p
        assert sure_feet_used_now_p <= 1
        can_use_rr_p = 0 if not rr else (1 - can_use_sure_feet_p) * (1 - self.rr_used_prob)
        assert can_use_rr_p <= 1
        rr_used_now_p = (1 - p) * can_use_rr_p
        assert rr_used_now_p < 1
        success_first = p
        success_skill = sure_feet_used_now_p * p
        success_reroll = rr_used_now_p * p
        success = success_first + success_skill + success_reroll
        if success >= 1:
            raise Exception(f"{success}: {success_first} + {success_skill} + {success_reroll}")
        self.prob *= success
        self.sure_feet_used_prob += success_skill
        self.rr_used_prob += success_reroll
        self.update_costs()


class SortedList:

    def __init__(self, sort_lambda):
        self.list = []
        self.sort_lambda = sort_lambda

    def first(self):
        return self.list[0]

    def clear(self):
        self.list.clear()

    def append(self, o):
        self.list.append(o)
        self.list.sort(key=self.sort_lambda)

    def remove(self, o):
        self.list.remove(o)

    def __len__(self):
        return len(self.list)

    def contains(self, o):
        return o in self.list


class ParetoFrontier:

    def __init__(self):
        self.nodes = []

    def __pareto_dominant(self, a:Node, b:Node):
        a_dom = 0
        b_dom = 0
        for i in range(len(a.costs)):
            if a.costs[i] < b.costs[i]:
                a_dom += 1
            if a.costs[i] < b.costs[i]:
                b_dom += 1
        if a_dom > 0 and b_dom == 0:
            return a
        if b_dom > 0 and a_dom == 0:
            return b
        return None

    def add(self, node):
        new = []
        for contestant in self.nodes:
            if contestant.costs == node.costs:
                return  # We already have a node with same costs
            dominant = self.__pareto_dominant(node, contestant)
            if dominant is None:
                new.append(contestant)  # Keep contestant
            elif dominant == contestant:
                return  # Node is dominated so we don't need it
        self.nodes = new
        self.nodes.append(node)

    def get_best(self):
        best = None
        for node in self.nodes:
            if best is None or node.prob > best.prob or (node.prob == best.prob and node.moves < best.moves):
                best = node
        return best


def _alter_state(game, player, from_position, moves_used):
    orig_player, orig_ball = None, None
    if from_position is not None or moves_used is not None:
        orig_player = copy.deepcopy(player)
        orig_ball = copy.deepcopy(game.get_ball())
    # Move player if another starting position is used
    if from_position is not None:
        assert game.get_player_at(from_position) is None or game.get_player_at(from_position) == player
        game.move(player, from_position)
        if from_position == game.get_ball_position() and game.get_ball().on_ground:
            game.get_ball().carried = True
    if moves_used != None:
        assert moves_used >= 0
        player.state.moves = moves_used
        if moves_used > 0:
            player.state.up = True
    return orig_player, orig_ball


def _reset_state(game, player, orig_player, orig_ball):
    if orig_player is not None:
        game.move(player, orig_player.position)
        player.state = orig_player.state
    if orig_ball is not None:
        game.ball = orig_ball


def _collect_path(node):
    steps = []
    n = node
    while n is not None:
        steps.append(n.position)
        n = n.parent
    steps.reverse()
    steps = steps[1:]
    path = Path(steps, prob=node.prob)
    path.dodge_used_prob = node.dodge_used_prob
    path.sure_feet_used_prob = node.sure_feet_used_prob
    path.rr_used_prob = node.rr_used_prob
    return path


def find_path(game, player, position=None, target_x=None, target_player=None, allow_rr=False, blitz=False, max_moves=None, all=False) -> Optional[Path]:

    # Only one target type
    if all:
        assert position is None and target_player is None and target_x is None
    else:
        assert position is None or (target_x is None and target_player is None)
        assert target_x is None or (position is None and target_player is None)
        assert target_player is None or (position is None and target_x is None and target_player.position is not None)

    # Goal positions used if no position is given
    goals = []
    if position is not None:
        goals.append(position)

    # If we are already at the target
    if (player is not None and player.position == position) or \
            (target_x is not None and player.position.x == target_x) or \
            (target_player is not None and target_player.position.is_adjacent(player.position)):
        return Path([], 1.0)

    # If the destination is blocked, we can't get there
    if position is not None and game.get_player_at(position) is not None:
        return None

    # Blitzes
    pareto_blitzes = {}

    # Max search depth
    if max_moves is None:
        ma = player.get_ma()
        max_moves = player.num_moves_left()
    else:
        ma = max_moves

    # Nodes on the Pareto-front are maintained
    pareto_frontiers = {}

    # Keep reference to best path to goal so far
    best = None

    # Make open list sorted by probability
    openset = SortedList(lambda x: x.prob)

    # Make initial node
    init_node = Node(player.position)
    openset.append(init_node)
    pareto_frontiers[init_node.position] = ParetoFrontier()
    pareto_frontiers[init_node.position].add(init_node)

    # while we have unexpanded nodes
    while len(openset) > 0:

        # pull out the first node in our open list
        current = openset.first()
        openset.remove(current)

        # Check if it's still on the pareto frontier
        if not current in pareto_frontiers[current.position].nodes:
            continue

        # Stop if this path can't become better than the best
        if best is not None:
            if current.prob < best.prob:
                continue
            if position is not None:
                if current.prob == best.prob and current.moves + current.position.distance(position) + (1 if blitz else 0) > best.moves:
                    continue
            elif target_player is not None:
                if current.prob == best.prob and current.moves + current.position.distance(target_player.position) - 1 + (1 if blitz else 0) > best.moves:
                    continue
            elif target_x is not None:
                if current.prob == best.prob and current.moves + abs(current.position.x - target_x) > best.moves:
                    continue
            else:
                if current.prob == best.prob and current.moves + 1 > best.moves:
                    continue

        # Expand
        for neighbour in game.get_adjacent_squares(current.position, occupied=False):

            # If out of moves or out of reach stop here
            if position is not None and current.moves + current.position.distance(position) > max_moves:
                continue
            if target_player is not None and current.moves + current.position.distance(target_player.position) - 1 + (1 if blitz else 0) > max_moves:
                continue
            if target_x is not None and current.moves + abs(current.position.x - target_x) + (1 if blitz else 0) > max_moves:
                continue
            if all and blitz:
                adjacent_opponents = game.get_adjacent_players(neighbour, team=game.get_opp_team(player.team), down=False)
                if adjacent_opponents and current.moves >= max_moves - 1:
                    continue
                if not adjacent_opponents and current.moves >= max_moves - 2:
                    continue

            # Make expanded node
            node = Node(neighbour, parent=current, moves=1)
            dodge_p = game.get_dodge_prob_from(player, from_position=current.position, to_position=neighbour)
            can_use_dodge = player.has_skill(Skill.DODGE) and not game.get_adjacent_players(current.position, player.team, down=False, skill=Skill.TACKLE)
            can_use_rr = allow_rr and game.can_use_reroll(player.team)
            if node.moves > ma:
                node.add_gfi_prob(5/6, sure_feet_skill=player.has_skill(Skill.SURE_FEET), rr=can_use_rr)
            if dodge_p < 1.0:
                node.add_dodge_prob(dodge_p, dodge_skill=can_use_dodge, rr=can_use_rr)

            # If a potential blitz position, copy node and add to blitzes
            if all and blitz and game.get_adjacent_players(node.position, down=False, team=game.get_opp_team(player.team)):
                if blitz and node.moves < max_moves:
                    blitz_node = copy.copy(node)
                    blitz_node.add_moves(1)
                    if blitz_node.moves >= ma:
                        can_use_rr = allow_rr and game.can_use_reroll(player.team)
                        blitz_node.add_gfi_prob(5 / 6, sure_feet_skill=player.has_skill(Skill.SURE_FEET), rr=can_use_rr)
                    if blitz_node.position not in pareto_blitzes:
                        pareto_blitzes[blitz_node.position] = ParetoFrontier()
                    pareto_blitzes[blitz_node.position].add(blitz_node)

            # Check if goal was reached
            goal_reached = False
            if target_x is not None and node.position.x == target_x:
                goal_reached = True
            elif target_player is not None and node.position.distance(target_player.position) == 1:
                goal_reached = True
            elif node.position == position:
                goal_reached = True

            if goal_reached:
                # Extra move/GFI when blitzing
                if blitz and node.moves == max_moves:
                    continue  # No moves left to blitz
                if blitz and node.moves >= ma:
                    can_use_rr = allow_rr and game.can_use_reroll(player.team)
                    node.add_moves(1)
                    node.add_gfi_prob(5/6, sure_feet_skill=player.has_skill(Skill.SURE_FEET), rr=can_use_rr)
                # Check if path beats the best
                if best is None:
                    best = node
                elif node.prob > best.prob:
                    best = node
                elif node.prob == best.prob and node.moves < best.moves:
                    best = node
            else:

                # No moves left
                if current.moves == max_moves:
                    continue

                # Add to pareto frontier
                if neighbour not in pareto_frontiers:
                    pareto_frontiers[neighbour] = ParetoFrontier()
                pareto_frontiers[neighbour].add(node)

                # If it's on the pareto frontier
                if node in pareto_frontiers[neighbour].nodes:

                    # Add it to the open set
                    openset.append(node)

    # Search is over - backtrack for goals to find safest path
    if all:
        paths = []
        # Reset pareto frontiers and recreate from blitzes
        if blitz:
            pareto_frontiers = pareto_blitzes
        # Pareto nodes?
        for position, frontier in pareto_frontiers.items():
            best = frontier.get_best()
            if best.parent is None:
                continue
            path = _collect_path(best)
            paths.append(path)
        return paths

    if best is None:
        return None

    node = best
    path = _collect_path(node)
    return path


def get_safest_path(game, player, position, from_position=None, allow_team_reroll=False, num_moves_used=0, blitz=False):
    """
    :param game:
    :param player: the player to move
    :param position: the location to move to
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param allow_team_reroll: allow team rerolls to be used.
    :return a path containing the list of squares that forms the safest (and thereafter shortest) path for the given player to the
    given position and the probability of success.
    """
    if from_position is not None and num_moves_used != 0:
        orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)
    path = find_path(game, player, position, allow_rr=allow_team_reroll, blitz=blitz)
    if from_position is not None and num_moves_used != 0:
        _reset_state(game, player, orig_player, orig_ball)

    return path


def get_safest_path_to_endzone(game, player, from_position=None, allow_team_reroll=False, num_moves_used=None):
    """
    :param game:
    :param player:
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param allow_team_reroll: allow team rerolls to be used.´
    :return: a path containing the list of squares that forms the safest (and thereafter shortest) path for the given player to
    a position in the opponent endzone.
    """
    if from_position is not None and num_moves_used != 0:
        orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)
    x = game.get_opp_endzone_x(player.team)
    path = find_path(game, player, target_x=x, allow_rr=allow_team_reroll, blitz=False)
    if from_position is not None and num_moves_used != 0:
        _reset_state(game, player, orig_player, orig_ball)
    return path


def get_safest_path_to_player(game, player, target_player, from_position=None, allow_team_reroll=False, num_moves_used=None, blitz=False):
    """
    :param game:
    :param player: the player to move
    :param target_player: the player to move adjacent to
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param allow_team_reroll: allow team rerolls to be used.
    :param blitz: whether it is a blitz move.
    :return a path containing the list of squares that forms the safest (and thereafter shortest) path for the given player to
    a position that is adjacent to the other player and the probability of success.
    """
    if from_position is not None and num_moves_used != 0:
        orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)
    path = find_path(game, player, allow_rr=allow_team_reroll, target_player=target_player, blitz=blitz)
    if from_position is not None and num_moves_used != 0:
        _reset_state(game, player, orig_player, orig_ball)

    return path


def get_all_paths(game, player, from_position=None, allow_team_reroll=False, num_moves_used=None, blitz=False):
    """
    :param game:
    :param player: the player to move
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param allow_team_reroll: allow team rerolls to be used.
    :param blitz: only finds blitz moves if True.
    :return a path containing the list of squares that forms the safest (and thereafter shortest) path for the given player to
    a position that is adjacent to the other player and the probability of success.
    """
    if from_position is not None and num_moves_used != 0:
        orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)
    path = find_path(game, player, allow_rr=allow_team_reroll, blitz=blitz, all=True)
    if from_position is not None and num_moves_used != 0:
        _reset_state(game, player, orig_player, orig_ball)

    return path
