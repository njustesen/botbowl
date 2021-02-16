"""
==========================
Author: Niels Justesen
Year: 2020
==========================
This module contains pathfinding functionalities for FFAI.
"""

from typing import Optional, List
from ffai.core.table import Rules
from ffai.core.model import Player, Square, D6
from ffai.core.table import Skill, WeatherType, Tile
import copy
import numpy as np
from queue import PriorityQueue
from collections import namedtuple


class Path:

    def __init__(self, steps: List['Square'], prob: float, rolls: Optional[List[float]], block_dice=None, is_foul=False, is_handoff=False):
        self.steps = steps
        self.prob = prob
        self.dodge_used_prob: float = 0
        self.sure_feet_used_prob: float = 0
        self.rr_used_prob: float = 0
        self.rolls = rolls
        self.block_dice = block_dice
        self.is_foul = is_foul
        self.is_handoff = is_handoff

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
        self.costs = [0, 0]
        #self.costs = [0, 0, 0, 0, 0]  # Not sure if we need these additional info
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
        self.costs[0] = round(1-self.prob, 2)
        self.costs[1] = self.moves
        #self.costs[2] = self.dodge_used_prob  # Not sure if we need these additional info
        #self.costs[3] = self.sure_feet_used_prob  # Not sure if we need these additional info
        #self.costs[4] = self.rr_used_prob  # Not sure if we need these additional info

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
        self.costs[0] = round(1-self.prob, 2)

    def p(self):
        return 1-self.costs[0]

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
        self.costs[0] = round(1-self.prob, 2)


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
            if best is None or node.p() > best.p() or (node.p() == best.p() and node.moves < best.moves):
                best = node
        return best


class Pathfinder:

    def __init__(self, game, player, position=None, target_x=None, target_player=None, allow_rr=False, blitz=False, max_moves=None, all=False):
        self.game = game
        self.player = player
        self.position = position
        self.target_x = target_x
        self.target_player = target_player
        self.allow_rr = allow_rr
        self.blitz = blitz
        self.all = all
        self.pareto_blitzes = {}
        self.pareto_frontiers = {}
        self.best = None
        self.openset = SortedList(lambda x: x.p())
        self.max_moves = max_moves
        # Max search depth
        if self.max_moves is None:
            self.ma = self.player.get_ma()
            self.max_moves = self.player.num_moves_left()
        else:
            self.ma = self.max_moves
        # Goal positions used if no position is given
        self.goals = []
        if self.position is not None:
            self.goals.append(self.position)
        # Only one target type
        if self.all:
            assert self.position is None and self.target_player is None and self.target_x is None
        else:
            assert self.position is None or (self.target_x is None and self.target_player is None)
            assert self.target_x is None or (self.position is None and self.target_player is None)
            assert self.target_player is None or (
                        self.position is None and self.target_x is None and self.target_player.position is not None)

    def _collect_path(self, node):
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

    def _can_beat_best(self, node):
        if self.best is not None:
            if node.prob < self.best.prob:
                return False
            if self.position is not None:
                if node.prob == self.best.prob and node.moves + node.position.distance(self.position) + (
                1 if self.blitz else 0) > self.best.moves:
                    return False
            elif self.target_player is not None:
                if node.prob == self.best.prob and node.moves + node.position.distance(
                        self.target_player.position) - 1 + (1 if self.blitz else 0) > self.best.moves:
                    return False
            elif self.target_x is not None:
                if node.prob == self.best.prob and node.moves + abs(
                        node.position.x - self.target_x) > self.best.moves:
                    return False
            else:
                if node.prob == self.best.prob and node.moves + 1 > self.best.moves:
                    return False
        return True

    def _target_out_of_reach(self, current, position):
        # If out of moves or out of reach stop here
        if self.position is not None and current.moves + current.position.distance(self.position) > self.max_moves:
            return True
        if self.target_player is not None and current.moves + current.position.distance(
                self.target_player.position) - 1 + (1 if self.blitz else 0) > self.max_moves:
            return True
        if self.target_x is not None and current.moves + abs(current.position.x - self.target_x) + (1 if self.blitz else 0) > self.max_moves:
            return True
        if self.all and self.blitz:
            adjacent_opponents = self.game.get_adjacent_players(position,
                                                                team=self.game.get_opp_team(self.player.team),
                                                                down=False)
            if adjacent_opponents and current.moves >= self.max_moves - 1:
                return True
            if not adjacent_opponents and current.moves >= self.max_moves - 2:
                return True
        return False

    def _get_child(self, current, neighbour):
        node = Node(neighbour, parent=current, moves=1)
        dodge_p = self.game.get_dodge_prob_from(self.player, from_position=current.position, to_position=neighbour)
        can_use_dodge = self.player.has_skill(Skill.DODGE) and not self.game.get_adjacent_players(current.position,
                                                                                                  self.player.team,
                                                                                                  down=False,
                                                                                                  skill=Skill.TACKLE)
        can_use_rr = self.allow_rr and self.game.can_use_reroll(self.player.team)
        if node.moves > self.ma:
            node.add_gfi_prob(5 / 6, sure_feet_skill=self.player.has_skill(Skill.SURE_FEET), rr=can_use_rr)
        if dodge_p < 1.0:
            node.add_dodge_prob(dodge_p, dodge_skill=can_use_dodge, rr=can_use_rr)
        return node

    def _add_blitz(self, node):
        blitz_node = copy.copy(node)
        blitz_node.add_moves(1)
        if blitz_node.moves >= self.ma:
            can_use_rr = self.allow_rr and self.game.can_use_reroll(self.player.team)
            blitz_node.add_gfi_prob(5 / 6, sure_feet_skill=self.player.has_skill(Skill.SURE_FEET), rr=can_use_rr)
        if blitz_node.position not in self.pareto_blitzes:
            self.pareto_blitzes[blitz_node.position] = ParetoFrontier()
        self.pareto_blitzes[blitz_node.position].add(blitz_node)

    def _goal_reached(self, node):
        if self.target_x is not None and node.position.x == self.target_x:
            return True
        elif self.target_player is not None and node.position.distance(self.target_player.position) == 1:
            return True
        elif node.position == self.position:
            return True
        return False

    def _collect_paths(self) -> List[Path]:
        paths = []
        # Reset pareto frontiers and recreate from blitzes
        if self.blitz:
            self.pareto_frontiers = self.pareto_blitzes
        # Pareto nodes?
        for position, frontier in self.pareto_frontiers.items():
            best = frontier.get_best()
            if best.parent is None:
                continue
            path = self._collect_path(best)
            paths.append(path)
        return paths

    def get_path(self) -> Optional[Path]:
        paths = self.get_paths()
        if len(paths) > 0:
            return paths[0]
        else:
            return None

    def get_paths(self) -> List[Path]:

        # If we are already at the target
        if (self.player is not None and self.player.position == self.position) or \
                (self.target_x is not None and self.player.position.x == self.target_x) or \
                (self.target_player is not None and self.target_player.position.is_adjacent(self.player.position)):
            return [Path([], 1.0)]

        # If the destination is blocked, we can't get there
        if self.position is not None and self.game.get_player_at(self.position) is not None:
            return []

        # Make initial node
        init_node = Node(self.player.position)
        self.openset.append(init_node)
        self.pareto_frontiers[init_node.position] = ParetoFrontier()
        self.pareto_frontiers[init_node.position].add(init_node)

        # while we have unexpanded nodes
        while len(self.openset) > 0:

            # pull out the first node in our open list
            current = self.openset.first()
            self.openset.remove(current)

            # Check if it's still on the pareto frontier
            if not current in self.pareto_frontiers[current.position].nodes:
                continue

            # Stop if this path can't become better than the best
            if not self._can_beat_best(current):
                continue

            # Expand
            for neighbour in self.game.get_adjacent_squares(current.position, occupied=False):

                if self._target_out_of_reach(current, neighbour):
                    continue

                # Make expanded node
                node = self._get_child(current, neighbour)

                # If a potential blitz position, copy node and add to blitzes
                if self.all and \
                            self.blitz and \
                            self.game.get_adjacent_players(node.position, down=False, team=self.game.get_opp_team(self.player.team)) and \
                            self.blitz and \
                            node.moves < self.max_moves:
                    self._add_blitz(node)

                # Check if goal was reached
                goal_reached = self._goal_reached(node)

                if goal_reached:
                    # Extra move/GFI when blitzing
                    if self.blitz and node.moves == self.max_moves:
                        continue  # No moves left to blitz
                    if self.blitz and node.moves >= self.ma:
                        can_use_rr = self.allow_rr and self.game.can_use_reroll(self.player.team)
                        node.add_moves(1)
                        node.add_gfi_prob(5/6, sure_feet_skill=self.player.has_skill(Skill.SURE_FEET), rr=can_use_rr)
                    # Check if path beats the best
                    if self.best is None:
                        self.best = node
                    elif node.prob > self.best.prob:
                        self.best = node
                    elif node.prob == self.best.prob and node.moves < self.best.moves:
                        self.best = node
                else:

                    # No moves left
                    if current.moves == self.max_moves:
                        continue

                    # Add to pareto frontier
                    if neighbour not in self.pareto_frontiers:
                        self.pareto_frontiers[neighbour] = ParetoFrontier()
                    self.pareto_frontiers[neighbour].add(node)

                    # If it's on the pareto frontier
                    if node in self.pareto_frontiers[neighbour].nodes:

                        # Add it to the open set
                        self.openset.append(node)

        # Search is over - backtrack for goals to find safest path
        if self.all:
            return self._collect_paths()

        if self.best is None:
            return []

        node = self.best
        path = self._collect_path(node)
        return [path]


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
    finder = Pathfinder(game, player, position, allow_rr=allow_team_reroll, blitz=blitz)
    path = finder.get_path()
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
    finder = Pathfinder(game, player, target_x=x, allow_rr=allow_team_reroll, blitz=False)
    path = finder.get_path()
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
    finder = Pathfinder(game, player, allow_rr=allow_team_reroll, target_player=target_player, blitz=blitz)
    path = finder.get_path()
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
    finder = Pathfinder(game, player, allow_rr=allow_team_reroll, blitz=blitz, all=True)
    paths = finder.get_paths()
    if from_position is not None and num_moves_used != 0:
        _reset_state(game, player, orig_player, orig_ball)

    return paths


class FNode:

    def __init__(self, parent, position, moves_left, gfis_left, euclidean_distance, prob, rolls, block_dice=None):
        self.parent = parent
        self.position = position
        self.moves_left = moves_left
        self.gfis_left = gfis_left
        self.euclidean_distance = euclidean_distance
        self.prob = prob
        self.rolls = rolls
        self.block_dice = block_dice

    def __lt__(self, other):
        return self.euclidean_distance < other.euclidean_distance


class Dijkstra:

    # Improvements:
    # Hashsets instead of arrays
    # profiler
    # Blitz moves

    DIRECTIONS = [Square(-1, -1),
                  Square(-1, 0),
                  Square(-1, 1),
                  Square(0, -1),
                  Square(0, 1),
                  Square(1, -1),
                  Square(1, 0),
                  Square(1, 1)]

    def __init__(self, game, player, directly_to_adjacent=False, can_block=False, can_handoff=False, can_foul=False):
        self.game = game
        self.player = player
        self.directly_to_adjacent = directly_to_adjacent
        self.can_block = can_block
        self.can_handoff = can_handoff
        self.can_foul = can_foul
        self.ma = player.get_ma() - player.state.moves
        self.gfis = 3 if player.has_skill(Skill.SPRINT) else 2
        self.locked_nodes = np.full((game.arena.height, game.arena.width), None)
        self.nodes = np.full((game.arena.height, game.arena.width), None)
        self.tzones = np.zeros((game.arena.height, game.arena.width), dtype=np.uint8)
        self.current_prob = 1
        self.open_set = PriorityQueue()
        self.risky_sets = {}
        for p in game.get_players_on_pitch():
            if p.team != player.team and p.has_tackle_zone():
                for square in game.get_adjacent_squares(p.position):
                    self.tzones[square.y][square.x] += 1

    def _get_pickup_target(self, to_pos):
        zones_to = self.tzones[to_pos.y][to_pos.x]
        modifiers = 1
        if not self.player.has_skill(Skill.BIG_HAND):
            modifiers -= zones_to
        if self.game.state.weather == WeatherType.POURING_RAIN:
            if not self.player.has_skill(Skill.BIG_HAND):
                modifiers -= 1
        if self.player.has_skill(Skill.EXTRA_ARMS):
            modifiers += 1
        target = Rules.agility_table[self.player.get_ag()] - modifiers
        return min(6, max(2, target))

    def _get_handoff_target(self, catcher):
        modifiers = self.game.get_catch_modifiers(catcher, handoff=True)
        target = Rules.agility_table[catcher.get_ag()] - modifiers
        return min(6, max(2, target))

    def _get_dodge_target(self, from_pos, to_pos):
        zones_from = self.tzones[from_pos.y][from_pos.x]
        if zones_from == 0:
            return None
        zones_to = self.tzones[to_pos.y][to_pos.x]
        modifiers = 1
        if not self.player.has_skill(Skill.BIG_HAND):
            modifiers -= zones_to
        if self.game.state.weather == WeatherType.POURING_RAIN:
            if not self.player.has_skill(Skill.BIG_HAND):
                modifiers -= 1
        if self.player.has_skill(Skill.EXTRA_ARMS):
            modifiers += 1
        target = Rules.agility_table[self.player.get_ag()] - modifiers
        return min(6, max(2, target))

    def _expand(self, node: FNode):
        if (node.moves_left == 0 and node.gfis_left == 0) or node.block_dice is not None:
            return
        for direction in Dijkstra.DIRECTIONS:
            next_node = self._expand_node(node, direction)
            if next_node is None:
                continue
            rounded_p = round(next_node.prob, 6)
            if rounded_p < self.current_prob:
                self._add_risky_move(rounded_p, next_node)
            else:
                self.open_set.put((next_node.euclidean_distance, next_node))
                self.nodes[next_node.position.y][next_node.position.x] = next_node

    def _expand_node(self, node, direction):
        to_pos = self.game.state.pitch.squares[node.position.y + direction.y][node.position.x + direction.x]
        player_at = self.game.get_player_at(to_pos)
        euclidean_distance = node.euclidean_distance + 1 if direction.x == 0 or direction.y == 0 else node.euclidean_distance + 1.41421
        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        if player_at is not None:
            if player_at.team == self.player.team and self.can_handoff and player_at.can_catch():
                target = self._get_handoff_target(player_at)
                p = node.prob * ((7 - target) / 6)
                next_node = FNode(node, to_pos, 0, 0, node.euclidean_distance, p, [target])
                if best_node is not None and self._best(next_node, best_node) == best_node:
                    return None
                if best_before is not None and self._dominant(next_node, best_before) == best_before:
                    return None
                return next_node
            elif player_at.team != self.player.team and self.can_block and player_at.state.up:
                block_dice = self.game.num_block_dice_at(attacker=self.player, defender=player_at, position=node.position, blitz=True)
                gfi = node.moves_left == 0
                moves_left_next = node.moves_left - 1 if not gfi else node.moves_left
                gfis_left_next = node.gfis_left - 1 if gfi else node.gfis_left
                rolls = []
                p = node.prob
                if gfi:
                    rolls.append(2)
                    p = p * (5 / 6)
                next_node = FNode(node, to_pos, moves_left_next, gfis_left_next, euclidean_distance, p, rolls, block_dice=block_dice)
                if best_node is not None and self._best(next_node, best_node) == best_node:
                    return None
                if best_before is not None and self._dominant(next_node, best_before) == best_before:
                    return None
                return next_node
            elif player_at.team != self.player.team and self.can_foul and not player_at.state.up:
                assists_from, assists_to = self.game.num_assists_at(self.player, player_at, node.position, foul=True)
                target = min(12, max(2, player_at.get_av() + 1 - assists_from + assists_to))
                p = D6.TWO_PROBS[target] if target > 0 else 1
                p = node.prob * p
                next_node = FNode(node, to_pos, 0, 0, node.euclidean_distance, p, [target])
                if best_node is not None and self._best(next_node, best_node) == best_node:
                    return None
                if best_before is not None and self._dominant(next_node, best_before) == best_before:
                    return None
                return next_node
            return None
        if not (1 <= to_pos.x < self.game.arena.width - 1 and 1 <= to_pos.y < self.game.arena.height - 1):
            return None
        gfi = node.moves_left == 0
        moves_left_next = max(0, node.moves_left - 1)
        gfis_left_next = node.gfis_left - 1 if gfi else node.gfis_left
        total_moves_left = moves_left_next + gfis_left_next
        if best_node is not None:
            best_total_moves_left = best_node.moves_left + best_node.gfis_left
            if total_moves_left < best_total_moves_left:
                return None
            if total_moves_left == best_total_moves_left and euclidean_distance > best_node.euclidean_distance:
                return None
        rolls = []
        p = node.prob
        if gfi:
            rolls.append(2)
            p = p * (5 / 6)
        if self.tzones[node.position.y][node.position.x] > 0:
            roll = self._get_dodge_target(node.position, to_pos)
            p = p * ((7 - roll) / 6)
            rolls.append(int(roll))
        if self.game.get_ball_position() == to_pos:
            roll = self._get_pickup_target(to_pos)
            p = p * ((7 - roll) / 6)
            rolls.append(int(roll))
        next_node = FNode(node, to_pos, moves_left_next, gfis_left_next, euclidean_distance, p, rolls)
        if best_before is not None and self._dominant(next_node, best_before) == best_before:
            return None
        return next_node

    def _add_risky_move(self, prob, node):
        if prob not in self.risky_sets:
            self.risky_sets[prob] = []
        self.risky_sets[prob].append(node)

    def get_paths(self):

        ma = self.player.get_ma() - self.player.state.moves
        self.ma = max(0, ma)
        gfis_used = 0 if ma >= 0 else -ma
        self.gfis = 3-gfis_used if self.player.has_skill(Skill.SPRINT) else 2-gfis_used

        if self.ma + self.gfis <= 0:
            return []

        node = FNode(None, self.player.position, self.ma, self.gfis, euclidean_distance=0, prob=1, rolls=[])
        if not self.player.state.up:
            node = self._expand_stand_up(node)
            self.nodes[node.position.y][node.position.x] = node
        self.open_set.put((0, node))
        self._expansion()
        self._clear()

        while len(self.risky_sets) > 0:
            self._prepare_nodes()
            self._expansion()
            self._clear()

        return self._collect_paths()

    def _expand_stand_up(self, node):
        if self.player.has_skill(Skill.JUMP_UP):
            return FNode(node, self.player.position, self.ma, self.gfis, euclidean_distance=0, prob=1, rolls=[])
        elif self.ma < 3:
            roll = max(2, min(6, 4-self.game.get_stand_up_modifier(self.player)))
            p = (7 - roll) / 6
            return FNode(node, self.player.position, 0, self.gfis, euclidean_distance=0, prob=p, rolls=[roll])
        return FNode(node, self.player.position, self.ma-3, self.gfis, euclidean_distance=0, prob=1, rolls=[])

    def _best(self, a: FNode, b: FNode):
        if self.directly_to_adjacent and a.position.distance(self.player.position) == 1 and a.moves_left > b.moves_left:
            return a
        if self.directly_to_adjacent and b.position.distance(self.player.position) == 1 and b.moves_left > a.moves_left:
            return b
        a_moves_left = a.moves_left + a.gfis_left
        b_moves_left = b.moves_left + b.gfis_left
        block = self.can_block and a.block_dice is not None
        if a.prob > b.prob:
            return a
        if b.prob > a.prob:
            return b
        if block and a.block_dice > b.block_dice:
            return a
        if block and b.block_dice > a.block_dice:
            return b
        if a_moves_left > b_moves_left:
            return a
        if b_moves_left > a_moves_left:
            return b
        if a.euclidean_distance < b.euclidean_distance:
            return a
        if b.euclidean_distance < a.euclidean_distance:
            return b
        return None

    def _dominant(self, a: FNode, b: FNode):
        if self.directly_to_adjacent and a.position.distance(self.player.position) == 1 and a.moves_left > b.moves_left:
            return a
        if self.directly_to_adjacent and b.position.distance(self.player.position) == 1 and b.moves_left > a.moves_left:
            return b
        a_moves_left = a.moves_left + a.gfis_left
        b_moves_left = b.moves_left + b.gfis_left
        if a.prob > b.prob and (a.block_dice is None or a.block_dice >= b.block_dice) and (a_moves_left > b_moves_left or (a_moves_left == b_moves_left and a.euclidean_distance < b.euclidean_distance)):
            return a
        if b.prob > a.prob and (b.block_dice is None or b.block_dice >= a.block_dice) and (b_moves_left > a_moves_left or (b_moves_left == a_moves_left and b.euclidean_distance < a.euclidean_distance)):
            return b
        return None

    def _clear(self):
        for y in range(self.game.arena.height):
            for x in range(self.game.arena.width):
                node = self.nodes[y][x]
                if node is not None:
                    before = self.locked_nodes[y][x]
                    if before is None or self._best(node, before) == node:
                        self.locked_nodes[y][x] = node
                    self.nodes[y][x] = None
        self.open_set = PriorityQueue()

    def _prepare_nodes(self):
        if len(self.risky_sets) > 0:
            probs = sorted(self.risky_sets.keys())
            self.current_prob = probs[-1]
            for node in self.risky_sets[probs[-1]]:
                best_before = self.locked_nodes[node.position.y][node.position.x]
                if best_before is not None and self._dominant(best_before, node) == best_before:
                    continue
                existing_node = self.nodes[node.position.y][node.position.x]
                if existing_node is None or self._best(existing_node, node) == node:
                    self.open_set.put((node.euclidean_distance, node))
                    self.nodes[node.position.y][node.position.x] = node
            del self.risky_sets[probs[-1]]

    def _expansion(self):
        while not self.open_set.empty():
            _, best_node = self.open_set.get()
            self._expand(best_node)

    def _collect_paths(self):
        paths = []
        for y in range(self.game.arena.height):
            for x in range(self.game.arena.width):
                node = self.locked_nodes[y][x]
                if node is not None and node.position != self.player.position:
                    prob = node.prob
                    steps = [node.position]
                    rolls = [node.rolls]
                    block_dice = node.block_dice
                    node = node.parent
                    while node is not None:
                        steps.append(node.position)
                        rolls.append(node.rolls)
                        node = node.parent
                    steps = list(reversed(steps))[1:]
                    rolls = list(reversed(rolls))[1:]
                    player_at = self.game.get_player_at(steps[-1])
                    is_foul = self.can_foul and player_at is not None and player_at.team != self.player.team
                    is_handoff = self.can_handoff and player_at is not None and player_at != self.player and player_at.team == self.player.team
                    path = Path(steps, prob=prob, rolls=rolls, block_dice=block_dice, is_foul=is_foul, is_handoff=is_handoff)
                    paths.append(path)
        return paths
