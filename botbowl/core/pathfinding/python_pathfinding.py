"""
==========================
Author: Niels Justesen
Year: 2020
==========================
This module contains pathfinding functionalities for botbowl.
"""

from botbowl.core.table import Rules
from botbowl.core.model import Square
from botbowl.core.forward_model import treat_as_immutable
from botbowl.core.table import Skill, WeatherType
import copy
import numpy as np
from queue import PriorityQueue


@treat_as_immutable
class Path:

    def __init__(self, node: 'Node'):
        super().__init__()
        self.final_node = node
        self._steps = None
        self.prob = node.prob
        self._rolls = None
        self.block_dice = node.block_dice
        self.handoff_roll = node.handoff_roll
        self.foul_roll = node.foul_roll

    @property
    def steps(self):
        if self._steps is None:
            self.collect_path()
        return self._steps

    @property
    def rolls(self):
        if self._rolls is None:
            self.collect_path()
        return self._rolls

    def __len__(self) -> int:
        return len(self.steps)

    def get_last_step(self) -> 'Square':
        return self.final_node.position

    def is_empty(self) -> bool:
        return len(self) == 0

    def collect_path(self):
        steps = []
        rolls = []
        node = self.final_node

        while node.parent is not None:
            steps.append(node.position)
            rolls.append(node.rolls)
            node = node.parent
        self._steps = list(reversed(steps))
        self._rolls = list(reversed(rolls))

    def __eq__(self, other):
        return self.prob == other.prob and \
               self.steps == other.steps and \
               self.rolls == other.rolls and \
               self.block_dice == other.block_dice and \
               self.handoff_roll == other.handoff_roll and \
               self.foul_roll == other.foul_roll


class Node:

    TRR = 0
    DODGE = 1
    SURE_FEET = 2
    SURE_HANDS = 3

    def __lt__(self, other):
        return self.euclidean_distance < other.euclidean_distance

    def __init__(self,
                 parent,
                 position,
                 moves_left,
                 gfis_left,
                 euclidean_distance,
                 rr_states=None,
                 block_dice=None,
                 foul_roll=None,
                 handoff_roll=None,
                 can_foul=False,
                 can_block=False,
                 can_handoff=False):
        self.parent = parent
        self.position = position
        self.moves_left = moves_left
        self.gfis_left = gfis_left
        self.euclidean_distance = euclidean_distance
        self.prob = parent.prob if parent is not None else 1
        self.foul_roll = foul_roll
        self.handoff_roll = handoff_roll
        self.rolls = []
        self.block_dice = block_dice
        self.rr_states = rr_states if rr_states is not None else parent.rr_states
        if parent is not None:
            self.can_foul = parent.can_foul
            self.can_block = parent.can_block
            self.can_handoff = parent.can_handoff
        else:
            self.can_foul = can_foul
            self.can_block = can_block
            self.can_handoff = can_handoff

    def _apply_roll(self, p, skill_rr, team_rr):
        # Find new states
        new_states = {}
        for state, prev_p in self.rr_states.items():
            p_success = prev_p * p
            if state in new_states:
                new_states[state] += p_success
            else:
                new_states[state] = prev_p * p
            if skill_rr is not None and state[skill_rr]:
                self._add_fail_state(new_states, state, prev_p, p, skill_rr)
            elif state[team_rr]:
                self._add_fail_state(new_states, state, prev_p, p, team_rr)
        '''
        # Merge new states with previous states
        for rr_state, rr_state_p in new_rr_states.items():
            if rr_state in self.rr_states:
                self.rr_states[rr_state] += rr_state_p
            else:
                self.rr_states[rr_state] = rr_state_p
        '''
        # Merge with self.rr_state
        self.rr_states = new_states
        self.prob = sum(self.rr_states.values())

    def _add_fail_state(self, new_states, prev_state, prev_state_p, p, index):
        fail_state = [rr for rr in prev_state]
        fail_state[index] = False
        fail_state_p = prev_state_p * (1 - p) * p
        fail_state = tuple(fail_state)
        if fail_state in new_states:
            new_states[fail_state] += fail_state_p
        else:
            new_states[fail_state] = fail_state_p

    def apply_gfi(self):
        self.rolls.append(2)
        self._apply_roll(5 / 6, self.SURE_FEET, self.TRR)

    def apply_dodge(self, target):
        self.rolls.append(target)
        self._apply_roll((7 - target) / 6, self.DODGE, self.TRR)

    def apply_pickup(self, target):
        self.rolls.append(target)
        self._apply_roll((7 - target) / 6, self.SURE_HANDS, self.TRR)
        # TODO: should pickup be added to path prob if it's the last step?

    def apply_handoff(self, target):
        self.handoff_roll = target
        self.can_handoff = False

    def apply_foul(self, target):
        self.foul_roll = target
        self.can_foul = False

    def apply_stand_up(self, target):
        self.rolls.append(target)
        self._apply_roll((7 - target) / 6, None, self.TRR)


class Pathfinder:

    DIRECTIONS = [Square(-1, -1),
                  Square(-1, 0),
                  Square(-1, 1),
                  Square(0, -1),
                  Square(0, 1),
                  Square(1, -1),
                  Square(1, 0),
                  Square(1, 1)]

    def __init__(self, game, player, trr=False, directly_to_adjacent=False, can_block=False, can_handoff=False, can_foul=False):
        self.game = game
        self.player = player
        self.trr = trr
        self.directly_to_adjacent = directly_to_adjacent
        self.can_block = can_block
        self.can_handoff = can_handoff
        self.can_foul = can_foul
        self.ma = player.num_moves_left()
        self.gfis = player.num_gfis_left()
        self.locked_nodes = np.full((game.arena.height, game.arena.width), None)
        self.nodes = np.full((game.arena.height, game.arena.width), None)
        self.tzones = np.zeros((game.arena.height, game.arena.width), dtype=np.uint8)
        self.current_prob = 1
        self.open_set = PriorityQueue()
        self.risky_sets = {}
        self.target_found = False
        for p in game.get_players_on_pitch():
            if p.team != player.team and p.has_tackle_zone():
                for square in game.get_adjacent_squares(p.position):
                    self.tzones[square.y][square.x] += 1

    def get_path(self, target):
        paths = self.get_paths(target)
        if len(paths) > 0:
            return paths[0]
        return None

    def get_paths(self, target=None):
        self.gfis = self.player.num_gfis_left()
        self.ma = self.player.num_moves_left()
        can_dodge = self.player.has_skill(Skill.DODGE) and Skill.DODGE not in self.player.state.used_skills
        can_sure_feet = self.player.has_skill(Skill.SURE_FEET) and Skill.SURE_FEET not in self.player.state.used_skills
        can_sure_hands = self.player.has_skill(Skill.SURE_HANDS)
        rr_states = {(self.trr, can_dodge, can_sure_feet, can_sure_hands): 1}  # RRs left and probability of success
        node = Node(None,
                    position=self.player.position,
                    moves_left=self.ma,
                    gfis_left=self.gfis,
                    euclidean_distance=0,
                    rr_states=rr_states,
                    can_foul=self.can_foul,
                    can_handoff=self.can_handoff,
                    can_block=self.can_block)
        if not self.player.state.up:
            node = self._expand_stand_up(node)
            self.nodes[node.position.y][node.position.x] = node
        self.open_set.put((0, node))
        self._expansion(target)
        self._clear()

        while not self.target_found and len(self.risky_sets) > 0:
            self._prepare_nodes()
            self._expansion(target)
            self._clear()

        return self._collect_paths(target)

    def _get_pickup_target(self, to_pos):
        zones_to = self.tzones[to_pos.y][to_pos.x]
        modifiers = 1
        if not self.player.has_skill(Skill.BIG_HAND):
            modifiers -= int(zones_to)
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
        zones_to = int(self.tzones[to_pos.y][to_pos.x])
        modifiers = 1

        ignore_opp_mods = False
        if self.player.has_skill(Skill.STUNTY):
            modifiers = 1
            ignore_opp_mods = True
        if self.player.has_skill(Skill.TITCHY):
            modifiers += 1
            ignore_opp_mods = True
        if self.player.has_skill(Skill.TWO_HEADS):
            modifiers += 1

        if not ignore_opp_mods:
            modifiers -= zones_to

        target = Rules.agility_table[self.player.get_ag()] - modifiers
        return min(6, max(2, target))

    def _expand(self, node: Node, target=None):
        if target is not None:
            # TODO: handoff?
            if type(target) == Square and target.distance(node.position) > node.moves_left + node.gfis_left:
                return
            if type(target) == int and abs(target - node.position.x) > node.moves_left + node.gfis_left:
                return
            if type(target) == Square and node.position == target:
                self.target_found = True
                return
            if type(target) == int and node.position.x == target:
                self.target_found = True
                return

        if node.block_dice is not None or node.handoff_roll is not None:
            return

        out_of_moves = False
        if node.moves_left + node.gfis_left <= 0:
            if not node.can_handoff and not node.can_foul:
                return
            out_of_moves = True

        for direction in self.DIRECTIONS:
            next_node = self._expand_node(node, direction, out_of_moves=out_of_moves)
            if next_node is None:
                continue
            rounded_p = round(next_node.prob, 6)
            if rounded_p < self.current_prob:
                self._add_risky_move(rounded_p, next_node)
            else:
                self.open_set.put((next_node.euclidean_distance, next_node))
                self.nodes[next_node.position.y][next_node.position.x] = next_node

    def _expand_node(self, node, direction, out_of_moves=False):
        euclidean_distance = node.euclidean_distance + 1 if direction.x == 0 or direction.y == 0 else node.euclidean_distance + 1.41421
        to_pos = self.game.state.pitch.squares[node.position.y + direction.y][node.position.x + direction.x]
        if not (1 <= to_pos.x < self.game.arena.width - 1 and 1 <= to_pos.y < self.game.arena.height - 1):
            return None
        player_at = self.game.get_player_at(to_pos)
        if player_at is not None:
            if player_at.team == self.player.team and node.can_handoff and player_at.can_catch():
                return self._expand_handoff_node(node, to_pos)
            elif player_at.team != self.player.team and node.can_block and player_at.state.up:
                return self._expand_block_node(node, euclidean_distance, to_pos, player_at)
            elif player_at.team != self.player.team and node.can_foul and not player_at.state.up:
                return self._expand_foul_node(node, to_pos, player_at)
            return None
        if not out_of_moves:
            return self._expand_move_node(node, euclidean_distance, to_pos)
        return None

    def _expand_move_node(self, node, euclidean_distance, to_pos):
        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        gfi = node.moves_left == 0
        moves_left_next = max(0, node.moves_left - 1)
        gfis_left_next = node.gfis_left - 1 if gfi else node.gfis_left
        total_moves_left = moves_left_next + gfis_left_next
        if best_node is not None:
            best_total_moves_left = best_node.moves_left + best_node.gfis_left
            if total_moves_left < best_total_moves_left:
                return None
            if total_moves_left == best_total_moves_left and euclidean_distance >= best_node.euclidean_distance:
                return None
        next_node = Node(node, to_pos, moves_left_next, gfis_left_next, euclidean_distance)
        if gfi:
            next_node.apply_gfi()
        if self.tzones[node.position.y][node.position.x] > 0:
            target = self._get_dodge_target(node.position, to_pos)
            next_node.apply_dodge(target)
        if self.game.get_ball_position() == to_pos:
            target = self._get_pickup_target(to_pos)
            next_node.apply_pickup(target)
        if best_before is not None and self._dominant(next_node, best_before) == best_before:
            return None
        return next_node

    def _expand_foul_node(self, node, to_pos, player_at):
        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        assists_from, assists_to = self.game.num_assists_at(self.player, player_at, node.position, foul=True)
        target = min(12, max(2, player_at.get_av() + 1 - assists_from + assists_to))
        next_node = Node(node, to_pos, 0, 0, node.euclidean_distance)
        next_node.apply_foul(target)
        if best_node is not None and self._best(next_node, best_node) == best_node:
            return None
        if best_before is not None and self._dominant(next_node, best_before) == best_before:
            return None
        return next_node

    def _expand_handoff_node(self, node, to_pos):
        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        player_at = self.game.get_player_at(to_pos)
        next_node = Node(node, to_pos, 0, 0, node.euclidean_distance)
        target = self._get_handoff_target(player_at)
        next_node.apply_handoff(target)
        if best_node is not None and self._best(next_node, best_node) == best_node:
            return None
        if best_before is not None and self._dominant(next_node, best_before) == best_before:
            return None
        return next_node

    def _expand_block_node(self, node, euclidean_distance, to_pos, player_at):
        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        block_dice = self.game.num_block_dice_at(attacker=self.player, defender=player_at, position=node.position,
                                                 blitz=True)
        gfi = node.moves_left == 0
        moves_left_next = node.moves_left - 1 if not gfi else node.moves_left
        gfis_left_next = node.gfis_left - 1 if gfi else node.gfis_left
        next_node = Node(node, to_pos, moves_left_next, gfis_left_next, euclidean_distance, block_dice=block_dice,
                         can_block=False)
        if gfi:
            next_node.apply_gfi()
        if best_node is not None and self._best(next_node, best_node) == best_node:
            return None
        if best_before is not None and self._dominant(next_node, best_before) == best_before:
            return None
        return next_node

    def _add_risky_move(self, prob, node):
        if prob not in self.risky_sets:
            self.risky_sets[prob] = []
        self.risky_sets[prob].append(node)

    def _expand_stand_up(self, node):
        if self.player.has_skill(Skill.JUMP_UP):
            return Node(node, self.player.position, self.ma, self.gfis, euclidean_distance=0)
        elif self.ma < 3:
            target = max(2, min(6, 4-self.game.get_stand_up_modifier(self.player)))
            next_node = Node(node, self.player.position, 0, self.gfis, euclidean_distance=0)
            next_node.apply_stand_up(target)
            return next_node
        next_node = Node(node, self.player.position, self.ma - 3, self.gfis, euclidean_distance=0)
        return next_node

    def _best(self, a: Node, b: Node):
        if self.directly_to_adjacent and a.position.distance(self.player.position) == 1 and a.moves_left > b.moves_left:
            return a
        if self.directly_to_adjacent and b.position.distance(self.player.position) == 1 and b.moves_left > a.moves_left:
            return b
        a_moves_left = a.moves_left + a.gfis_left
        b_moves_left = b.moves_left + b.gfis_left
        block = a.block_dice is not None
        foul = a.foul_roll is not None
        if a.prob > b.prob:
            return a
        if b.prob > a.prob:
            return b
        if foul and a.foul_roll < b.foul_roll:
            return a
        if foul and b.foul_roll < a.foul_roll:
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

    def _dominant(self, a: Node, b: Node):
        if self.directly_to_adjacent and a.position.distance(self.player.position) == 1 and a.moves_left > b.moves_left:
            return a
        if self.directly_to_adjacent and b.position.distance(self.player.position) == 1 and b.moves_left > a.moves_left:
            return b
        a_moves_left = a.moves_left + a.gfis_left
        b_moves_left = b.moves_left + b.gfis_left
        # TODO: Write out as above
        if a.prob > b.prob and (a.foul_roll is None or a.foul_roll <= b.foul_roll) and (a.block_dice is None or a.block_dice >= b.block_dice) and (a_moves_left > b_moves_left or (a_moves_left == b_moves_left and a.euclidean_distance < b.euclidean_distance)):
            return a
        if b.prob > a.prob and (b.foul_roll is None or b.foul_roll <= a.foul_roll) and (b.block_dice is None or b.block_dice >= a.block_dice) and (b_moves_left > a_moves_left or (b_moves_left == a_moves_left and b.euclidean_distance < a.euclidean_distance)):
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

    def _expansion(self, target=None):
        while not self.open_set.empty():
            _, best_node = self.open_set.get()
            self._expand(best_node, target)

    def _collect_paths(self, target=None):
        if type(target) == Square:
            node = self.locked_nodes[target.y][target.x]
            if node is not None:
                return [Path(node)]
            return []
        paths = []
        for y in range(self.game.arena.height):
            for x in range(self.game.arena.width):
                if self.player.position.x == x and self.player.position.y == y:
                    continue
                if type(target) == int and not target == x:
                    continue
                node = self.locked_nodes[y][x]
                if node is not None:
                    paths.append(Path(node))
        return paths


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
    can_handoff = game.is_handoff_available() and game.get_ball_carrier() == player
    finder = Pathfinder(game, player, trr=allow_team_reroll, can_block=blitz, can_handoff=can_handoff)
    path = finder.get_path(target=position)
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
    finder = Pathfinder(game, player, trr=allow_team_reroll)
    path = finder.get_path(target=x)
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
    finder = Pathfinder(game, player, trr=allow_team_reroll, can_block=blitz)
    paths = finder.get_paths()
    if from_position is not None and num_moves_used != 0:
        _reset_state(game, player, orig_player, orig_ball)

    return paths


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
