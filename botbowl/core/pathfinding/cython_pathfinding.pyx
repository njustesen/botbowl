# distutils: language=c++

"""
==========================
Author: Mattias Bermell
Year: 2021
==========================
This module contains the pathfinding algorithm implemented in cython
which compiles to a faster module. The algorithm is intended to generate
the exact same result as the python implementation.
"""

import botbowl.core.table as table
import botbowl.core.model as model
import botbowl.core.forward_model as forward_model
from .python_pathfinding import _alter_state, _reset_state
import copy

from libcpp.map cimport map as mapcpp
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue
from libcpp.memory cimport shared_ptr

import cython
cimport cython
from cython.operator import dereference

from .pathing_node cimport Node, Square
ctypedef shared_ptr[Node] NodePtr


cdef object to_botbowl_Square(Square sq):
    return model.Square(sq.x, sq.y)

cdef Square from_botbowl_Square(object sq):
    return Square(sq.x, sq.y)

cdef Square DIRECTIONS[8]
DIRECTIONS[0] = Square(-1,-1)
DIRECTIONS[1] = Square(-1, 0)
DIRECTIONS[2] = Square(-1, 1)
DIRECTIONS[3] = Square( 0,-1)
DIRECTIONS[4] = Square( 0, 1)
DIRECTIONS[5] = Square( 1,-1)
DIRECTIONS[6] = Square( 1, 0)
DIRECTIONS[7] = Square( 1, 1)

cdef int agi_table[11] 
agi_table[0] = 6
agi_table[1] = 6 
agi_table[2] = 5
agi_table[3] = 4
agi_table[4] = 3
agi_table[5] = 2
agi_table[6] = 1
agi_table[7] = 1
agi_table[8] = 1
agi_table[9] = 1
agi_table[10] = 1


cdef class Path:
    cdef:
        NodePtr final_node
        public object _steps, _rolls
        public double prob
        public object block_dice, handoff_roll, foul_roll

    cdef void set_node(self, NodePtr n):
        self.final_node = n
        self.prob = n.get().prob
        self.block_dice = None if n.get().block_dice == 0 else n.get().block_dice
        self.handoff_roll = None if n.get().handoff_roll == 0 else n.get().handoff_roll
        self.foul_roll = None if n.get().foul_roll == 0 else n.get().foul_roll

    cpdef object get_last_step(self):
        if self._steps is None:
            return to_botbowl_Square( self.final_node.get().position )
        else:
            return self._steps[-1]

    @property
    def steps(self):
        if self._steps is None:
            self._collect_path()
        return self._steps

    @property
    def rolls(self):
        if self._rolls is None:
            self._collect_path()
        return self._rolls

    def __len__(self) -> int:
        return len(self.steps)

    def is_empty(self):
        return self.final_node.use_count() == 0

    cdef void _collect_path(self):
        cdef NodePtr node = self.final_node
        cdef list steps = []
        cdef list rolls = []

        while node.get().parent.use_count() > 0:
            steps.append( to_botbowl_Square(node.get().position) )
            rolls.append(node.get().rolls)
            node = node.get().parent
        self._steps = list(reversed(steps))
        self._rolls = list(reversed(rolls))

    def __reduce__(self):
        # Need custom reduce() because built in reduce() can't handle the c++ objects
        def recreate_self(steps, rolls, block_dice, foul_roll, handoff_roll, prob):
            path = Path()
            path._steps = steps
            path._rolls = rolls
            path.block_dice = block_dice
            path.foul_roll = foul_roll
            path.handoff_roll = handoff_roll
            path.prob = prob
            return path
        return recreate_self, (self.steps, self.rolls, self.block_dice, self.foul_roll, self.handoff_roll, self.prob)

    def __eq__(self, other):
        return  self.prob == other.prob and \
                self.steps == other.steps and \
                self.rolls == other.rolls and \
                self.block_dice == other.block_dice and \
                self.handoff_roll == other.handoff_roll and \
                self.foul_roll == other.foul_roll

# Make the forward model treat Path as an immutable type.
forward_model.immutable_types.add(Path)

cdef Path create_path(NodePtr node):
    cdef Path path = Path()
    path.set_node(node)
    return path

cdef class Pathfinder:
    cdef public object game, player, players_on_pitch
    cdef bint trr, can_block, can_handoff, can_foul, target_found, target_is_int, target_is_square, has_target, directly_to_adjacent, is_stunty
    cdef int ma, gfis, dodge_target, pitch_width, pitch_height
    cdef double current_prob
    cdef int tzones[17][28] # init as zero
    cdef Square ball_pos, start_pos
    cdef Square target_square
    cdef int target_x
    cdef NodePtr locked_nodes[17][28] # initalized as empty pointers
    cdef NodePtr nodes[17][28] # initalized as empty pointers
    cdef priority_queue[NodePtr] open_set
    cdef mapcpp[double, vector[NodePtr]] risky_sets 

    def __init__(self, game, player, trr=False, directly_to_adjacent=False, can_block=False, can_handoff=False, can_foul=False):

        self.players_on_pitch = game.state.pitch.board

        self.game = game
        self.pitch_width = self.game.arena.width - 1
        self.pitch_height = game.arena.height -1
        self.start_pos = from_botbowl_Square( player.position )
        ball = self.game.get_ball_position()
        if ball is not None:
            self.ball_pos = from_botbowl_Square(ball)
        else:
            self.ball_pos = Square(-1, -1)

        self.has_target = False
        self.target_is_int = False
        self.target_is_square = False
        self.target_found = False

        self.player = player # todo, assert no skills that aren't handled: twitchy, two_heads, break_tackle, etc..

        self.trr = trr
        self.can_block = can_block
        self.can_handoff = can_handoff
        self.can_foul = can_foul
        self.is_stunty = player.has_skill(table.Skill.STUNTY)
        self.directly_to_adjacent = directly_to_adjacent
        self.dodge_target = agi_table[self.player.get_ag()] - 1
        self.current_prob = 1.0
                
        for p in game.get_players_on_pitch():
            if p.team != player.team and p.has_tackle_zone():
                for square in game.get_adjacent_squares(p.position):
                    self.tzones[square.y][square.x] += 1

    cpdef get_path(self, object target):
        if type(target) == model.Square:
            self.target_square = from_botbowl_Square(target)
            self.target_is_square = True
            self.has_target = True
        elif type(target) == int:
            self.target_x = int(target)
            self.target_is_int = True
            self.has_target = True

        paths = self.get_paths()
        if len(paths) > 0:
            return paths[0]
        return None

    cpdef object get_paths(self):
        cdef:
            Square start_square
            NodePtr node
            int ma, gfis_used
            bint can_dodge, can_sure_feet, can_sure_hands

        self.ma = self.player.num_moves_left()
        self.gfis = self.player.num_gfis_left()

        start_square = from_botbowl_Square(self.player.position)

        can_dodge = self.player.has_skill(table.Skill.DODGE) and table.Skill.DODGE not in self.player.state.used_skills
        can_sure_feet = self.player.has_skill(table.Skill.SURE_FEET) and table.Skill.SURE_FEET not in self.player.state.used_skills
        can_sure_hands = self.player.has_skill(table.Skill.SURE_HANDS)

        # Create root node
        node = NodePtr(new Node(start_square, self.ma, self.gfis, 0, self.trr, can_dodge, can_sure_feet, can_sure_hands, self.can_foul, self.can_block, self.can_handoff))

        if not self.player.state.up:
            node = self._expand_stand_up(node)
            self.nodes[node.get().position.y][node.get().position.x] = node
        self.open_set.push(node)
        self._expansion()
        self._clear()

        while (not self.target_found) and (not self.risky_sets.empty() ):
            self._prepare_nodes()
            self._expansion()
            self._clear()

        return self._collect_paths()

    cdef int _get_pickup_target(self, Square to_pos):
        cdef int target = self.dodge_target + self.tzones[to_pos.y][to_pos.x]
        if self.game.state.weather == table.WeatherType.POURING_RAIN:
            target += 1
        return min(6, max(2, target))

    cdef int _get_handoff_target(self, object catcher):
        cdef int modifiers = self.game.get_catch_modifiers(catcher, handoff=True)
        cdef int target = agi_table[catcher.get_ag()] - modifiers
        return min(6, max(2, target))

    cdef int _get_dodge_target(self, Square from_pos, Square to_pos):
        cdef int target = self.dodge_target + (0 if self.is_stunty else self.tzones[to_pos.y][to_pos.x])
        return min(6, max(2, target))

    cdef void _expand(self, NodePtr node):
        cdef bint out_of_moves = node.get().moves_left + node.get().gfis_left <= 0
        cdef NodePtr next_node
        cdef double rounded_p
        cdef Node * parent = <Node *> node.get().parent.get()
        cdef Square to_square

        if node.get().parent.use_count() == 0 or node.get().parent.get().position == node.get().position:
            parent = NULL

        if self.has_target:
            if self.target_is_square and self.target_square.distance(node.get().position) > node.get().moves_left + node.get().gfis_left:
                return
            if self.target_is_int and abs(self.target_x - node.get().position.x) > node.get().moves_left + node.get().gfis_left:
                return
            if self.target_is_square and node.get().position == self.target_square:
                self.target_found = True
                return
            if self.target_is_int and node.get().position.x == self.target_x:
                self.target_found = True
                return

        if node.get().block_dice != 0 or node.get().handoff_roll != 0:
            return

        if out_of_moves and (not node.get().can_handoff) and (not node.get().can_foul):
            return

        for direction in DIRECTIONS:
            if parent is not NULL:
                to_square = direction + node.get().position
                if parent.position == to_square:
                    continue

                if parent.position.distance(to_square) < 2 and \
                        (self.tzones[to_square.y][to_square.x] == 0 or \
                        self.tzones[parent.position.y][parent.position.x] == 0 ):
                    continue

            next_node = self._expand_node(node, direction, out_of_moves)
            if next_node.use_count() == 0:
                continue
            rounded_p = next_node.get().prob

            if rounded_p < self.current_prob:
                self.risky_sets[rounded_p].push_back(next_node) #add risky move. if 'prob' is not a key, it's inited with default constructor
            else:
                self.open_set.push(next_node)
                self.nodes[next_node.get().position.y][next_node.get().position.x] = next_node

    cdef NodePtr _expand_node(self, NodePtr node, Square direction, bint out_of_moves):
        cdef:
            double euclidean_distance
            Square to_pos

        euclidean_distance = node.get().euclidean_distance + 1.0 if direction.x == 0 or direction.y == 0 else node.get().euclidean_distance + 1.41421
        to_pos = node.get().position + direction
        if not (1 <= to_pos.x < self.pitch_width and 1 <= to_pos.y < self.pitch_height): 
            return NodePtr()
        player_at = self.players_on_pitch[to_pos.y][to_pos.x]

        if player_at is not None:
            if node.get().can_handoff and player_at.team == self.player.team and player_at.can_catch():
                return self._expand_handoff_node(node, to_pos)
            elif node.get().can_block and player_at.team != self.player.team and player_at.state.up:
                return self._expand_block_node(node, euclidean_distance, to_pos, player_at)
            elif node.get().can_foul and player_at.team != self.player.team and not player_at.state.up:
                return self._expand_foul_node(node, to_pos, player_at)
            return NodePtr()
        if not out_of_moves:
            return self._expand_move_node(node, euclidean_distance, to_pos)
        return NodePtr()

    cdef NodePtr _expand_move_node(self, NodePtr node, double euclidean_distance, Square to_pos):
        cdef:
            NodePtr best_node, best_before, next_node
            bint gfi
            int target, moves_left_next, gfis_left_next, total_moves_left

        gfi = node.get().moves_left == 0
        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]

        moves_left_next = max(0, node.get().moves_left - 1)
        gfis_left_next = node.get().gfis_left - 1 if gfi else node.get().gfis_left
        total_moves_left = moves_left_next + gfis_left_next
        if best_node.use_count()>0:
            best_total_moves_left = best_node.get().moves_left + best_node.get().gfis_left
            if total_moves_left <= best_total_moves_left:
                return NodePtr()

        next_node = NodePtr(new Node(node, to_pos, moves_left_next, gfis_left_next, euclidean_distance))
        if gfi:
            next_node.get().apply_gfi()
        if self.tzones[node.get().position.y][node.get().position.x] > 0:
            target = self._get_dodge_target(node.get().position, to_pos)
            next_node.get().apply_dodge(target)
        if to_pos == self.ball_pos:
            target = self._get_pickup_target(to_pos)
            next_node.get().apply_pickup(target)
        if best_before.use_count()>0 and self._dominant(next_node, best_before) == best_before:
            return NodePtr()
        return next_node

    cdef NodePtr _expand_foul_node(self, NodePtr node, Square to_pos, object player_at):
        cdef:
            NodePtr best_node, best_before, next_node
            int target, assists_from, assists_to

        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        assists_from, assists_to = self.game.num_assists_at(self.player, player_at, to_botbowl_Square(node.get().position), foul=True)
        target = min(12, max(2, player_at.get_av() + 1 - assists_from + assists_to))
        next_node = NodePtr( new Node(node, to_pos, 0, 0, node.get().euclidean_distance) )
        next_node.get().apply_foul(target)

        if best_node.use_count()>0 and self._best(next_node, best_node) == best_node:
            return NodePtr()
        if best_before.use_count()>0 and self._dominant(next_node, best_before) == best_before:
            return NodePtr()
        return next_node

    cdef NodePtr _expand_handoff_node(self, NodePtr node, Square to_pos):
        cdef:
            NodePtr best_node, best_before, next_node
            int target
            object player_at
        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        player_at = self.players_on_pitch[to_pos.y][to_pos.x]

        next_node = NodePtr( new Node(node, to_pos, 0, 0, node.get().euclidean_distance))
        target = self._get_handoff_target(player_at)
        next_node.get().apply_handoff(target)

        if best_node.use_count()>0 and self._best(next_node, best_node) == best_node:
            return NodePtr()
        if best_before.use_count()>0 and self._dominant(next_node, best_before) == best_before:
            return NodePtr()
        return next_node

    cdef NodePtr _expand_block_node(self, NodePtr node, double euclidean_distance, Square to_pos, object player_at):
        cdef:
            NodePtr best_node, best_before, next_node
            int target, moves_left_next, gfis_left_next, block_dice
            bint gfi

        best_node = self.nodes[to_pos.y][to_pos.x]
        best_before = self.locked_nodes[to_pos.y][to_pos.x]
        block_dice = self.game.num_block_dice_at(attacker=self.player, defender=player_at,
                                                 position= to_botbowl_Square( node.get().position), blitz=True)
        gfi = node.get().moves_left == 0
        moves_left_next = node.get().moves_left - 1 if not gfi else node.get().moves_left
        gfis_left_next = node.get().gfis_left - 1 if gfi else node.get().gfis_left
        next_node = NodePtr( new Node(node, to_pos, moves_left_next, gfis_left_next, euclidean_distance, block_dice))
        next_node.get().can_block = False

        if gfi:
            node.get().apply_gfi()
        if best_node.use_count()>0 and self._best(next_node, best_node) == best_node:
            return NodePtr()
        if best_before.use_count()>0 and self._dominant(next_node, best_before) == best_before:
            return NodePtr()
        return next_node

    cdef NodePtr _expand_stand_up(self, NodePtr node):
        cdef:
            int target
            NodePtr next_node

        if self.player.has_skill(table.Skill.JUMP_UP):
            return NodePtr (new Node(node, from_botbowl_Square(self.player.position), self.ma, self.gfis, 0))
        elif self.ma < 3:
            target = max(2, min(6, 4-self.game.get_stand_up_modifier(self.player)))
            next_node = NodePtr (new Node(node, from_botbowl_Square(self.player.position), 0, self.gfis, 0))
            node.get().apply_stand_up(target)
            return next_node
        next_node = NodePtr (new Node(node, self.start_pos, self.ma - 3, self.gfis, 0))
        return next_node

    cdef NodePtr _best(self, NodePtr a, NodePtr b):
        cdef:
            int a_moves_left, b_moves_left

        # Directly to adjescent
        if self.directly_to_adjacent and a.get().position.distance( self.start_pos ) == 1 and a.get().moves_left > b.get().moves_left:
            return a
        if self.directly_to_adjacent and b.get().position.distance( self.start_pos ) == 1 and b.get().moves_left > a.get().moves_left:
            return b

        if a.get().prob > b.get().prob:
            return a
        if b.get().prob > a.get().prob:
            return b

        if a.get().foul_roll != 0:
            if a.get().foul_roll < b.get().foul_roll:
                return a
            if b.get().foul_roll < a.get().foul_roll:
                return b

        if a.get().block_dice != 0:
            if a.get().block_dice > b.get().block_dice:
                return a
            if b.get().block_dice > a.get().block_dice:
                return b

        a_moves_left = a.get().moves_left + a.get().gfis_left
        b_moves_left = b.get().moves_left + b.get().gfis_left

        if a_moves_left > b_moves_left:
            return a
        if b_moves_left > a_moves_left:
            return b
        if a.get().euclidean_distance < b.get().euclidean_distance:
            return a
        if b.get().euclidean_distance < a.get().euclidean_distance:
            return b
        return NodePtr()

    cdef NodePtr _dominant(self, NodePtr a, NodePtr b):
        if self.directly_to_adjacent and a.get().position.distance( self.start_pos ) == 1 and a.get().moves_left > b.get().moves_left:
            return a
        if self.directly_to_adjacent and b.get().position.distance( self.start_pos ) == 1 and b.get().moves_left > a.get().moves_left:
            return b

        a_moves_left = a.get().moves_left + a.get().gfis_left
        b_moves_left = b.get().moves_left + b.get().gfis_left
        # TODO: Write out as above
        if a.get().prob > b.get().prob and (a.get().foul_roll==0 or a.get().foul_roll <= b.get().foul_roll) and (a.get().block_dice==0 or a.get().block_dice >= b.get().block_dice) and (a_moves_left > b_moves_left or (a_moves_left == b_moves_left and a.get().euclidean_distance < b.get().euclidean_distance)):
            return a
        if b.get().prob > a.get().prob and (b.get().foul_roll==0 or b.get().foul_roll <= a.get().foul_roll) and (b.get().block_dice==0 or b.get().block_dice >= a.get().block_dice) and (b_moves_left > a_moves_left or (b_moves_left == a_moves_left and b.get().euclidean_distance < a.get().euclidean_distance)):
            return b
        return NodePtr()

    cdef void _clear(self):
        cdef NodePtr node, before

        for y in range(17):
            for x in range(28):
                node = self.nodes[y][x]
                if node.use_count()>0:
                    before = self.locked_nodes[y][x]
                    if before.use_count() == 0 or self._best(node, before) == node:
                        self.locked_nodes[y][x] = node
                    self.nodes[y][x] = NodePtr()
        self.open_set = priority_queue[NodePtr]()

    cdef void _prepare_nodes(self):
        cdef NodePtr node, existing_node, best_before

        if not self.risky_sets.empty():
            self.current_prob = dereference(self.risky_sets.rbegin()).first # get highest probability in risky_sets
            for node in self.risky_sets[self.current_prob]:
                best_before = self.locked_nodes[node.get().position.y][node.get().position.x]
                if best_before.use_count()>0 and self._dominant(best_before, node) == best_before:
                    continue
                existing_node = self.nodes[node.get().position.y][node.get().position.x]
                if existing_node.use_count() == 0 or self._best(existing_node, node) == node:
                    self.open_set.push(node)
                    self.nodes[node.get().position.y][node.get().position.x] = node
            self.risky_sets.erase(self.current_prob)

    cdef void _expansion(self):
        cdef NodePtr best_node
        while not self.open_set.empty():
            best_node = self.open_set.top()
            self.open_set.pop()
            self._expand(best_node)

    cdef object _collect_paths(self):
        cdef:
            NodePtr node
            Path path
            list paths

        if self.target_is_square:
            node = self.locked_nodes[self.target_square.y][self.target_square.x]
            if node.use_count()>0:
                return [create_path(node)]
            return []

        paths = []
        for y in range(17):
            for x in range(28):
                if self.start_pos.x == x and self.start_pos.y == y:
                    continue
                node = self.locked_nodes[y][x]
                if node.use_count()> 0:
                    paths.append(create_path(node))
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
