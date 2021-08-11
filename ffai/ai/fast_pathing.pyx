# distutils: language=c++

import cython
cimport cython
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue


from libcpp.map cimport map as mapcpp
#from libcpp.functional.less cimport less
from cython.operator import dereference, postincrement

# import Path class? Because reversable


cdef struct Square:
    int x
    int y

ctypedef unsigned int Reroll_State

ctypedef mapcpp[Reroll_State, float] Rr_state_map
ctypedef mapcpp[Reroll_State, float].iterator Rr_state_map_iter

cdef Reroll_State NO_RR = 0
cdef Reroll_State TRR = 1
cdef Reroll_State DODGE = 2
cdef Reroll_State SURE_FEET = 4
cdef Reroll_State SURE_HANDS = 8

cdef struct Node:
    Node* parent
    Square* position
    int moves_left, gfis_left, foul_roll, handoff_roll, block_dice
    float euclidean_distance, prob
    vector[int] rolls
    Rr_state_map rr_states


cdef Node create_default_node(Node* parent, Square* position, int moves_left, int gfis_left,
                             float euclidean_distance, int block_dice=-1, int foul_roll=-1, int handoff_roll=-1):
    cdef Node self

    cdef vector[int] rolls

    self.parent = parent
    self.position = position
    self.moves_left = moves_left
    self.gfis_left = gfis_left
    self.euclidean_distance = euclidean_distance
    self.foul_roll = foul_roll
    self.handoff_roll = handoff_roll
    self.rolls = rolls
    self.block_dice = block_dice
    return self


cdef Node create_root_node(Square* position, int moves_left, int gfis_left,
                            float euclidean_distance, Rr_state_map rr_states):

    cdef Node self = create_default_node(NULL, position, moves_left, gfis_left, euclidean_distance)
    self.prob = 1.0
    self.rr_states = rr_states

    return self

cdef Node create_node(Node* parent, Square* position, int moves_left, int gfis_left,
                      float euclidean_distance, int block_dice=-1, int foul_roll=-1, int handoff_roll=-1):

    cdef Node self = create_default_node(parent, position, moves_left, gfis_left, euclidean_distance, block_dice, foul_roll, handoff_roll)
    self.prob = parent.prob
    self.rr_states = parent.rr_states

    return self


cdef void _apply_roll(Node* self, float p, int skill_rr, int team_rr):

    cdef Rr_state_map new_states
    cdef Rr_state_map_iter it = self.rr_states.begin()
    cdef float p_success

    cdef Reroll_State state
    cdef float prev_p

    while it != self.rr_states.end():
        state = dereference(it).first
        prev_p = dereference(it).second

        p_success = prev_p * p
        if new_states.find(state) != new_states.end(): # state in new_states
            new_states[state] += p_success
        else:
            new_states[state] = prev_p * p

        if skill_rr != 0 and (state & skill_rr):
            _add_fail_state(&new_states, state, prev_p, p, skill_rr)
        elif (state & team_rr):
            _add_fail_state(&new_states, state, prev_p, p, team_rr)

        postincrement(it)

    # Merge with self.rr_state
    self.rr_states = new_states

    #sum(self.rr_states.values())
    self.prob = 0
    it = self.rr_states.begin()
    while it != self.rr_states.end():
        self.prob += dereference(it).second
        postincrement(it)

cdef _add_fail_state(Rr_state_map * new_states, Reroll_State prev_state, float prev_state_p, float p, int index):
    cdef Reroll_State fail_state = prev_state ^ index
    cdef float fail_state_p = prev_state_p * (1 - p) * p
    if new_states.find(fail_state) != new_states.end():
        dereference(new_states)[fail_state] += fail_state_p
    else:
        dereference(new_states)[fail_state] = fail_state_p


cdef apply_gfi(Node * self):
    self.rolls.push_back(2)
    _apply_roll(self, 5.0 / 6.0, SURE_FEET, TRR)


cdef apply_dodge(Node * self, int target):
    self.rolls.push_back(target)
    _apply_roll(self, (7 - target) / 6.0, SURE_FEET, TRR)

cdef apply_pickup(Node * self, int target):
    self.rolls.push_back(target)
    _apply_roll(self, (7 - target) / 6.0, SURE_HANDS, TRR)

cdef apply_handoff(Node * self, int target):
    self.rolls.push_back(target)

cdef apply_foul(Node * self, int target):
    self.rolls.push_back(target)

cdef apply_stand_up(Node * self, int target):
    self.rolls.push_back(target)
    _apply_roll(self, (7 - target) / 6.0, NO_RR, TRR)


cdef Square DIRECTIONS[8]
DIRECTIONS[0].x = -1; DIRECTIONS[0].y = -1
DIRECTIONS[1].x = -1; DIRECTIONS[1].y = 0
DIRECTIONS[2].x = -1; DIRECTIONS[2].y = 1
DIRECTIONS[3].x = 0; DIRECTIONS[3].y = -1
DIRECTIONS[4].x = 0; DIRECTIONS[4].y = 1
DIRECTIONS[5].x = 1; DIRECTIONS[5].y = -1
DIRECTIONS[6].x = 1; DIRECTIONS[6].y = 0
DIRECTIONS[7].x = 1; DIRECTIONS[7].y = 1


cdef class Pathfinder:
    cdef public object game
    cdef public object player
    cdef object trr
    cdef object directly_to_adjacent
    cdef object can_block
    cdef object can_handoff
    cdef object can_foul
    cdef object ma
    cdef object gfis
    cdef Node* locked_nodes[17][28] # initalized as NULL by default
    cdef Node* nodes[17][28] # initalized as NULL by default
    cdef int tzones[17][28]
    cdef object current_prob
    cdef object open_set
    cdef object risky_set
    cdef object target_found

    def __init__(self, game, player, trr=False, directly_to_adjacent=False, can_block=False, can_handoff=False, can_foul=False):
        self.game = game
        self.player = player
        self.trr = trr
        self.directly_to_adjacent = directly_to_adjacent
        self.can_block = can_block
        self.can_handoff = can_handoff
        self.can_foul = can_foul
        self.ma = player.get_ma() - player.state.moves
        # self.gfis = 3 if player.has_skill(Skill.SPRINT) else 2
        self.current_prob = 1
        # self.open_set = PriorityQueue()
        self.risky_sets = {}
        self.target_found = False
        for p in game.get_players_on_pitch():
            if p.team != player.team and p.has_tackle_zone():
                for square in game.get_adjacent_squares(p.position):
                    self.tzones[square.y][square.x] += 1

"""
    def get_path(self, target):

    cdef get_paths(self, target=None):

    cdef _get_pickup_target(self, to_pos):

    cdef _get_handoff_target(self, catcher):

    cdef _get_dodge_target(self, from_pos, to_pos):

    cdef _expand(self, node: Node, target=None):

    cdef _expand_node(self, node, direction, out_of_moves=False):

    cdef _expand_move_node(self, node, euclidean_distance, to_pos):

    cdef _expand_foul_node(self, node, to_pos, player_at):

    cdef _expand_handoff_node(self, node, to_pos):

    cdef _expand_block_node(self, node, euclidean_distance, to_pos, player_at):

    cdef _add_risky_move(self, prob, node):

    cdef _expand_stand_up(self, node):

    cdef _best(self, a: Node, b: Node):

    cdef _dominant(self, a: Node, b: Node):

    cdef _clear(self):

    cdef _prepare_nodes(self):

    cdef _expansion(self, target=None):

    cdef _collect_paths(self, target=None):

    cdef _collect_path(self, node):
"""

#cpdef get_all_paths(...)