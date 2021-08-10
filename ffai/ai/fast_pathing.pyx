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


"""
cpdef apply_gfi(self)

cpdef apply_dodge(self, int target)

cpdef apply_pickup(self, int target)

cpdef apply_handoff(self, int target)

cpdef apply_foul(self, int target)

cpdef apply_stand_up(self, int target)


cdef class Pathfinder:
    cdef public object game
    cdef public object player
    cdef public object trr
    cdef public object directly_to_adjacent
    cdef public object can_block
    cdef public object can_handoff
    cdef public object can_foul
    cdef public object ma
    cdef public object gfis
    cdef public object locked_nodes
    cdef public object nodes
    cdef public object tzones
    cdef public object current_prob
    cdef public object open_set
    cdef public object risky_set
    cdef public object target_found

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