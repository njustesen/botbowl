# distutils: language=c++

import cython
cimport cython
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue
from libcpp.map cimport map as mapcpp

# import Path class? Because reversable


cdef struct Square:
    int x
    int y


cdef int TRR = 0
cdef int DODGE = 1
cdef int SURE_FEET = 2
cdef int SURE_HANDS = 3


cdef struct Node:
    Node* parent
    Square* position
    int moves_left, gfis_left, foul_roll, handoff_roll, block_dice
    float euclidean_distance, prob
    vector[int] rolls
    #dict rr_states

#cdef inline int hash_rr_states(


cdef Node create_node(Node* parent,
                 Square* position,
                 int moves_left,
                 int gfis_left,
                 float euclidean_distance,
                 dict rr_states=None,
                 int block_dice=-1,
                 int foul_roll=-1,
                 int handoff_roll=-1):

    cdef vector[int] rolls
    cdef float prob = 1.0
    cdef Node self
    cdef mapcpp[(int, int),float] mattias

    if parent is not NULL:
        prob = parent.prob

    self.parent = parent
    self.position = position
    self.moves_left = moves_left
    self.gfis_left = gfis_left
    self.euclidean_distance = euclidean_distance
    self.prob = prob
    self.foul_roll = foul_roll
    self.handoff_roll = handoff_roll
    self.rolls = rolls
    self.block_dice = block_dice
    #self.rr_states = ???


    #self.rr_states = rr_states if rr_states is not None else parent.rr_states

    return self

cdef _apply_roll(self, float p, int skill_rr, int team_rr):

    cdef dict new_states = {}
    cdef float p_success
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

"""
cpdef _add_fail_state(self, dict new_states, dict prev_state, float prev_state_p, float p, int index)

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

    cdef get_path(self, target):

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