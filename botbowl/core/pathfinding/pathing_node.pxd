cdef extern from "pathing_node.cpp":
    pass

from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

# Declare the class with cdef
cdef extern from "pathing_node.h" namespace "node_ns":

    cdef cppclass Square:
        Square() except +
        Square(int, int) except +
        int x,y
        int distance(Square)
    bint operator ==(Square, Square)
    Square operator +(Square, Square)
    Square operator -(Square, Square)
    #using rr_used = std::tuple < bool, bool, bool, bool >;
    #using rr_state = std::map < rr_used, double >;

    cdef cppclass Node:
        int moves_left, gfis_left, foul_roll, handoff_roll, block_dice
        bint can_foul, can_block, can_handoff;
        double euclidean_distance, prob
        Square position
        shared_ptr[Node] parent
        vector[int] rolls

        Node() except +
        Node(Square, int, int, double, bint, bint, bint, bint, bint, bint, bint) except +  #root node constructor
        Node(shared_ptr[Node], Square, int, int, double, int) except +   # non-root node with block dice
        Node(shared_ptr[Node], Square, int, int, double) except +        # non-root node without block dice

        void apply_gfi()
        void apply_dodge(int)
        void apply_pickup(int)
        void apply_handoff(int)
        void apply_foul(int)
        void apply_stand_up(int)

    bint operator <(Node, Node)
    bint operator <(shared_ptr[Node], shared_ptr[Node])

