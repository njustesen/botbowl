#include "fast_pathing_models.h"

namespace fast_path_np {
    Square::Square(int x, int y){
        this->x = x
        this->y = y
    }

    Node::Node(Node * parent, Square position, int moves_left, int gfis_left, float euclidean_distance, int reroll_states=0, block_dice=0, foul_roll=0, handoff_roll=0 )
        : parent(parent), position(position), moves_left(moves_left), gfis_left(gfis_left),
    {
    }

}