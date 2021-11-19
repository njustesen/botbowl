#include "pathing_node.h"
#include <iostream>

namespace node_ns {

    Square::Square(): x(0), y(0) {}
    Square::Square(int x, int y) : x(x), y(y) {}
    int Square::distance(Square other){
        return std::max(abs(other.x - x), abs(other.y - y));
    }

    bool operator==(const Square& n1, const Square& n2){
        return n1.x == n2.x && n1.y == n2.y;
    }
    Square operator+(const Square& n1, const Square& n2){
        return Square(n1.x+n2.x, n1.y+n2.y);
    }
    Square operator-(const Square& n1, const Square& n2){
        return Square(n1.x-n2.x, n1.y-n2.y);
    }

    Node::Node() {}

    //root node constructor
    Node::Node(Square position, int moves_left, int gfis_left, double euclidean_distance,
                bool trr, bool dodge, bool sure_feet, bool sure_hands, bool can_foul, bool can_block, bool can_handoff)
         : position(position), moves_left(moves_left), gfis_left(gfis_left), foul_roll(0),
         handoff_roll(0), block_dice(0), can_foul(can_foul), can_block(can_block), can_handoff(can_handoff),
         euclidean_distance(euclidean_distance), prob(1.0)
    {
        rr_used key;
        key[TRR] = trr;
        key[DODGE] = dodge;
        key[SURE_FEET] = sure_feet;
        key[SURE_HANDS] = sure_hands;
        rr_states.insert( rr_map_pair(key, 1.0));
    }

    // non-root node with block dice
    Node::Node(node_ptr parent, Square position, int moves_left, int gfis_left,
                double euclidean_distance, int block_dice)
        : parent(parent), position(position), moves_left(moves_left), gfis_left(gfis_left),
        foul_roll(0), handoff_roll(0), block_dice(block_dice), can_foul(parent->can_foul), can_block(parent->can_block), can_handoff(parent->can_handoff),
        euclidean_distance(euclidean_distance),
        prob(parent->prob), rr_states(parent->rr_states)
    {}

    // non-root node without block dice
    Node::Node(node_ptr parent, Square position, int moves_left, int gfis_left, double euclidean_distance)
        : parent(parent), position(position), moves_left(moves_left), gfis_left(gfis_left), foul_roll(0), handoff_roll(0), block_dice(0), can_foul(parent->can_foul), can_block(parent->can_block), can_handoff(parent->can_handoff),
         euclidean_distance(euclidean_distance), prob(parent->prob), rr_states(parent->rr_states)
    {}

    Node::~Node() { }

    void Node::_apply_roll(double p, int skill_rr, int team_rr){
        rr_state new_states;

        for (auto it = rr_states.begin(); it != rr_states.end(); ++it){
            rr_used state = it->first;
            double prev_p = it->second;

            new_states[state] += prev_p * p; // if state is not in new_states, it will be init as 0.

            if (skill_rr != NO_SKILL_REROLL && state[skill_rr]) {
                _add_fail_state( new_states, state, prev_p, p, skill_rr);
            } else if (state[team_rr]) {
                _add_fail_state( new_states, state, prev_p, p, team_rr);
            }
        }
        rr_states = new_states;
        prob = 0;
        for (auto it = rr_states.begin(); it != rr_states.end(); ++it){
            prob += it->second;
        }
    }

    void Node::_add_fail_state(rr_state & new_states, rr_used & prev_state, double prev_state_p, double p, int index){
        auto fail_state = prev_state;
        double fail_state_p = prev_state_p * (1.0 - p) * p;
        fail_state[index] = false;
        new_states[fail_state] += fail_state_p; // if fail_state not in new states, it is init as 0
    }

    void Node::apply_gfi(){
        rolls.push_back(2);
        _apply_roll(5.0/6.0, SURE_FEET, TRR);
    }
    void Node::apply_dodge(int target){
        rolls.push_back(target);
        _apply_roll((7.0-target)/6.0, DODGE, TRR);
    }
    void Node::apply_pickup(int target){
        rolls.push_back(target);
        _apply_roll((7.0-target)/6.0, SURE_HANDS, TRR);
    }
    void Node::apply_handoff(int target){
        handoff_roll = target;
        can_handoff = false;
    }
    void Node::apply_foul(int target){
        foul_roll = target;
        can_foul = false;
    }
    void Node::apply_stand_up(int target){
        rolls.push_back(target);
        _apply_roll((7.0-target)/6.0, NO_SKILL_REROLL, TRR);
    }

    bool operator<(const Node & n1, const Node & n2) {
        return n1.euclidean_distance > n2.euclidean_distance; // inverse to give priority to shorter paths
    }

    bool operator<(const node_ptr& n1, const node_ptr& n2){
        return n1->euclidean_distance > n2->euclidean_distance; // inverse to give priority to shorter paths
    }

} //namespace