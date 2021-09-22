#include "pathing_node.h"
#include <iostream>

namespace node_ns {


    Square::Square(): x(0), y(0) {}
    Square::Square(int x, int y) : x(x), y(y) {}

    Node::Node() {}

    //root node constructor
    Node::Node(Square position, int moves_left, int gfis_left, double euclidean_distance, bool trr, bool dodge, bool sure_feet, bool sure_hands)
         : position(position), moves_left(moves_left), gfis_left(gfis_left),
         euclidean_distance(euclidean_distance)
    {
        // self.prob = parent.prob if parent is not None else 1
        // self.rr_states = rr_states if rr_states is not None else parent.rr_states

        prob = 1.0;
        rr_used key;
        key[TRR] = trr;
        key[DODGE] = dodge;
        key[SURE_FEET] = sure_feet;
        key[SURE_HANDS] = sure_hands;
        rr_states.insert( rr_map_pair(key, 1.0));

        //std::cout << "reroll states are!" << std::endl;
        //for (auto it = rr_states.begin(); it != rr_states.end(); ++it){
        //    rr_used state = it->first;
        //    double prev_p = it->second;
        //    std::cout << state[0] << " " << state[1] << " " << state[2] << " " << state[3] << std::endl;
        //}
    }

    // non-root node with block dice
    Node::Node(node_ptr parent, Square position, int moves_left, int gfis_left, double euclidean_distance, int block_dice)
        : parent(parent), position(position), moves_left(moves_left), gfis_left(gfis_left),
         euclidean_distance(euclidean_distance), block_dice(block_dice)
    {
        prob = parent->prob;
        rr_states = parent->rr_states;
    }

    // non-root node without block dice
    Node::Node(node_ptr parent, Square position, int moves_left, int gfis_left, double euclidean_distance)
        : parent(parent), position(position), moves_left(moves_left), gfis_left(gfis_left),
         euclidean_distance(euclidean_distance)
    {
        prob = parent->prob;
        rr_states = parent->rr_states;
    }

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
        fail_state[index] = false;
        double fail_state_p = prev_state_p * (1.0 - p) * p;

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
    }
    void Node::apply_foul(int target){
        foul_roll = target;
    }
    void Node::apply_stand_up(int target){
        rolls.push_back(target);
        _apply_roll((7.0-target)/6.0, NO_SKILL_REROLL, TRR);
    }

    bool operator<(const Node & n1, const Node & n2) {
        return n1.euclidean_distance < n2.euclidean_distance;
    }

    bool operator<(const node_ptr& n1, const node_ptr& n2){
        return n1->euclidean_distance < n2->euclidean_distance;
    }

}