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
    Node::Node(Square position, int moves_left, int gfis_left, double euclidean_distance, bool trr, bool dodge, bool sure_feet, bool sure_hands)
         : position(position), moves_left(moves_left), gfis_left(gfis_left),
         euclidean_distance(euclidean_distance),
         foul_roll(0), handoff_roll(0), block_dice(0)
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
         euclidean_distance(euclidean_distance), block_dice(block_dice),
         foul_roll(0), handoff_roll(0)
    {
        prob = parent->prob;
        rr_states = parent->rr_states;
    }

    // non-root node without block dice
    Node::Node(node_ptr parent, Square position, int moves_left, int gfis_left, double euclidean_distance)
        : parent(parent), position(position), moves_left(moves_left), gfis_left(gfis_left),
         euclidean_distance(euclidean_distance),
         foul_roll(0), handoff_roll(0), block_dice(0)
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

    /*
    const node_ptr & dominant_node(const node_ptr & a, const node_ptr & b, const Square & start_pos){
        if (a->position.distance(start_pos) == 1 && a->moves_left > b->moves_left)
            return a;
        if (b->position.distance(self.player.position) == 1 and b.moves_left > a.moves_left:
            return b
        a_moves_left = a.moves_left + a.gfis_left
        b_moves_left = b.moves_left + b.gfis_left
        # TODO: Write out as above
        if a.prob > b.prob and (a.foul_roll is None or a.foul_roll <= b.foul_roll) and (a.block_dice is None or a.block_dice >= b.block_dice) and (a_moves_left > b_moves_left or (a_moves_left == b_moves_left and a.euclidean_distance < b.euclidean_distance)):
            return a
        if b.prob > a.prob and (b.foul_roll is None or b.foul_roll <= a.foul_roll) and (b.block_dice is None or b.block_dice >= a.block_dice) and (b_moves_left > a_moves_left or (b_moves_left == a_moves_left and b.euclidean_distance < a.euclidean_distance)):
            return b
        return None
    }

    const node_ptr & best_node(const node_ptr & a, const node_ptr & b, const Square & start_pos){
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
    }
    */

}