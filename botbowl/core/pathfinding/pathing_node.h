#ifndef PATHING_NODE_H_
#define PATHING_NODE_H_

#include <vector>
#include <memory>
#include <array>
#include <map>

namespace node_ns {

    class Square {
        public:
            int x, y;
            Square();
            Square(int, int);
            int distance(Square);
    };
    bool operator==(const Square& n1, const Square& n2);
    Square operator+(const Square& n1, const Square& n2);
    Square operator-(const Square& n1, const Square& n2);


    const int NO_SKILL_REROLL = -1;
    const int TRR = 0;
    const int DODGE = 1;
    const int SURE_FEET = 2;
    const int SURE_HANDS = 3;

    class Node;

    using rr_used = std::array<bool, 4>;
    using rr_map_pair = std::pair<rr_used, double>;
    using rr_state = std::map<rr_used, double>;
    using node_ptr = std::shared_ptr<Node>;

    class Node {
        public:
            const node_ptr parent;
            Square position;
            int moves_left, gfis_left, foul_roll, handoff_roll, block_dice;
            bool can_foul, can_block, can_handoff;
            double euclidean_distance, prob;
            std::vector<int> rolls;
            rr_state rr_states;

            Node();

            //root node constructor
            Node(Square position, int moves_left, int gfis_left, double euclidean_distance, bool trr, bool dodge, bool sure_feet, bool sure_hands, bool can_foul, bool can_block, bool can_handoff);

            // non-root node with block dice
            Node(node_ptr parent, Square position, int moves_left, int gfis_left, double euclidean_distance, int block_dice);

            // non-root node without block dice
            Node(node_ptr parent, Square position, int moves_left, int gfis_left, double euclidean_distance);

            ~Node();

            void _apply_roll(double p, int skill_rr, int team_rr);
            void _add_fail_state(rr_state & new_states, rr_used & prev_state, double prev_state_p, double p, int index);
            void apply_gfi();
            void apply_dodge(int target);
            void apply_pickup(int target);
            void apply_handoff(int target);
            void apply_foul(int target);
            void apply_stand_up(int target);
    };

    //const node_ptr & dominant_node(const node_ptr &, const node_ptr &, const Square &);
    //const node_ptr & best_node(const node_ptr &, const node_ptr &, const Square &);

    bool operator<(const Node& n1, const Node& n2);
    bool operator<(const node_ptr& n1, const node_ptr& n2);


}

#endif