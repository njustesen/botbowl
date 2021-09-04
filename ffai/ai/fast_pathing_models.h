#ifndef FAST_PATHING_MODELS_H_ 
#define FAST_PATHING_MODELS_H_

#include <vector>

namespace fast_path_np {
    class Square {
        public: 
            int x, y 
            Square(int, int)
    }
    
    class Node {
        public: 
            Node * parent;
            Square position; 
            int moves_left, gfis_left, foul_roll, handoff_roll, block_dice;
            float euclidean_distance, prob;
            std::vector<int> rolls;
            Rr_state_map rr_states;

            Node::Node(Node * parent, Square position, int moves_left, int gfis_left, float euclidean_distance, int reroll_states=0, block_dice=0, foul_roll=0, handoff_roll=0 )
            void _apply_roll(Node* self, float p, int skill_rr, int team_rr);
            void _add_fail_state(Rr_state_map * new_states, Reroll_State prev_state, float prev_state_p, float p, int index);
            void apply_gfi();
            void apply_dodge(int target);
            void apply_pickup(int target);
            void apply_handoff(int target);
            void apply_foul(int target);
            void apply_stand_up(int target);
    }

}
#endif 