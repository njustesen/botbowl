from tests.util import *

import ffai.ai.fast_pathing as fast_pf
import ffai.ai.pathfinding as slow_pf


def test_all_paths():
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]

    player.extra_skills.append(Skill.DODGE)
    player.extra_skills.append(Skill.SURE_FEET)
    team_rr = True

    player.role.ma = 7
    position = Square(7, 7)
    game.put(player, position)

    opp_player = game.get_reserves(game.state.home_team)[0]
    game.put(opp_player, Square(8, 8))

    opp_player = game.get_reserves(game.state.home_team)[1]
    game.put(opp_player, Square(5, 7))

    opp_player = game.get_reserves(game.state.home_team)[2]
    game.put(opp_player, Square(8, 5))

    fast_paths = fast_pf.Pathfinder(game, player, trr=team_rr).get_paths()
    slow_paths = slow_pf.Pathfinder(game, player, directly_to_adjacent=True, trr=team_rr).get_paths()

    def create_comparable(path):
        return f"({path.steps[-1].x}, {path.steps[-1].y}) p={path.prob:.5f} len={len(path.steps)}"

    # Use this function for better debugging
    #def create_path_string(path):
    #    return "->".join([f"({step.x}, {step.y})" for step in path.steps]) + f"rolls={path.rolls}"


    for slow_path, fast_path in zip(slow_paths, fast_paths):
        slow_s = create_comparable(slow_path)
        fast_s = create_comparable(fast_path)
        assert slow_s == fast_s

        #if slow_s != fast_s:
        #    print(f"slow = {slow_s} \n {create_path_string(slow_path)}")
        #    print(f"fast = {fast_s} \n {create_path_string(fast_path)}")