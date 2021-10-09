from tests.util import *

import ffai.ai.fast_pathing as fast_pf
import ffai.ai.pathfinding as slow_pf
import os


def test_that_cython_and_python_test_files_are_equal():
    """
    This unconventional test makes sure that the same tests are being
    made for cython and python pathfinding. It's not perfect and open for proposals!
    This destroy mutation testing for pathfinding.
    """

    python_test_file = "test_pathfinding.py"
    cython_test_file = "test_pathfinding_cython.py"

    if os.getcwd().split("/")[-1] == "ffai":
        python_test_file = "tests/ai/" + python_test_file
        cython_test_file = "tests/ai/" + cython_test_file

    assert os.path.isfile(python_test_file) and os.path.isfile(cython_test_file)

    i = 1
    with open(cython_test_file, "r") as cython_file, open(python_test_file, "r") as python_file:
        for cython_line, python_line in zip(cython_file, python_file):
            if not cython_line[0:6] == python_line[0:6] == "import":
                assert cython_line == python_line, \
                        f'On line {i}: "{cython_line.strip()}" != "{python_line.strip()}"'
            i += 1


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

    # Don't compare every individual step, only destination, probability and path length.
    def create_comparable(path):
        return f"({path.steps[-1].x}, {path.steps[-1].y}) p={path.prob:.5f} len={len(path.steps)}"

    # Use this function for easier debugging
    #def create_path_string(path):
    #    return "->".join([f"({step.x}, {step.y})" for step in path.steps]) + f"rolls={path.rolls}"

    for slow_path, fast_path in zip(slow_paths, fast_paths):
        slow_s = create_comparable(slow_path)
        fast_s = create_comparable(fast_path)
        assert slow_s == fast_s

        #if slow_s != fast_s:
        #    print(f"slow = {slow_s} \n {create_path_string(slow_path)}")
        #    print(f"fast = {fast_s} \n {create_path_string(fast_path)}")