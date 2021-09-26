from ffai import Skill
from tests.util import get_game_turn, Square

import time

import ffai.ai.fast_pathing as pf
import ffai.ai.pathfinding as slow_pf

game = get_game_turn(empty=True)
player = game.get_reserves(game.state.away_team)[0]

player.extra_skills.append(Skill.DODGE)
#player.extra_skills.append(Skill.SURE_FEET)
team_rr = True

player.role.ma = 7
position = Square(7, 7)
game.put(player, position)


opp_player = game.get_reserves(game.state.home_team)[0]
game.put(opp_player, Square(8,8))

opp_player = game.get_reserves(game.state.home_team)[1]
game.put(opp_player, Square(5,7))

opp_player = game.get_reserves(game.state.home_team)[2]
game.put(opp_player, Square(8,5))

N = 10

start_time = time.time()
for _ in range(N):
    paths = pf.Pathfinder(game, player, trr=team_rr).get_paths()
fast_ex_time = time.time() - start_time

start_time = time.time()
for _ in range(N):
    slow_paths = slow_pf.Pathfinder(game, player, directly_to_adjacent=True, trr=team_rr).get_paths()
slow_ex_time = time.time() - start_time

print(fast_ex_time)
print(slow_ex_time)


def create_string(path):
    t = f"({path.steps[-1].x}, {path.steps[-1].y}) p={path.prob:.5f} len={len(path.steps)}"
    return t

def create_path_string(path):
    return "->".join([f"({step.x}, {step.y})" for step in path.steps]) + f"rolls={path.rolls}"

fine = True
for slow_path, fast_path in zip(slow_paths, paths):

    slow_s = create_string(slow_path)
    fast_s = create_string(fast_path)

    if slow_s != fast_s:
        print(f"slow = {slow_s} \n {create_path_string(slow_path)}")
        print(f"fast = {fast_s} \n {create_path_string(fast_path)}")
        fine = False
        break
    #else:
    #    print(f"OK - {fast_s}")
if fine:
    l = len(slow_paths)
    print(f"{l}/{l} paths OK!")
    print(f"{create_string(paths[-1])}")


#assert create_string(slow_paths) == create_string(paths)
#print(create_string(slow_paths))


