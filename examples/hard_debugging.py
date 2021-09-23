from tests.util import get_game_turn, Square

import ffai.ai.fast_pathing as pf
import ffai.ai.pathfinding as slow_pf

game = get_game_turn(empty=True)
player = game.get_reserves(game.state.away_team)[0]
player.role.ma = 7
position = Square(1, 1)
game.put(player, position)

opp_player = game.get_reserves(game.state.home_team)[0]
game.put(opp_player, Square(3,4))


#pather = slow_pf.Pathfinder(game, player)
pather = pf.Pathfinder(game, player)
paths = pather.get_paths()

slow_pather = slow_pf.Pathfinder(game, player)
slow_paths = slow_pather.get_paths()

def create_string(pathss):
    s = []
    for path in pathss:
        t = f"({path.steps[-1].x} {path.steps[-1].y}) p={path.prob:.6f} len={len(path.steps)}"
        s.append(t)
    return s

slow_result = create_string(slow_paths)
fast_result = create_string(paths)

for slow, fast in zip(slow_result, fast_result):
    if slow == fast:
        print(f"ok    - {slow}")
    else:
        print(f"WRONG - {slow} != {fast}")

#assert create_string(slow_paths) == create_string(paths)
#print(create_string(slow_paths))


