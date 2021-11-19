from timeit import timeit
from tests.util import get_game_turn
import botbowl.core.pathfinding.python_pathfinding as python_pathfinding
import botbowl.core.pathfinding.cython_pathfinding as cython_pathfinding


def benchmark_cython_pathfinding():
    game = get_game_turn(empty=False)
    players_on_pitch = game.get_players_on_pitch()

    def get_paths_of_all_players(pf):
        for player in players_on_pitch:
            pf.Pathfinder(game, player, trr=True).get_paths()

    scope = {'get_paths_of_all_players': get_paths_of_all_players,
             'cython_pathfinding': cython_pathfinding,
             'python_pathfinding': python_pathfinding}

    cython_time = timeit("get_paths_of_all_players(cython_pathfinding)",number=2, globals=scope)
    python_time = timeit("get_paths_of_all_players(python_pathfinding)",number=2, globals=scope)

    print(f"cython_time = {cython_time}")
    print(f"python_time = {python_time}")
    print(f"{python_time/cython_time:.2f} times better")

if __name__ == "__main__":
    benchmark_cython_pathfinding()
