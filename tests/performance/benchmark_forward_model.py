from tests.util import *
from time import perf_counter as timer
import random

from copy import deepcopy
from botbowl.ai.registry import make_bot

import cProfile
import io
import pstats
#from pstats import SortKey


config = load_config("gym-11")
config.pathfinding_enabled = False
ruleset = load_rule_set(config.ruleset)
home = load_team_by_filename("human", ruleset)
away = load_team_by_filename("human", ruleset)


def run_game_with_forward_model():
    game = Game(1, home, away, RandomBot('random', seed=1), RandomBot('random', seed=1), config)
    game.enable_forward_model()
    game.init()


def run_game_without_forward_model():
    game = Game(1, home, away, RandomBot('random', seed=1), RandomBot('random', seed=1), config)
    game.init()


def normal_games():
    nbr_of_games = 10

    games_disabled_model = [Game(i, home, away, RandomBot('random', i), RandomBot('random', i), config, seed=i)
                            for i in range(nbr_of_games)]
    games_enabled_model = [Game(i, home, away, RandomBot('random', i), RandomBot('random', i), config, seed=i)
                           for i in range(nbr_of_games)]

    for game in games_enabled_model:
        game.enable_forward_model()

    start_time = timer()
    for game in games_disabled_model:
        game.init()
    end_time = timer()

    elapsed_time_fm_disabled = end_time - start_time

    start_time = timer()
    for game in games_enabled_model:
        game.init()
    end_time = timer()

    elapsed_time_fm_enabled = end_time - start_time

    print(f"{nbr_of_games} without forward model: {elapsed_time_fm_disabled:.2f} seconds")
    print(f"{nbr_of_games} with forward model: {elapsed_time_fm_enabled:.2f} seconds")
    print(f"Forward model is {elapsed_time_fm_enabled/elapsed_time_fm_disabled:.2f}x slower")


def profile_forward_model_random_games():
    pr = cProfile.Profile()

    pr.enable()
    for i in range(10):
        run_game_with_forward_model()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(50)
    print(s.getvalue())


if __name__ == "__main__":
    normal_games()
    #profile_forward_model_random_games()
