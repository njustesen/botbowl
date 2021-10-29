from tests.util import *

from timeit import default_timer as timer
import random

from copy import deepcopy
from botbowl.ai.registry import make_bot


import cProfile
import io
import pstats
#from pstats import SortKey


config = load_config("gym-11")
ruleset = load_rule_set(config.ruleset)
home = load_team_by_filename("human", ruleset)
away = load_team_by_filename("human", ruleset)
away_agent = make_bot("random")
home_agent = make_bot("random")


def run_game_with_forward_model():
    game = Game(1, home, away, home_agent, away_agent, config)
    game.enable_forward_model()
    game.init()


def run_game_without_forward_model():
    game = Game(1, home, away, home_agent, away_agent, config)
    game.init()


def normal_games():
    nbr_of_games = 10

    games_disabled_model = [Game(1, home, away, home_agent, away_agent, config, seed=i) for i in range(nbr_of_games)]
    games_enabled_model = [Game(1, home, away, home_agent, away_agent, config, seed=i) for i in range(nbr_of_games)]

    for game in games_enabled_model:
        game.enable_forward_model()

    start_time = timer()
    for game in games_disabled_model:
        game.init()
    end_time = timer()

    elapsed_disabled = end_time - start_time

    start_time = timer()
    for game in games_enabled_model:
        game.init()
    end_time = timer()

    elapsed_enabled = end_time - start_time

    print(f"{elapsed_disabled} vs. {elapsed_enabled}. increase = {round((elapsed_enabled/elapsed_disabled-1)*100)}% ")


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
    profile_forward_model_random_games()
