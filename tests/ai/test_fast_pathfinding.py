from tests.util import *
import pytest

import ffai.ai.fast_pathing as pf


PROP_PRECISION = 0.000000001


def test_all_paths():
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 6
    position = Square(1, 1)
    game.put(player, position)
    pather = pf.Pathfinder(game, player)

    print("hello!")