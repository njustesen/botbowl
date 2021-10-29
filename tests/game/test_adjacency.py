import pytest
from botbowl.core.game import *
from tests.util import *


def test_get_adjacent_opponents():
    game = get_game_turn(empty=True)
    team1 = game.state.home_team
    team2 = game.state.away_team
    player = game.get_reserves(team1)[0]
    game.state.player_by_id[player.player_id] = player
    game.reserves_to_pitch(player, Square(6, 6))
    i = 1
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            if x == 0 and y == 0:
                continue
            opponent = game.get_reserves(team2)[0]
            game.state.player_by_id[opponent.player_id] = opponent
            position = Square(player.position.x + x, player.position.y + y)
            game.reserves_to_pitch(opponent, position)
            assert len(game.get_adjacent_opponents(player)) == i
            assert game.num_tackle_zones_in(player) == i
            assert game.get_adjacent_opponents(opponent) == [player]
            assert game.num_tackle_zones_in(opponent) == 1
            assert len(game.get_adjacent_players(player.position, team=opponent.team)) == i
            assert game.get_adjacent_players(opponent.position, team=player.team) == [player]
            assert len(game.get_assisting_players(player, opponent)) == 0
            assert len(game.get_assisting_players(opponent, player)) == i-1
            i += 1

