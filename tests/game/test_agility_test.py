from tests.util import *
import pytest


@pytest.mark.parametrize("ag", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_agility_test(ag):
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    player = team.players[1]
    player.role.ag = ag
    player.role.skills = []
    game.put(player, Square(10, 1))
    agility_roll = Rules.agility_table[player.get_ag()]
    if ag == 1:
        assert agility_roll == 6
    elif ag == 2:
        assert agility_roll == 5
    elif ag == 3:
        assert agility_roll == 4
    elif ag == 4:
        assert agility_roll == 3
    elif ag == 5:
        assert agility_roll == 2
    elif ag == 6:
        assert agility_roll == 1
