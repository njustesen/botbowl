import pytest
from tests.util import *


def place_player(game, team, x, y):
    xx = x if team == game.state.away_team else game.arena.width - x - 1
    yy = y
    sq = Square(xx, yy)
    game.step(Action(ActionType.PLACE_PLAYER, player=game.get_dugout(team).reserves[0], position=sq))
    return game.get_player_at(sq)


def clear_board(game):
    for team in [game.state.home_team, game.state.away_team]:
        for player in game.get_players_on_pitch(team):
            game.pitch_to_reserves(player)


@pytest.mark.parametrize("home_team", [True, False])
def test_scrimmage_setup(home_team):
    game = get_game_setup(home_team)
    team = game.state.home_team if home_team else game.state.away_team
    proc = game.get_procedure()
    assert type(proc) == Setup

    # Empty board
    assert not game.is_setup_legal(team)
    assert not game.is_setup_legal_count(team)
    assert not game.is_setup_legal_scrimmage(team)
    assert game.is_setup_legal_wings(team)

    # 3 Players on board - not on scrimmage
    place_player(game, team, 8, 6)
    place_player(game, team, 8, 7)
    place_player(game, team, 8, 8)
    assert game.is_setup_legal_count(team)
    assert not game.is_setup_legal(team)
    assert not game.is_setup_legal_scrimmage(team)
    assert game.is_setup_legal_wings(team)
    clear_board(game)

    # 1-7 players on scrimmage
    place_player(game, team, 13, 5)
    assert not game.is_setup_legal(team)
    assert not game.is_setup_legal_scrimmage(team)
    place_player(game, team, 13, 6)
    assert not game.is_setup_legal(team)
    assert not game.is_setup_legal_scrimmage(team)
    place_player(game, team, 13, 7)
    assert game.is_setup_legal(team)
    assert game.is_setup_legal_scrimmage(team)
    place_player(game, team, 13, 8)
    assert game.is_setup_legal(team)
    assert game.is_setup_legal_scrimmage(team)
    place_player(game, team, 13, 9)
    assert game.is_setup_legal(team)
    assert game.is_setup_legal_scrimmage(team)
    place_player(game, team, 13, 10)
    assert game.is_setup_legal(team)
    assert game.is_setup_legal_scrimmage(team)
    clear_board(game)


@pytest.mark.parametrize("home_team", [True, False])
def test_wings_setup(home_team):
    game = get_game_setup(home_team)
    team = game.state.home_team if home_team else game.state.away_team
    proc = game.get_procedure()
    assert type(proc) == Setup

    # Top Wing
    place_player(game, team, 10, 1)
    assert not game.is_setup_legal(team)
    assert game.is_setup_legal_wings(team)
    place_player(game, team, 10, 2)
    assert not game.is_setup_legal(team)
    assert game.is_setup_legal_wings(team)
    place_player(game, team, 10, 3)
    assert not game.is_setup_legal(team)
    assert not game.is_setup_legal_wings(team)
    clear_board(game)

    # Bottom Wing
    place_player(game, team, 10, 12)
    assert not game.is_setup_legal(team)
    assert game.is_setup_legal_wings(team)
    place_player(game, team, 10, 13)
    assert not game.is_setup_legal(team)
    assert game.is_setup_legal_wings(team)
    place_player(game, team, 10, 14)
    assert not game.is_setup_legal(team)
    assert not game.is_setup_legal_wings(team)
    clear_board(game)


@pytest.mark.parametrize("home_team", [True, False])
def test_player_count_setup(home_team):
    game = get_game_setup(home_team)
    team = game.state.home_team if home_team else game.state.away_team
    proc = game.get_procedure()
    assert type(proc) == Setup
    assert not game.is_setup_legal_count(team)
    # Max players
    place_player(game, team, 10, 1)
    assert not game.is_setup_legal_count(team)
    place_player(game, team, 10, 2)
    assert not game.is_setup_legal_count(team)
    place_player(game, team, 10, 5)
    place_player(game, team, 10, 6)
    place_player(game, team, 13, 7)
    place_player(game, team, 13, 8)
    place_player(game, team, 13, 9)
    place_player(game, team, 10, 10)
    place_player(game, team, 10, 11)
    place_player(game, team, 10, 12)
    place_player(game, team, 10, 13)
    assert game.is_setup_legal_scrimmage(team)
    assert game.is_setup_legal_wings(team)
    assert game.is_setup_legal_count(team)
    assert game.is_setup_legal(team)
    player = place_player(game, team, 9, 8)
    assert not game.is_setup_legal(team)
    game.pitch_to_reserves(player)
    assert game.is_setup_legal(team)
    assert game.get_agent_team(game.actor) == team
    game.step(Action(ActionType.END_SETUP))
    assert game.get_agent_team(game.actor) != team
