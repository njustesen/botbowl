import pytest
from ffai.core.game import *


def get_game(home_team, seed=0):
    config = get_config("ff-11")
    config.kick_off_table = False
    ruleset = get_rule_set(config.ruleset)
    home = get_team_by_filename("human", ruleset)
    away = get_team_by_filename("human", ruleset)
    home_agent = Agent("human1", human=True)
    away_agent = Agent("human2", human=True)
    game = Game(1, home, away, home_agent, away_agent, config)
    game.set_seed(seed)
    game.init()
    game.step(Action(ActionType.START_GAME))
    game.step(Action(ActionType.HEADS))
    if game.actor == game.home_agent:
        if home_team:
            game.step(Action(ActionType.KICK))
        else:
            game.step(Action(ActionType.RECEIVE))
    if game.actor == game.away_agent:
        if home_team:
            game.step(Action(ActionType.RECEIVE))
        else:
            game.step(Action(ActionType.KICK))
    return game


def place_player(game, team, x, y):
    xx = x if team == game.state.away_team else game.arena.width - x - 1
    yy = y
    sq = Square(xx, yy)
    game.step(Action(ActionType.PLACE_PLAYER, player=game.get_dugout(team).reserves[0], pos=sq))
    return game.get_player_at(sq)


def clear_board(game):
    for team in [game.state.home_team, game.state.away_team]:
        for player in game.get_players_on_pitch(team):
            game.pitch_to_reserves(player)


@pytest.mark.parametrize("home_team", [True, False])
def test_scrimmage_setup(home_team):
    game = get_game(home_team)
    team = game.state.home_team if home_team else game.state.away_team
    proc = game.state.stack.peek()
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
    game = get_game(home_team)
    team = game.state.home_team if home_team else game.state.away_team
    proc = game.state.stack.peek()
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
    game = get_game(home_team)
    team = game.state.home_team if home_team else game.state.away_team
    proc = game.state.stack.peek()
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
    assert game.agent_team(game.actor) == team
    game.step(Action(ActionType.END_SETUP))
    assert game.agent_team(game.actor) != team


#if __name__ == "__main__":
#    test_scrimmage_setup(True)
#    test_scrimmage_setup(False)
#    test_wings_setup(True)
#    test_wings_setup(False)
#    test_player_count_setup(True)
#    test_player_count_setup(False)
