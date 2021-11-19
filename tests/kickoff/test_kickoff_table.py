import pytest
from botbowl.core.game import *
from tests.util import *


def test_get_the_ref():
    game = get_game_kickoff()
    D6.fix(1)  # Scatter
    D6.fix(1)
    D6.fix(1)
    home_bribes = game.state.home_team.state.bribes
    away_bribes = game.state.away_team.state.bribes
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(OutcomeType.KICKOFF_GET_THE_REF)
    assert game.state.home_team.state.bribes == home_bribes + 1
    assert game.state.away_team.state.bribes == away_bribes + 1


def test_riot():
    for i in range(7):
        rolls = [1]
        if 0 < i < 7:
            rolls = [1, 2, 3, 4, 5, 6]
        for roll in rolls:
            game = get_game_kickoff()
            game.state.home_team.state.turn = i
            game.state.away_team.state.turn = i
            D6.fix(1)  # Scatter
            D6.fix(1)
            D6.fix(2)
            if 0 < i < 7:
                D6.fix(roll)  # Riot roll
            game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
            if i == 7:
                assert game.state.home_team.state.turn == i - 1
                assert game.state.away_team.state.turn == i - 1
            elif i == 0:
                assert game.state.home_team.state.turn == i + 1
                assert game.state.away_team.state.turn == i + 1
            else:
                if roll < 4:
                    assert game.state.home_team.state.turn == i + 1
                    assert game.state.away_team.state.turn == i + 1
                else:
                    assert game.state.home_team.state.turn == i - 1
                    assert game.state.away_team.state.turn == i - 1


def test_perfect_defence():
    game = get_game_kickoff()
    D6.fix(1)  # Scatter
    D6.fix(2)
    D6.fix(2)
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    proc = game.get_procedure()
    assert game.has_report_of_type(OutcomeType.KICKOFF_PERFECT_DEFENSE)
    assert type(proc) == Setup
    team = game.get_agent_team(game.actor)
    assert team == game.get_kicking_team()
    for action_choice in game.state.available_actions:
        assert action_choice.action_type in [ActionType.END_SETUP, ActionType.PLACE_PLAYER, ActionType.SETUP_FORMATION_SPREAD, ActionType.SETUP_FORMATION_ZONE]
        for player in action_choice.players:
            assert player in game.get_players_on_pitch(team)
            square = Square(player.position.x, player.position.y)
            assert game.get_player_at(square) == player

            if square.x <= 13:
                square = Square(square.x-1, square.y)
            else:
                square = Square(square.x, square.y+1)

            game.step(Action(ActionType.PLACE_PLAYER, player=player, position=square))
            assert game.get_player_at(square) == player


def get_empty_square_without_adjacent_players(game, x=None, y=None):
    for action_choice in game.state.available_actions:
        for square in action_choice.positions:
            if game.get_player_at(square) is None:
                for adjacent in game.get_adjacent_squares(square):
                    if game.get_player_at(adjacent) is not None:
                        break
                else:
                    if (x is None or square.x == x) and (y is None or square.y == y):
                        return square
    return None


def test_high_kick():
    game = get_game_kickoff()
    D6.fix(1)  # Scatter
    D8.fix(6)  # Scatter
    D6.fix(3)
    D6.fix(2)
    ball_placed_at = get_empty_square_without_adjacent_players(game, y=6)
    assert ball_placed_at is not None
    game.step(Action(ActionType.PLACE_BALL, position=ball_placed_at))
    proc = game.get_procedure()
    assert game.has_report_of_type(OutcomeType.KICKOFF_HIGH_KICK)
    assert type(proc) == HighKick
    team = game.get_receiving_team()
    assert game.actor == game.get_team_agent(team)
    catcher = None
    for action_choice in game.state.available_actions:
        print(action_choice.action_type)
        assert action_choice.action_type in [ActionType.SELECT_PLAYER, ActionType.SELECT_NONE]
        if action_choice.action_type == ActionType.SELECT_NONE:
            continue
        for player in action_choice.players:
            assert player in game.get_players_on_pitch(team)
            assert game.num_tackle_zones_in(player) == 0
            catcher = player
        for player in game.get_players_on_pitch(team):
            if game.num_tackle_zones_in(player) == 0:
                assert player in action_choice.players
            else:
                assert player not in action_choice.players
    if catcher is not None:
        D6.fix(6)  # Catch
        game.step(Action(ActionType.SELECT_PLAYER, player=catcher))
        assert game.has_ball(catcher)


def test_high_kick_touchback():
    game = get_game_kickoff()
    D6.fix(6)  # Scatter
    D8.fix(2)  # Scatter
    D6.fix(3)
    D6.fix(2)
    ball_placed_at = get_empty_square_without_adjacent_players(game, y=1)
    assert ball_placed_at is not None
    game.step(Action(ActionType.PLACE_BALL, position=ball_placed_at))
    proc = game.get_procedure()
    assert game.has_report_of_type(OutcomeType.KICKOFF_HIGH_KICK)
    assert game.has_report_of_type(OutcomeType.TOUCHBACK)
    assert type(proc) == Touchback
    team = game.get_receiving_team()
    assert game.actor == game.get_team_agent(team)


def test_blitz_movement():
    game = get_game_kickoff()
    D6.fix(1)  # Scatter
    D6.fix(5)  # Blitz
    D6.fix(5)  # Blitz
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(OutcomeType.KICKOFF_BLITZ)
    actor = game.actor
    team = game.get_agent_team(actor)
    assert team == game.get_kicking_team()
    assert game.is_blitz()
    for player in game.get_players_on_pitch(team):
        if game.num_tackle_zones_in(player) >= 1:
            for action_choice in game.state.available_actions:
                assert player.position not in action_choice.positions
        else:
            game.step(Action(ActionType.START_MOVE, player=player))
            x = player.position.x
            game.step(Action(ActionType.MOVE, position=Square(player.position.x - 1, player.position.y)))
            assert player.position.x == x - 1
            game.step(Action(ActionType.END_PLAYER_TURN, player=player))
    game.step(Action(ActionType.END_TURN))


def test_blitz_touchdown():
    game = get_game_kickoff()
    D6.fix(1)  # Scatter
    D6.fix(5)  # Blitz
    D6.fix(5)  # Blitz
    team = game.get_agent_team(game.actor)
    endzone_x = game.get_opp_endzone_x(team)
    target = Square(endzone_x, 5)
    game.step(Action(ActionType.PLACE_BALL, position=target))
    assert game.has_report_of_type(OutcomeType.KICKOFF_BLITZ)
    team = game.get_agent_team(game.actor)
    assert team == game.get_kicking_team()
    assert game.is_blitz()
    player = team.players[0]
    assert player.position is not None
    game.move(player, target)
    game.move(game.get_ball(), target)
    D6.fix(6)  # Catch
    game.step(Action(ActionType.END_TURN))
    assert game.has_report_of_type(OutcomeType.TOUCHDOWN)


def test_quick_snap():
    game = get_game_kickoff()
    D6.fix(1)  # Scatter
    D6.fix(4)
    D6.fix(5)  # Quick snap
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(OutcomeType.KICKOFF_QUICK_SNAP)
    actor = game.actor
    team = game.get_agent_team(actor)
    assert team == game.get_receiving_team()
    assert game.is_quick_snap()
    for player in game.get_players_on_pitch(team):
        for action_choice in game.state.available_actions:
            if action_choice.action_type != ActionType.END_TURN:
                adjacent_squares = game.get_adjacent_squares(player.position, occupied=False)
                if len(adjacent_squares) == 0:
                    assert player not in action_choice.players
                else:
                    assert player in action_choice.players
                    game.step(Action(ActionType.START_MOVE, player=player))
                    for action_choice in game.state.available_actions:
                        if action_choice.action_type == ActionType.MOVE:
                            assert len(action_choice.positions) == len(adjacent_squares)
                            game.step(Action(ActionType.MOVE, player=player, position=action_choice.positions[0]))
    game.step(Action(ActionType.END_TURN))


@pytest.mark.parametrize("kickoff_event", [OutcomeType.KICKOFF_CHEERING_FANS, OutcomeType.KICKOFF_BRILLIANT_COACHING])
def test_cheering_fans_brilliant_coaching_equal_fame_and_roll(kickoff_event):
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    game.state.home_team.cheerleaders = 0
    game.state.away_team.cheerleaders = 0
    game.state.home_team.ass_coaches = 0
    game.state.away_team.ass_coaches = 0
    home_rr = game.state.home_team.state.rerolls
    away_rr = game.state.away_team.state.rerolls
    D6.fix(1)  # Scatter
    if kickoff_event == OutcomeType.KICKOFF_CHEERING_FANS:
        D6.fix(3)
        D6.fix(3)  # Cheering fans
    else:
        D6.fix(4)
        D6.fix(4)  # Brilliant coaching
    D3.fix(1)  # home team roll
    D3.fix(1)  # away team roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(kickoff_event)
    assert game.state.home_team.state.rerolls == home_rr + 1
    assert game.state.away_team.state.rerolls == away_rr + 1


@pytest.mark.parametrize("kickoff_event", [OutcomeType.KICKOFF_CHEERING_FANS, OutcomeType.KICKOFF_BRILLIANT_COACHING])
def test_cheering_fans_brilliant_coaching_equal_roll(kickoff_event):
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 1
    game.state.home_team.cheerleaders = 0
    game.state.away_team.cheerleaders = 0
    game.state.home_team.ass_coaches = 0
    game.state.away_team.ass_coaches = 0
    home_rr = game.state.home_team.state.rerolls
    away_rr = game.state.away_team.state.rerolls
    D6.fix(1)  # Scatter
    if kickoff_event == OutcomeType.KICKOFF_CHEERING_FANS:
        D6.fix(3)
        D6.fix(3)  # Cheering fans
    else:
        D6.fix(4)
        D6.fix(4)  # Brilliant coaching
    D3.fix(2)  # home team roll
    D3.fix(1)  # away team roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(kickoff_event)
    assert game.state.home_team.state.rerolls == home_rr + 1
    assert game.state.away_team.state.rerolls == away_rr + 1


@pytest.mark.parametrize("kickoff_event", [OutcomeType.KICKOFF_CHEERING_FANS, OutcomeType.KICKOFF_BRILLIANT_COACHING])
def test_cheering_fans_brilliant_coaching_unequal_roll(kickoff_event):
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 1
    game.state.home_team.cheerleaders = 0
    game.state.away_team.cheerleaders = 0
    game.state.home_team.ass_coaches = 0
    game.state.away_team.ass_coaches = 0
    home_rr = game.state.home_team.state.rerolls
    away_rr = game.state.away_team.state.rerolls
    D6.fix(1)  # Scatter
    if kickoff_event == OutcomeType.KICKOFF_CHEERING_FANS:
        D6.fix(3)
        D6.fix(3)  # Cheering fans
    else:
        D6.fix(4)
        D6.fix(4)  # Brilliant coaching
    D3.fix(3)  # home team roll
    D3.fix(1)  # away team roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(kickoff_event)
    assert game.state.home_team.state.rerolls == home_rr + 1
    assert game.state.away_team.state.rerolls == away_rr


@pytest.mark.parametrize("kickoff_event", [OutcomeType.KICKOFF_CHEERING_FANS, OutcomeType.KICKOFF_BRILLIANT_COACHING])
def test_cheering_fans_brilliant_coaching_cheerleaders_ass_coaches_equal(kickoff_event):
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 1
    game.state.home_team.cheerleaders = 1
    game.state.away_team.cheerleaders = 0
    game.state.home_team.ass_coaches = 1
    game.state.away_team.ass_coaches = 0
    home_rr = game.state.home_team.state.rerolls
    away_rr = game.state.away_team.state.rerolls
    D6.fix(1)  # Scatter
    if kickoff_event == OutcomeType.KICKOFF_CHEERING_FANS:
        D6.fix(3)
        D6.fix(3)  # Cheering fans
    else:
        D6.fix(4)
        D6.fix(4)  # Brilliant coaching
    D3.fix(1)  # home team roll
    D3.fix(1)  # away team roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(kickoff_event)
    assert game.state.home_team.state.rerolls == home_rr + 1
    assert game.state.away_team.state.rerolls == away_rr + 1


@pytest.mark.parametrize("kickoff_event", [OutcomeType.KICKOFF_CHEERING_FANS, OutcomeType.KICKOFF_BRILLIANT_COACHING])
def test_cheering_fans_brilliant_coaching_ten_fame(kickoff_event):
    game = get_game_kickoff()
    game.state.home_team.state.fame = 10
    game.state.away_team.state.fame = 0
    game.state.home_team.cheerleaders = 0
    game.state.away_team.cheerleaders = 0
    game.state.home_team.ass_coaches = 0
    game.state.away_team.ass_coaches = 0
    home_rr = game.state.home_team.state.rerolls
    away_rr = game.state.away_team.state.rerolls
    D6.fix(1)  # Scatter
    if kickoff_event == OutcomeType.KICKOFF_CHEERING_FANS:
        D6.fix(3)
        D6.fix(3)  # Cheering fans
    else:
        D6.fix(4)
        D6.fix(4)  # Brilliant coaching
    D3.fix(1)  # home team roll
    D3.fix(3)  # away team roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert game.has_report_of_type(kickoff_event)
    assert game.state.home_team.state.rerolls == home_rr + 1
    assert game.state.away_team.state.rerolls == away_rr


weather_dice_rolls = [[1, 1], [1, 2], [2, 2], [5, 5], [5, 6], [6, 6]]
@pytest.mark.parametrize("dice_roll", weather_dice_rolls)
def test_changing_weather(dice_roll):
    game = get_game_kickoff()
    game.state.weather = WeatherType.NICE
    D6.fix(1)  # Scatter
    D6.fix(3)
    D6.fix(4)  # Changing weather
    D6.fix(dice_roll[0])  # Weather roll
    D6.fix(dice_roll[1])  # Weather roll

    if game.state.available_actions[0].positions[0].x <= 13:
        pos = Square(6, 7)
    else:
        pos = Square(19, 7)

    game.step(Action(ActionType.PLACE_BALL, position=pos))
    assert game.has_report_of_type(OutcomeType.KICKOFF_CHANGING_WHEATHER)
    if np.sum(dice_roll) == 2:
        assert game.state.weather == WeatherType.SWELTERING_HEAT
    elif np.sum(dice_roll) == 3:
        assert game.state.weather == WeatherType.VERY_SUNNY
    elif 4 <= np.sum(dice_roll) <= 10:
        assert game.state.weather == WeatherType.NICE
        assert game.has_report_of_type(OutcomeType.GENTLE_GUST_IN_BOUNDS)
    elif np.sum(dice_roll) == 11:
        assert game.state.weather == WeatherType.POURING_RAIN
    elif np.sum(dice_roll) == 12:
        assert game.state.weather == WeatherType.BLIZZARD


def test_throw_a_rock_same_stunned():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(5)  # Throw a rock
    D6.fix(6)  # Throw a rock
    D3.fix(1)  # home team roll
    D3.fix(1)  # away team roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 1
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 1


def test_throw_a_rock_same_knocked_out():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(5)  # Throw a rock
    D6.fix(6)  # Throw a rock
    D3.fix(1)  # home team roll
    D3.fix(1)  # away team roll
    # Random player not determined by dice roll - should it?
    D6.fix(4)  # Injury roll
    D6.fix(4)  # Injury roll
    D6.fix(4)  # Injury roll
    D6.fix(4)  # Injury roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 10
    assert len(game.get_players_on_pitch(game.state.away_team)) == 10
    assert len(game.get_knocked_out(game.state.home_team)) == 1
    assert len(game.get_knocked_out(game.state.away_team)) == 1
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 0
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 0


def test_throw_a_rock_same_cas():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(5)  # Throw a rock
    D6.fix(6)  # Throw a rock
    D3.fix(1)  # home team roll
    D3.fix(1)  # away team roll
    # Random player not determined by dice roll - should it?
    D6.fix(5)  # Injury roll
    D6.fix(5)  # Injury roll
    D6.fix(1)  # Badly hurt
    D8.fix(1)  # Badly hurt
    D6.fix(5)  # Injury roll
    D6.fix(5)  # Injury roll
    D6.fix(1)  # Badly hurt
    D8.fix(1)  # Badly hurt
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 10
    assert len(game.get_players_on_pitch(game.state.away_team)) == 10
    assert len(game.get_casualties(game.state.home_team)) == 1
    assert len(game.get_casualties(game.state.away_team)) == 1
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 0
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 0


def test_throw_a_rock_home_stunned():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(5)  # Throw a rock
    D6.fix(6)  # Throw a rock
    D3.fix(2)  # home team roll
    D3.fix(1)  # away team roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 0
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 1


def test_throw_a_rock_away_stunned():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(5)  # Throw a rock
    D6.fix(6)  # Throw a rock
    D3.fix(1)  # home team roll
    D3.fix(2)  # away team roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 1
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 0


def test_throw_a_rock_home_fame_stunned():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 1
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(5)  # Throw a rock
    D6.fix(6)  # Throw a rock
    D3.fix(1)  # home team roll
    D3.fix(1)  # away team roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 0
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 1


def test_throw_a_rock_away_fame_stunned():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 1
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(5)  # Throw a rock
    D6.fix(6)  # Throw a rock
    D3.fix(1)  # home team roll
    D3.fix(1)  # away team roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    D6.fix(1)  # Injury roll
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 1
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 0


def test_pitch_invasion_no_fame():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(6)  # Pitch invasion
    D6.fix(6)  # Pitch invasion
    for team in game.state.teams:
        i = 1
        for player in game.get_players_on_pitch(team):
            D6.fix(i)
            if i < 6:
                i += 1
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 6
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 6
    for team in game.state.teams:
        i = 1
        for player in sorted(game.get_players_on_pitch(team), key=lambda p: p.nr):
            if i < 6:
                assert player.state.up
                i += 1
            else:
                assert not player.state.up


def test_pitch_invasion_home_fame():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 1
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(6)  # Pitch invasion
    D6.fix(6)  # Pitch invasion
    for team in game.state.teams:
        i = 1
        for player in game.get_players_on_pitch(team):
            D6.fix(i)
            if i < 6:
                i += 1
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 6
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 7
    for team in game.state.teams:
        i = 1
        for player in sorted(game.get_players_on_pitch(team), key=lambda p: p.nr):
            if i + game.get_opp_team(team).state.fame < 6:
                assert player.state.up
                i += 1
            else:
                assert not player.state.up


def test_pitch_invasion_away_fame():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 1
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    D6.fix(1)  # Scatter
    D6.fix(6)  # Pitch invasion
    D6.fix(6)  # Pitch invasion
    for team in game.state.teams:
        i = 1
        for player in game.get_players_on_pitch(team):
            D6.fix(i)
            if i < 6:
                i += 1
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 7
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 6
    for team in game.state.teams:
        i = 1
        for player in sorted(game.get_players_on_pitch(team), key=lambda p: p.nr):
            if i + game.get_opp_team(team).state.fame < 6:
                assert player.state.up
                i += 1
            else:
                assert not player.state.up


def test_pitch_invasion_ball_and_chain():
    game = get_game_kickoff()
    game.state.home_team.state.fame = 0
    game.state.away_team.state.fame = 0
    assert len(game.get_players_on_pitch(game.state.home_team)) == 11
    assert len(game.get_players_on_pitch(game.state.away_team)) == 11
    game.get_players_on_pitch(game.state.home_team)[0].extra_skills.append(Skill.BALL_AND_CHAIN)
    game.get_players_on_pitch(game.state.away_team)[0].extra_skills.append(Skill.BALL_AND_CHAIN)
    D6.fix(1)  # Scatter
    D6.fix(6)  # Pitch invasion
    D6.fix(6)  # Pitch invasion
    for team in game.state.teams:
        for player in game.get_players_on_pitch(team):
            D6.fix(6)
    game.step(Action(ActionType.PLACE_BALL, position=game.state.available_actions[0].positions[0]))
    assert len(game.get_players_on_pitch(game.state.home_team)) == 10
    assert len(game.get_players_on_pitch(game.state.away_team)) == 10
    assert len(game.get_knocked_out(game.state.home_team)) == 1
    assert len(game.get_knocked_out(game.state.away_team)) == 1
    assert len([player for player in game.get_players_on_pitch(game.state.home_team) if not player.state.up]) == 10
    assert len([player for player in game.get_players_on_pitch(game.state.away_team) if not player.state.up]) == 10
    for team in game.state.teams:
        for player in sorted(game.get_players_on_pitch(team), key=lambda p: p.nr):
            if player.has_skill(Skill.BALL_AND_CHAIN):
                assert player.state.knocked_out
            else:
                assert not player.state.up
