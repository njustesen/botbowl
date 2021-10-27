from tests.util import *
import pytest


# Follows https://www.thenaf.net/wp-content/uploads/2018/02/TTM-flowchart-v3.jpg


def test_pickup_teammate_no_right_stuff():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = []
    right_stuff.role.skills = []
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))

    with pytest.raises(InvalidActionError):
        game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))


def test_failed_pickup_teammate():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(1)  # Cause fumble
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=Square(5, 5)))
    assert game.has_report_of_type(OutcomeType.FUMBLE)
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
    assert right_stuff.position == right_stuff_position
    assert right_stuff.state.up
    assert not right_stuff.state.in_air
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)


def test_successfull_land():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert passer.state.used
    assert not right_stuff.state.used
    assert right_stuff.position == Square(target_square.x + 3, target_square.y)
    assert right_stuff.state.up


def test_successfull_land_on_ball():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    D6.fix(6)  # potential pickup ball
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    ball = game.get_ball()
    game.move(ball, Square(8, 5))
    ball.is_carried = False  # touchback might give the ball to some player and makes it carried
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert passer.state.used
    assert not right_stuff.state.used
    assert right_stuff.position == Square(target_square.x + 3, target_square.y)
    assert right_stuff.state.up
    assert ball.position != right_stuff.position
    assert not ball.is_carried


def test_failed_landing():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(1)  # Land
    D6.fix(1)  # Armor roll
    D6.fix(1)  # Armor roll
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.FAILED_LAND)
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
    assert right_stuff.position == Square(target_square.x + 3, target_square.y)
    assert not right_stuff.state.up


def test_failed_landing_ball():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    ball = game.get_ball()
    game.put(ball, right_stuff_position)
    ball.is_carried = True
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(1)  # Land
    D6.fix(1)  # Armor roll
    D6.fix(1)  # Armor roll
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert game.has_report_of_type(OutcomeType.TURNOVER)
    assert not ball.is_carried


def test_successful_landing_with_ball():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    ball = game.get_ball()
    game.put(ball, right_stuff_position)
    ball.is_carried = True
    assert game.has_ball(right_stuff)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
    landing_square = Square(target_square.x + 3, target_square.y)
    assert right_stuff.position == landing_square
    assert right_stuff.state.up
    assert ball.is_carried
    assert ball.position == landing_square


def test_successful_landing_endzone():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    x_endzone = game.get_opp_endzone_x(team)
    y = 6
    endzone = Square(x_endzone, y)
    if x_endzone == 1:
        passer_square = Square(x_endzone + 5, y)
        right_stuff_square = Square(x_endzone + 6, y)
        target_square = Square(endzone.x + 1, endzone.y)
    else:
        passer_square = Square(x_endzone - 5, y)
        right_stuff_square = Square(x_endzone - 6, y)
        target_square = Square(endzone.x - 1, endzone.y)
    game.put(passer, passer_square)
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    game.put(right_stuff, right_stuff_square)
    ball = game.get_ball()
    ball.position = right_stuff.position
    ball.is_carried = True
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(4)  # Backward scatter
    D8.fix(5)  # Forward scatter
    if x_endzone == 1:
        D8.fix(4)  # Backward scatter
    else:
        D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert game.has_report_of_type(OutcomeType.TOUCHDOWN)


def test_successful_landing_crowd():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    passer_square = Square(5, 5)
    right_stuff_square = Square(4, 4)
    target_square = Square(5, 1)
    game.put(passer, passer_square)
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    game.put(right_stuff, right_stuff_square)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(4)  # Backward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(2)  # Up scatter
    D6.fix(1)  # injury
    D6.fix(1)  # injury
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert game.has_report_of_type(OutcomeType.PLAYER_OUT_OF_BOUNDS)
    assert game.has_report_of_type(OutcomeType.STUNNED)
    assert right_stuff.position is None
    assert right_stuff in game.get_dugout(team).reserves
    assert passer.state.used


def test_successful_landing_on_opp_players():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    passer_square = Square(5, 5)
    right_stuff_square = Square(4, 4)
    target_square = Square(8, 8)
    game.put(passer, passer_square)
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    game.put(right_stuff, right_stuff_square)
    opp_team = game.get_opp_team(team)
    opp_player_a = opp_team.players[0]
    opp_player_b = opp_team.players[1]
    game.put(opp_player_a, Square(9, 8))
    game.put(opp_player_b, Square(10, 8))
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(4)  # Backward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Armor
    D6.fix(6)  # Armor
    D6.fix(4)  # injury
    D6.fix(5)  # injury
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Armor (in case)
    D6.fix(6)  # Armor (in case)
    D6.fix(6)  # Injury (in case)
    D6.fix(6)  # Injury (in case)
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert game.has_report_of_type(OutcomeType.PLAYER_HIT_PLAYER)
    assert game.has_report_of_type(OutcomeType.KNOCKED_OUT)
    assert opp_player_a.position is None
    assert opp_player_a in game.get_dugout(opp_team).reserves
    assert opp_player_b.position is not None
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert right_stuff.state.up
    assert passer.state.used


def test_successful_landing_on_own_player():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    passer_square = Square(5, 5)
    right_stuff_square = Square(4, 4)
    target_square = Square(8, 8)
    game.put(passer, passer_square)
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    game.put(right_stuff, right_stuff_square)
    player = team.players[3]
    game.put(player, Square(9, 8))
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(4)  # Backward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Armor
    D6.fix(6)  # Armor
    D6.fix(4)  # injury
    D6.fix(5)  # injury
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert game.has_report_of_type(OutcomeType.PLAYER_HIT_PLAYER)
    assert game.has_report_of_type(OutcomeType.KNOCKED_OUT)
    assert game.has_report_of_type(OutcomeType.TURNOVER)
    assert player.position is None
    assert player in game.get_dugout(team).reserves
    assert right_stuff.position is not None
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert right_stuff.state.up


def test_ttm_distances_and_modifiers():
    game = get_game_turn()
    game.clear_board()
    team = game.get_agent_team(game.actor)
    passer = team.players[0]
    passer.role.skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(2,2,))
    quick_mod = game.get_pass_modifiers(passer, PassDistance.QUICK_PASS, ttm=True)
    short_mod = game.get_pass_modifiers(passer, PassDistance.SHORT_PASS, ttm=True)
    assert quick_mod == 0
    assert short_mod == -1


def test_ttm_distance():
    game = get_game_turn()
    game.clear_board()
    team = game.get_agent_team(game.actor)
    passer = team.players[0]
    passer.role.skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(2, 2))
    right_stuff = team.players[1]
    right_stuff.role.skills = [Skill.RIGHT_STUFF]
    game.put(right_stuff, Square(3, 2))
    squares, distances = game.get_pass_distances(passer, right_stuff)
    for distance in distances:
        assert distance == PassDistance.QUICK_PASS or distance == PassDistance.SHORT_PASS


def test_throw_teammate_while_having_ball():
    # passer has the ball and has throw team mate
    # after selecting START_PASS he should be either throw the ball or pick up
    # a teammate, if he picks up the teammate he should only be allowed to
    # throw the teammate.
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    ball = game.get_ball()
    ball.position = passer.position
    ball.is_carried = True
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D8.fix(5)  # Forward scatter
    D6.fix(6)  # Land
    game.step(Action(ActionType.PICKUP_TEAM_MATE, player=passer, position=right_stuff.position))
    assert right_stuff.state.in_air
    assert not (ActionType.PASS in [aa.action_type for aa in game.state.available_actions])
    # throw teammate
    target_square = Square(5, 5)
    game.step(Action(ActionType.THROW_TEAM_MATE, player=passer, position=target_square))
    assert game.has_report_of_type(OutcomeType.INACCURATE_PASS)  # Always inaccurate
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_LAND)
    assert not game.has_report_of_type(OutcomeType.TURNOVER)
    assert passer.state.used
    assert not right_stuff.state.used
    assert Square(target_square.x + 3, target_square.y)
    assert right_stuff.state.up
    assert ball.position == passer.position
    assert ball.is_carried == True


def test_throw_ball_no_pickup():
    # passer has the ball and has throw team mate
    # after selecting START_PASS he should be either throw the ball or pick up
    # a teammate, if he picks up the teammate he should only be allowed to
    # throw the teammate.
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    game.clear_board()
    passer = team.players[0]
    passer.role.skills = []
    passer.role.ag = 2
    passer.extra_skills = [Skill.THROW_TEAM_MATE]
    game.put(passer, Square(1, 1))
    ball = game.get_ball()
    ball.position = passer.position
    ball.is_carried = True
    right_stuff = team.players[1]
    right_stuff.role.skills = []
    right_stuff.extra_skills = [Skill.RIGHT_STUFF]
    right_stuff_position = Square(2, 1)
    game.put(right_stuff, right_stuff_position)
    catcher = team.players[3]
    catcher.role.skills = []
    catcher.role.ag = 3
    game.put(catcher, Square(3, 3))
    game.step(Action(ActionType.START_PASS, player=passer))
    D6.fix(6)  # Accurate pass
    D6.fix(6)  # Successful catch
    game.step(Action(ActionType.PASS, player=passer, position=catcher.position))
    assert not (ActionType.PICKUP_TEAM_MATE in [aa.action_type for aa in game.state.available_actions])
    assert ball.position == catcher.position
    assert ball.is_carried == True
