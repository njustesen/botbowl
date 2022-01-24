import pytest
from botbowl.core.game import *
from tests.util import *


def test_no_dodge():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    player = [player for player in players if game.num_tackle_zones_in(player) == 0][0]
    # make sure there won't be a dodge roll needed
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) == 0

    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 1, player.position.y)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert player.position == to
    assert player.state.up
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)


def test_dodge_fail_one():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    # make sure we don't get stuck waiting for reroll actions
    game.state.teams[0].state.rerolls = 0

    opponents = game.get_players_on_pitch(game.get_opp_team(current_team))
    game.put(player, Square(11, 11))

    opp_player = opponents[1]
    game.put(opp_player, Square(12, 12))
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(11, 12)
    assert game.get_player_at(to) is None
    D6.fix(1)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert player.position == to
    assert not player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)


def test_dodge_skill_reroll_single_use_limit():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    player.extra_skills = [Skill.DODGE]
    # make sure we don't get stuck waiting for re-roll actions
    game.state.teams[0].state.rerolls = 0

    opponents = game.get_players_on_pitch(game.get_opp_team(current_team))
    game.put(player, Square(11, 11))

    opp_player = opponents[1]
    game.put(opp_player, Square(12, 12))
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    from_square = player.position
    to = Square(11, 12)
    assert game.get_player_at(to) is None
    D6.fix(1)  # fail first dodge
    D6.fix(4)  # pass on dodge skill
    D6.fix(1)  # fail second dodge
    D6.fix(6)  # second dodge skill use will pass - if the code is wrong!
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert player.position == to
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)

    # check for single use - attempt second dodge out of opponents tackle zone
    to2 = Square(10, 12)
    game.step(Action(ActionType.MOVE, player=player, position=to2))
    assert player.position == to2
    assert not player.state.up


def test_dodge_skill_reroll_failed():
    game = get_game_turn()
    current_team = game.get_agent_team(game.actor)

    players = game.get_players_on_pitch(team=current_team)
    player = players[1]
    player.extra_skills = [Skill.DODGE]

    opponents = game.get_players_on_pitch(game.get_opp_team(current_team))
    game.put(player, Square(11, 11))

    opp_player = opponents[1]
    game.put(opp_player, Square(12, 12))
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    from_square = player.position
    to = Square(11, 12)
    assert game.get_player_at(to) is None
    D6.fix(1)  # fail dodge roll
    D6.fix(1)  # fail dodge re-roll
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert player.position == to
    assert not player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)


def test_dodge_no_modifier():
    game = get_game_turn(empty=True)
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, game.state.home_team)
    game.put(player, Square(5, 5))
    opp_player = Player("2", role, "test", 1, game.state.away_team)
    game.put(opp_player, Square(6, 6))
    # ensure there are opposing tackle zones on the player (1)
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) == 1
    target = Square(4, 4)
    # ensure there are no opposing tackle zones on the target
    assert len(game.get_adjacent_players(target, game.get_opp_team(player.team))) == 0
    modifier = game.get_dodge_modifiers(player, target)
    assert modifier == 1


def test_dodge_one_modifier():
    game = get_game_turn(empty=True)
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, game.state.home_team)
    game.put(player, Square(11, 11))
    opp_player = Player("2", role, "test", 1, game.state.away_team)
    game.put(opp_player, Square(12, 12))
    target = Square(12, 11)
    # make sure we have a dodge scenario
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) > 0
    # ensure there is an opposing tackle zone on the target (1)
    assert len(game.get_adjacent_players(target, game.get_opp_team(player.team))) == 1
    modifier = game.get_dodge_modifiers(player, target)
    assert modifier == 0


def test_dodge_two_modifier():
    game = get_game_turn(empty=True)
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, game.state.home_team)
    game.put(player, Square(11, 11))
    opp_player = Player("2", role, "test", 1, game.state.away_team)
    game.put(opp_player, Square(12, 12))
    opp_player = Player("3", role, "test", 2, game.state.away_team)
    game.put(opp_player, Square(11, 12))
    pos = Square(12, 11)
    # make sure we have a dodge scenario
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) > 0
    # ensure there are opposing tackle zones on the target (2)
    assert len(game.get_adjacent_players(pos, game.get_opp_team(player.team))) == 2

    modifier = game.get_dodge_modifiers(player, pos)
    assert modifier == -1


def test_prehensile_tail_modifier():
    game = get_game_turn()
    role = Role("Lineman", "orc", 6, 3, 3, 9, [], 50000, False)
    player = Player("1", role, "test", 1, game.state.home_team)
    tail_player = Player("1", role, "test", 1, game.state.away_team, extra_skills=[Skill.PREHENSILE_TAIL])
    game.put(player, Square(5, 5))
    game.put(tail_player, Square(6, 6))
    target = Square(4, 4)
    # make sure we have a dodge scenario
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) > 0

    # ensure there are no tackle zones on the target
    assert len(game.get_adjacent_players(target, game.get_opp_team(player.team))) == 0

    modifier = game.get_dodge_modifiers(player, target)
    assert modifier == 0


def test_dodge_use_break_tackle():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    player = players[1]
    player.role.skills = []
    player.extra_skills = [Skill.BREAK_TACKLE]
    player.role.st = 4
    player.role.ag = 3
    game.put(player, Square(11, 11))

    opponents = game.get_players_on_pitch(game.get_opp_team(team))
    opp_player = opponents[0]
    game.put(opp_player, Square(12, 11))

    # make sure there is one enemy in tackle zone
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) == 1
    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 1, player.position.y + 1)
    D6.fix(3)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)
    game.step(Action(ActionType.USE_SKILL))
    assert player.position == to
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)
    assert game.has_report_of_type(OutcomeType.SKILL_USED)


def test_break_tackle_reroll():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    player = players[1]
    player.role.skills = []
    player.extra_skills = [Skill.BREAK_TACKLE]
    player.role.st = 3
    player.role.ag = 2
    game.put(player, Square(11, 11))

    opponents = game.get_players_on_pitch(game.get_opp_team(team))
    opp_player = opponents[0]
    game.put(opp_player, Square(12, 11))

    # make sure there is one enemy in tackle zone
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) == 1

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 1, player.position.y + 1)
    D6.fix(3)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)
    D6.fix(4)
    game.step(Action(ActionType.USE_REROLL))
    assert not game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)
    game.step(Action(ActionType.USE_SKILL))
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)
    assert player.position == to
    assert player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)


def test_break_tackle_twice():
    game = get_game_turn()
    team = game.get_agent_team(game.actor)
    # get a player
    players = game.get_players_on_pitch(team, False, True)
    player = players[1]
    player.role.skills = []
    player.extra_skills = [Skill.BREAK_TACKLE]
    player.role.st = 3
    player.role.ag = 2
    game.put(player, Square(11, 11))

    opponents = game.get_players_on_pitch(game.get_opp_team(team))
    opp_player = opponents[0]
    game.put(opp_player, Square(12, 11))

    # make sure there is one enemy in tackle zone
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) == 1

    game.set_available_actions()
    game.step(Action(ActionType.START_MOVE, player=player))
    to = Square(player.position.x + 1, player.position.y + 1)
    D6.fix(4)
    game.step(Action(ActionType.MOVE, player=player, position=to))
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)
    game.step(Action(ActionType.USE_SKILL))
    assert game.has_report_of_type(OutcomeType.SUCCESSFUL_DODGE)
    assert game.has_report_of_type(OutcomeType.SKILL_USED)
    # make sure there is one enemy in tackle zone
    assert len(game.get_adjacent_players(player.position, game.get_opp_team(player.team))) == 1
    D6.fix(4)
    to_second = Square(player.position.x - 1, player.position.y - 1)
    game.step(Action(ActionType.MOVE, player=player, position=to_second))
    game.step(Action(ActionType.DONT_USE_REROLL))
    assert player.position == to_second
    assert not player.state.up
    assert game.has_report_of_type(OutcomeType.FAILED_DODGE)
