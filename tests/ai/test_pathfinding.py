from tests.util import Square, get_game_turn, Skill, Action, ActionType, get_custom_game_turn
import pytest
import unittest.mock
import numpy as np
import botbowl.core.procedure

import botbowl.core.pathfinding.python_pathfinding as python_pathfinding
pathfinding_modules_to_test = [python_pathfinding]

# We only test the cython pathfindng module if it exists. Otherwise half of the tests fail.
# This way only test_compare_cython_python_paths() fails because it explicitly uses cython pf module.
try:
    import botbowl.core.pathfinding.cython_pathfinding as cython_pathfinding
    pathfinding_modules_to_test.append(cython_pathfinding)
except ImportError:
    cython_pathfinding = None  # this will make the cython vs. python compare test fail.

PROP_PRECISION = 0.000000001


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_neighbors(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    position = Square(5, 5)
    game.put(player, position)
    for neighbor in game.get_adjacent_squares(position):
        path = pf.get_safest_path(game, player, neighbor)
        assert len(path) == 1 and path.steps[0] == neighbor
        assert path.prob == 1.0


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_out_of_bounds(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    position = Square(1, 1)
    game.put(player, position)
    for neighbor in game.get_adjacent_squares(position):
        if game.is_out_of_bounds(neighbor):
            path = pf.get_safest_path(game, player, neighbor)
            assert path is None


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_gfi(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    position = Square(5, 5)
    game.put(player, position)
    for gfis in [0, 1, 2]:
        moves = player.get_ma() + gfis
        target = Square(position.x + moves, position.y)
        path = pf.get_safest_path(game, player, target)
        assert len(path.steps) == moves and path.get_last_step()  == target
        assert path.prob == (5 / 6) ** gfis


skills_and_rerolls_perms = [
    (True, True),
    (True, False),
    (False, True),
    (False, False)
]


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
@pytest.mark.parametrize("skills_and_rerolls", skills_and_rerolls_perms)
def test_get_safest_path(skills_and_rerolls, pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    position = Square(1, 1)
    game.put(player, position)
    skill, reroll = skills_and_rerolls
    if skill:
        player.extra_skills = [Skill.SURE_FEET]
    for y in range(game.arena.height):
        for x in range(game.arena.width):
            square = Square(x, y)
            if position != square and not game.is_out_of_bounds(square):
                if position.distance(square) != player.get_ma() + 2:
                    continue
                    # TODO: REMOVE
                path = pf.get_safest_path(game, player, square, allow_team_reroll=reroll)
                if position.distance(square) > player.get_ma() + 2:
                    assert path is None
                else:
                    p = 5 / 6
                    if position.distance(square) == player.get_ma() + 2:
                        p_clean = p * p
                        p_reroll_one = (p * (1 - p) * p)
                        p_reroll_both = (1 - p) * p * (1 - p) * p
                        if reroll and not skill or not reroll and skill:
                            p = p_clean + p_reroll_one * 2
                        elif reroll and skill:
                            p = p_clean + p_reroll_one * 2 + p_reroll_both
                        else:
                            p = p_clean
                        assert path is not None
                        assert path.prob == pytest.approx(p, PROP_PRECISION)
                    elif position.distance(square) == player.get_ma() + 1:
                        assert path is not None
                        if reroll or skill:
                            p = p + (1 - p) * p
                        assert path.prob == pytest.approx(p, PROP_PRECISION)
                    else:
                        assert path is not None
                        p = 1.0
                        assert path.prob == p


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_invalid_path(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    position = Square(1, 1)
    game.put(player, position)
    target_a = Square(12, 12)
    path = pf.get_safest_path(game, player, target_a)
    assert path is None


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_avoid_path(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    position = Square(1, 8)
    game.put(player, position)
    opp = game.get_reserves(game.state.away_team)[0]
    opp_position = Square(3, 8)
    game.put(opp, opp_position)
    target_a = Square(6, 8)
    path = pf.get_safest_path(game, player, target_a)
    assert path is not None
    assert len(path.steps) == 6
    assert path.prob == 1.0


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
@pytest.mark.parametrize("sure_feet", [True, False])
def test_sure_feet_over_ag4_dodge(sure_feet, pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    player.role.ma = 4
    player.role.ag = 4
    if sure_feet:
        player.extra_skills = [Skill.SURE_FEET]
    game.put(player, Square(4, 1))
    opp1 = game.get_reserves(game.state.away_team)[0]
    game.put(opp1, Square(3, 3))
    opp2 = game.get_reserves(game.state.away_team)[1]
    game.put(opp2, Square(5, 3))
    target = Square(2, 4)
    path = pf.get_safest_path(game, player, target)
    assert path is not None
    if sure_feet:
        assert len(path.steps) == 5
        p = (5 / 6)
        assert path.prob == p + (1 - p) * p
    else:
        assert len(path.steps) == 4
        p = (5 / 6)
        assert path.prob == p


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_dodge_needed_path_long(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    position = Square(1, 8)
    game.put(player, position)
    opp = game.get_reserves(game.state.away_team)[0]
    opp_position = Square(3, 8)
    game.put(opp, opp_position)
    target_a = Square(9, 8)
    path = pf.get_safest_path(game, player, position=target_a)
    assert path is not None
    assert len(path.steps) == 8
    assert path.prob == (4 / 6) * (5 / 6) * (5 / 6)


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_path_to_endzone_home(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.home_team)[0]
    player.role.ma = 6
    position = Square(4, 4)
    game.put(player, position)
    path = pf.get_safest_path_to_endzone(game, player)
    assert path is not None
    assert len(path) == 3


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_path_to_endzone_away(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 6
    position = Square(23, 4)
    game.put(player, position)
    path = pf.get_safest_path_to_endzone(game, player)
    assert path is not None
    assert len(path) == 3


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_all_paths(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 6
    position = Square(1, 1)
    game.put(player, position)
    paths = pf.get_all_paths(game, player)
    assert paths is not None
    assert len(paths) == ((player.num_moves_left() + 1) * (player.num_moves_left() + 1)) - 1


def test_that_unittest_mock_patch_works():
    """
    This test makes sure that unittest.mock.patch works as expected in other tests.
    """
    with unittest.mock.patch('botbowl.core.procedure.Pathfinder', None):
        game = get_game_turn()
        game.config.pathfinding_enabled = True
        team = game.get_agent_team(game.actor)
        player = game.get_players_on_pitch(team=team)[0]
        with pytest.raises(TypeError):
            game.step(Action(ActionType.START_MOVE, player=player))


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_blitz_action_type_is_block(pf):
    with unittest.mock.patch('botbowl.core.procedure.Pathfinder', pf.Pathfinder):
        game = get_game_turn()
        game.config.pathfinding_enabled = True
        team = game.get_agent_team(game.actor)
        player = game.get_players_on_pitch(team=team)[0]
        player.role.ma = 16
        game.step(Action(ActionType.START_BLITZ, player=player))
        assert np.sum([len(action.positions) for action in game.get_available_actions() if action.action_type == ActionType.BLOCK]) == 11


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_handoff_action_type_is_handoff(pf):
    with unittest.mock.patch('botbowl.core.procedure.Pathfinder', pf.Pathfinder):
        game = get_game_turn()
        game.config.pathfinding_enabled = True
        team = game.get_agent_team(game.actor)
        player = game.get_players_on_pitch(team=team)[0]
        game.move(game.get_ball(), player.position)
        game.get_ball().is_carried = True
        player.role.ma = 16
        game.step(Action(ActionType.START_HANDOFF, player=player))
        assert np.sum([len(action.positions) for action in game.get_available_actions() if action.action_type == ActionType.HANDOFF]) == 10


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_handoff_action_type_is_foul(pf):
    with unittest.mock.patch('botbowl.core.procedure.Pathfinder', pf.Pathfinder):
        game = get_game_turn()
        game.config.pathfinding_enabled = True
        team = game.get_agent_team(game.actor)
        player = game.get_players_on_pitch(team=team)[0]
        game.move(game.get_ball(), player.position)
        game.get_ball().carried = True
        player.role.ma = 16
        for opp_player in game.get_players_on_pitch(team=game.get_opp_team(team)):
            opp_player.state.up = False
        game.step(Action(ActionType.START_FOUL, player=player))
        assert np.sum([len(action.positions) for action in game.get_available_actions() if action.action_type == ActionType.FOUL]) == 11


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_blitz_action_type_is_block_with_stab(pf):
    with unittest.mock.patch('botbowl.core.procedure.Pathfinder', pf.Pathfinder):
        game = get_game_turn()
        game.config.pathfinding_enabled = True
        team = game.get_agent_team(game.actor)
        player = game.get_players_on_pitch(team=team)[0]
        player.role.ma = 16
        player.role.skills = [Skill.STAB]
        game.step(Action(ActionType.START_BLITZ, player=player))
        assert len([action.action_type for action in game.get_available_actions() if action.action_type == ActionType.BLOCK]) == 1
        assert len([action.action_type for action in game.get_available_actions() if action.action_type == ActionType.STAB]) == 1


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_all_blitz_paths_one(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 6
    game.put(player, Square(1, 1))
    opp_player = game.get_reserves(game.state.home_team)[0]
    game.put(opp_player, Square(3, 3))
    paths = pf.get_all_paths(game, player, blitz=True)
    assert paths is not None
    blitz_paths = [path for path in paths if path.block_dice is not None]
    assert len(blitz_paths) == 1


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_all_blitz_paths_two(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 6
    game.put(player, Square(1, 1))
    opp_player = game.get_reserves(game.state.home_team)[0]
    game.put(opp_player, Square(3, 3))
    opp_player = game.get_reserves(game.state.home_team)[1]
    game.put(opp_player, Square(4, 3))
    paths = pf.get_all_paths(game, player, blitz=True)
    assert paths is not None
    blitz_paths = [path for path in paths if path.block_dice is not None]
    assert len(blitz_paths) == 2


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_handoff_after_gfi(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 6
    player.state.moves = 8
    game.put(player, Square(1, 1))
    other_player = game.get_reserves(game.state.away_team)[1]
    game.put(other_player, Square(2, 2))
    pathfinder = pf.Pathfinder(game,
                               player,
                               can_handoff=True)
    paths = pathfinder.get_paths()
    assert len(paths) == 1
    assert len(paths[0].steps) == 1
    assert paths[0].steps[0] == other_player.position


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_foul_after_gfi(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 6
    player.state.moves = 8
    game.put(player, Square(1, 1))
    opp_player = game.get_reserves(game.state.home_team)[0]
    opp_player.state.up = False
    game.put(opp_player, Square(2, 2))
    pathfinder = pf.Pathfinder(game,
                               player,
                               can_foul=True)
    paths = pathfinder.get_paths()
    assert len(paths) == 1
    assert len(paths[0].steps) == 1
    assert paths[0].steps[0] == opp_player.position


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_foul(pf):
    game, (player, opp_player_1, opp_player_2) = get_custom_game_turn(player_positions=[(1, 1)],
                                                                      opp_player_positions=[(1, 2), (1, 3)])

    player.role.ma = 1
    opp_player_1.state.up = False
    opp_player_2.state.up = False

    pathfinder = pf.Pathfinder(game,
                               player,
                               directly_to_adjacent=True,
                               can_foul=True,
                               trr=False)
    paths = pathfinder.get_paths()
    total_fouls = 0
    for path in paths:
        fouls = 0
        for step in path.steps:
            if step in [opp_player_1.position, opp_player_2.position]:
                fouls += 1
                assert step == path.get_last_step()
        assert fouls <= 1
        total_fouls += fouls
    assert total_fouls == 2


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_handoff(pf):
    game, (player, teammate_1, teammate_2) = get_custom_game_turn(player_positions=[(1, 1), (1, 2), (1, 3)],
                                                                  ball_position=(1, 1))

    player.role.ma = 1
    pathfinder = pf.Pathfinder(game,
                               player,
                               directly_to_adjacent=True,
                               can_handoff=True,
                               trr=False)

    paths = pathfinder.get_paths()
    total_handoffs = 0
    for path in paths:
        handoffs = 0
        for step in path.steps:
            if step in [teammate_1.position, teammate_2.position]:
                handoffs += 1
                assert step == path.get_last_step()
        assert handoffs <= 1
        total_handoffs += handoffs
    assert total_handoffs == 2


def test_compare_cython_python_paths():
    """
    This test compares paths calculated by cython and python. They should be same.
    It only compares destination, probability and number of steps, not each individual step.
    """
    assert hasattr(cython_pathfinding, 'Pathfinder'), 'Cython pathfinding was not imported'
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]

    player.extra_skills.append(Skill.DODGE)
    player.extra_skills.append(Skill.SURE_FEET)

    player.role.ma = 7
    position = Square(7, 7)
    game.put(player, position)

    for sq in [Square(8, 8), Square(5, 7), Square(8, 5)]:
        opp_player = game.get_reserves(game.state.home_team)[0]
        game.put(opp_player, sq)

    game.move(game.get_ball(), Square(9, 9))
    game.get_ball().is_carried = False

    cython_paths = cython_pathfinding.Pathfinder(game, player, trr=True).get_paths()
    python_paths = python_pathfinding.Pathfinder(game, player, directly_to_adjacent=True, trr=True).get_paths()

    def create_path_str_to_compare(path):
        return f"({path.get_last_step() .x}, {path.get_last_step() .y}) p={path.prob:.5f} len={len(path.steps)}"

    def create_path_str_to_debug(path):
        """ Only used for debugging, see below """
        return "->".join([f"({step.x}, {step.y})" for step in path.steps]) + f"rolls={path.rolls}"

    for python_path, cython_path in zip(python_paths, cython_paths):
        python_str = create_path_str_to_compare(python_path)
        cython_str = create_path_str_to_compare(cython_path)
        assert python_str == cython_str

        # Uncomment this when debugging. And comment the assertion above
        # if python_str != cython_str:
        #    print(f"slow = {python_str} \n {create_path_string(python_str)}")
        #    print(f"fast = {cython_str} \n {create_path_string(cython_str)}")

@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_straight_paths(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    game.put(player, Square(7, 7))
    paths = pf.get_all_paths(game, player)

    for path in paths:
        if path.get_last_step().x == 7:
            assert all(step.x==7 for step in path.steps)
        if path.get_last_step().y == 7:
            assert all(step.y == 7 for step in path.steps)