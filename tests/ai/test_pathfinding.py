from tests.util import Square, get_game_turn, Skill
import pytest

import ffai.core.pathfinding.python_pathfinding as python_pathfinding
pathfinding_modules_to_test = [python_pathfinding]

# We only test the cython pathfindng module if it exists. Otherwise half of the tests fail.
# This way only test_compare_cython_python_paths() fails because it explicitly uses cython pf module.
try:
    import ffai.core.pathfinding.cython_pathfinding as cython_pathfinding
    pathfinding_modules_to_test.append(cython_pathfinding)
except ImportError:
    cython_pathfinding = None #this will make the cython vs. python compare test fail.

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
        assert len(path.steps) == moves and path.steps[-1] == target
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


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_all_blitz_paths(pf):
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
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 1
    game.put(player, Square(1, 1))

    opp_player_1 = game.get_reserves(game.state.home_team)[0]
    opp_player_1.state.up = False
    opp_position_1 = Square(1, 2)
    game.put(opp_player_1, opp_position_1)

    opp_player_2 = game.get_reserves(game.state.home_team)[1]
    opp_player_2.state.up = False
    opp_position_2 = Square(1, 3)
    game.put(opp_player_2, opp_position_2)

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
            if step in [opp_position_1, opp_position_2]:
                fouls += 1
                assert step == path.steps[-1]
        assert fouls <= 1
        total_fouls += fouls
    assert total_fouls == 2


@pytest.mark.parametrize("pf", pathfinding_modules_to_test)
def test_handoff(pf):
    game = get_game_turn(empty=True)
    player = game.get_reserves(game.state.away_team)[0]
    player.role.ma = 1
    game.move(game.get_ball(), player.position)
    game.put(player, Square(1, 1))
    game.get_ball().is_carried = True
    teammate_1 = game.get_reserves(game.state.away_team)[1]
    teammate_position_1 = Square(1, 2)
    game.put(teammate_1, teammate_position_1)
    teammate_2 = game.get_reserves(game.state.away_team)[2]
    teammate_position_2 = Square(1, 3)
    game.put(teammate_2, teammate_position_2)

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
            if step in [teammate_position_1, teammate_position_2]:
                handoffs += 1
                assert step == path.steps[-1]
        assert handoffs <= 1
        total_handoffs += handoffs
    assert total_handoffs == 2


def test_compare_cython_python_paths():
    """
    This test compares paths calculated by cython and python. They should be same
    However, it does not compare every individual step, only destination, probability and path length.
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
        return f"({path.steps[-1].x}, {path.steps[-1].y}) p={path.prob:.5f} len={len(path.steps)}"

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
