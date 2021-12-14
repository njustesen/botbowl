from tests.util import *
import pytest


@pytest.mark.parametrize("positions", [((5, 2), (5, 1)),
                                       ((2, 5), (1, 5)),
                                       ((25, 5), (26, 5)),
                                       ((5, 14), (5, 15)) ])
def test_crowd_surf_ball_carrier(positions):
    blocker_pos, ball_carrier_pos = positions
    game, (blocker, ball_carrier) = get_custom_game_turn(player_positions=[blocker_pos],
                                                         opp_player_positions=[ball_carrier_pos],
                                                         ball_position=ball_carrier_pos)
    throw_in_from_square = ball_carrier.position
    delta_x = ball_carrier.position.x - blocker.position.x
    delta_y = ball_carrier.position.y - blocker.position.y

    with only_fixed_rolls(game, block_dice=[BBDieResult.PUSH]):
        game.step(Action(ActionType.START_BLOCK, player=blocker))
        game.step(Action(ActionType.BLOCK, player=ball_carrier))
        game.step(Action(ActionType.SELECT_PUSH))
        push_to = game.get_square(ball_carrier.position.x + delta_x, ball_carrier.position.y + delta_y)
        game.step(Action(ActionType.PUSH, position=push_to))

    # Fix crowd surf injury roll, throw in distance roll, throw in direction roll, bounce direction roll
    throw_in_distance_1 = 2
    throw_in_distance_2 = 2

    with only_fixed_rolls(game, d6=[2, 2, throw_in_distance_1, throw_in_distance_2], d3=[2], d8=[7]):
        game.step(Action(ActionType.FOLLOW_UP, position=blocker.position))

    # throw in is made from Square(5,1) and
    # should go in positive y-direction for 2+2+1=5 step. Ending up at Square(5,6)
    throw_in_distance = throw_in_distance_1 + throw_in_distance_2

    final_square = game.get_square(throw_in_from_square.x - throw_in_distance*delta_x,
                                   throw_in_from_square.y - throw_in_distance*delta_y + 1)

    assert game.has_report_of_type(OutcomeType.THROW_IN)
    assert game.get_ball().position == final_square
