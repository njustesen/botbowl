from tests.util import *


def test_crowd_surf_ball_carrier():
    game, blocker, ball_carrier = get_custom_game_turn(player_positions=[(5, 2)],
                                                       opp_player_positions=[(5, 1)],
                                                       ball_position=(5, 1))

    with only_fixed_rolls(game, block_dice=[BBDieResult.PUSH]):
        game.step(Action(ActionType.START_BLOCK, player=blocker))
        game.step(Action(ActionType.BLOCK, player=ball_carrier))
        game.step(Action(ActionType.SELECT_PUSH, position=Square(5, 0)))
        game.step(Action(ActionType.PUSH, position=Square(5, 0)))

    # Fix crowd surf injury roll, throw in distance roll, throw in direction roll, bounce direction roll
    with only_fixed_rolls(game, d6=[2, 2, 2, 2], d3=[2], d8=[7]):
        game.step(Action(ActionType.FOLLOW_UP, position=blocker.position))

    # throw in is made from Square(5,1) and
    # should go in positive y-direction for 2+2+1=5 step. Ending up at Square(5,6)
    final_square = Square(5, 6)

    assert game.has_report_of_type(OutcomeType.THROW_IN)
    assert game.get_ball().position == final_square
