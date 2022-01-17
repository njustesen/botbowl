import botbowl
from examples.scripted_bot_example import MyScriptedBot
from examples.random_bot_example import MyRandomBot

from tests.util import get_game_coin_toss

from pprint import pprint

def test_scripted_bot():
    game = get_game_coin_toss(0)
    game.config.pathfinding_enabled = False

    scripted_bot_home = MyScriptedBot('scripted_test_bot1')
    scripted_bot_home.new_game(game, game.state.home_team)

    scripted_bot_away = MyScriptedBot('scripted_test_bot2')
    scripted_bot_away.new_game(game, game.state.away_team)

    game.step(botbowl.Action(botbowl.ActionType.START_GAME))

    while not game.state.game_over:
        if game.state.available_actions[0].team is game.state.away_team:
            action = scripted_bot_away.act(game)
        elif game.state.available_actions[0].team is game.state.home_team:
            action = scripted_bot_home.act(game)
        else:
            raise ValueError('Should not get here!')

        # This is needed because Reroll procedure's available actions are dependent on game.actor.human
        if action.action_type is botbowl.ActionType.DONT_USE_REROLL:
            action.action_type = botbowl.ActionType.USE_REROLL

        game.step(action)
