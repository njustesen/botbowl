import botbowl
from examples.scripted_bot_example import MyScriptedBot
from examples.random_bot_example import MyRandomBot

from tests.util import get_game_coin_toss

from random import randint

from pprint import pprint


def test_scripted_bot():
    game = get_game_coin_toss(seed=randint(0, 2**32))
    game.config.pathfinding_enabled = False

    scripted_bot_home = MyScriptedBot('scripted_test_bot1')
    scripted_bot_home.new_game(game, game.state.home_team)
    game.replace_home_agent(scripted_bot_home)

    scripted_bot_away = MyScriptedBot('scripted_test_bot2')
    scripted_bot_away.new_game(game, game.state.away_team)
    game.replace_away_agent(scripted_bot_away)

    game.set_available_actions()

    game.step(botbowl.Action(botbowl.ActionType.START_GAME))