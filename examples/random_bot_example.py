#!/usr/bin/env python3

from ffai.core.game import *
from ffai.core.model import *
from ffai.ai.registry import register_bot, make_bot
import numpy as np


class MyRandomBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team

    def act(self, game):
        # Select a random action type
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            # Ignore PLACE_PLAYER actions
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break

        # Select a random position and/or player
        pos = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None

        # Make action object
        action = Action(action_choice.action_type, pos=pos, player=player)

        # Return action to the framework
        return action

    def end_game(self, game):
        pass


# Register the bot to the framework
register_bot('my-random-bot', MyRandomBot)


if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = get_config("ff-11-bot-bowl-i.json")
    ruleset = get_rule_set(config.ruleset)
    arena = get_arena(config.arena)
    home = get_team_by_id("human-1", ruleset)
    away = get_team_by_id("human-2", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 10 games
    game_times = []
    for i in range(10):
        away_agent = make_bot("my-random-bot")
        home_agent = make_bot("my-random-bot")

        game = Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")
