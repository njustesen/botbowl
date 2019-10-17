"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains an example bot that takes random actions.
"""

from ffai.core.procedure import *
from ffai.ai.registry import register_bot
import time


class JustInTimeBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)
        
    def new_game(self, game, team):
        self.my_team = team
        
    def act(self, game):
        while time.time() < game.seconds_left(self.my_team):
            time.sleep(0.01)
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break
        pos = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
        action = Action(action_choice.action_type, pos=pos, player=player)
        return action

    def end_game(self, game):
        pass


# Register bots
register_bot('just-in-time', JustInTimeBot)

