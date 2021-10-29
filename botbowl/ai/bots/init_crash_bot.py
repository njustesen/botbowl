"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains an example bot that takes random actions.
"""
from botbowl.core.model import Agent
from botbowl.ai.registry import register_bot


class InitCrashBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        v = 1 / 0
        
    def new_game(self, game, team):
        pass
        
    def act(self, game):
        return None

    def end_game(self, game):
        pass
