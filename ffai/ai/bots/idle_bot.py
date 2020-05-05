"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains an example bot that takes random actions.
"""
from ffai.core.model import Agent
from ffai.ai.registry import register_bot
import time


class IdleBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        
    def new_game(self, game, team):
        pass
        
    def act(self, game):
        time.sleep(1000000)
        return None

    def end_game(self, game):
        pass

