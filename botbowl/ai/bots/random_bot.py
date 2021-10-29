"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains a random bot that takes random actions.
"""
import numpy as np
from botbowl.core.model import Agent, ActionType, Action
from botbowl.ai.registry import register_bot


class RandomBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.actions_taken = 0
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team
        self.actions_taken = 0

    def act(self, game):
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break
        position = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
        action = Action(action_choice.action_type, position=position, player=player)
        self.actions_taken += 1
        return action

    def end_game(self, game):
        pass

