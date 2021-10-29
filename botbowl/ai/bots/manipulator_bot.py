"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains an example bot that takes random actions.
"""
from botbowl.core.model import Agent, ActionType, Action
from botbowl.ai.registry import register_bot
import numpy as np


class ManipulatorBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        if game.home_agent == self:
            game.state.home_team.state.score = 1000
        elif game.away_agent == self:
            game.state.away_team.state.score = 1000

    def act(self, game):
        if game.home_agent == self:
            game.state.home_team.state.score = 1000
        elif game.away_agent == self:
            game.state.away_team.state.score = 1000
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break
        pos = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
        action = Action(action_choice.action_type, position=pos, player=player)
        return action

    def end_game(self, game):
        if game.home_agent == self:
            game.state.home_team.state.score = 1000
        elif game.away_agent == self:
            game.state.away_team.state.score = 1000
        pass
