"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains a random bot that takes random actions.
"""
import numpy as np
from ffai.core.model import Agent, ActionType, Action, Square 
from ffai.ai.registry import register_bot
from ffai.core.procedure import PlaceBall 

class AlmostRandomBot(Agent):


    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.actions_taken = 0
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team
        self.actions_taken = 0
        self.opp_team = game.get_opp_team(team)

    def act(self, game):
        
        self.actions_taken += 1
        
        if len(game.state.pitch.board) == 17: # 11 player board  
            proc = game.get_procedure()
            if isinstance(proc, PlaceBall):
                return self.place_ball(game)
            
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            if action_choice.action_type != ActionType.PLACE_PLAYER:
                break
        position = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
        action = Action(action_choice.action_type, position=position, player=player)
        
        return action    
    
    def place_ball(self, game):
        """
        Place the ball when kicking. Don't cause touchbacks! 
        """
        left_center = Square(7, 8)
        right_center = Square(20, 8)
        if game.is_team_side(left_center, self.opp_team):
            return Action(ActionType.PLACE_BALL, position=left_center)
        return Action(ActionType.PLACE_BALL, position=right_center)    
    
    def end_game(self, game):
        pass

register_bot("almost-random", AlmostRandomBot)
