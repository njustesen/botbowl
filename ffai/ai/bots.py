from bb.core.model import Agent, ActionType, Action
import numpy as np


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
        pos = self.rnd.choice(action_choice.positions) if len(action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(action_choice.players) > 0 else None
        action = Action(action_choice.action_type, pos=pos, player=player)
        self.actions_taken += 1
        return action

    def end_game(self, game):
        pass
        #winner = game.get_winner()
        #if winner is None:
        #    print("It's a draw")
        #elif winner == self.my_team:
        #    print("I ({}) won".format(self.name))
        #else:
        #    print("I ({}) lost".format(self.name))
        #print("I took", self.actions_taken, "actions")

