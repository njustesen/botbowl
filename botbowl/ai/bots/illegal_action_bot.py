from botbowl.ai.bots.random_bot import RandomBot
from botbowl.core.model import Action
from botbowl.core.table import ActionType


class IllegalActionBot(RandomBot):
    def __init__(self, name):
        super().__init__(name)
        self.i = 0

    def act(self, game):
        self.i += 1
        if self.i % 2 == 0:
            return Action(ActionType.USE_APOTHECARY) #the illegal action
        else:
            return super().act(game)
