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


class ViolatorBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)
        
    def new_game(self, game, team):
        pass
        
    def act(self, game):
        while time.time() < game.action_termination_time() + game.config.time_limits.disqualification:
            time.sleep(1)
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


class JustInTimeBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)
        
    def new_game(self, game, team):
        pass
        
    def act(self, game):
        while time.time() < game.action_termination_time():
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


class ChrashBot(Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        
    def new_game(self, game, team):
        pass
        
    def act(self, game):
        v = 1 / 0
        return None

    def end_game(self, game):
        pass


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

# Register bots
register_bot('random', RandomBot)
register_bot('crash', ChrashBot)
register_bot('idle', IdleBot)
register_bot('violator', ViolatorBot)
register_bot('init-crash', InitCrashBot)
register_bot('just-in-time', JustInTimeBot)


class ProcBot(Agent):

    def __init__(self, name):
        super().__init__(name)

    def act(self, game):

        # Get current procedure
        proc = game.state.stack.peek()

        # Call private function
        if isinstance(proc, CoinTossFlip):
            return self.coin_toss_flip(game)
        if isinstance(proc, CoinTossKickReceive):
            return self.coin_toss_kick_receive(game)
        if isinstance(proc, Setup):
            return self.setup(game)
        if isinstance(proc, PlaceBall):
            return self.place_ball(game)
        if isinstance(proc, HighKick):
            return self.high_kick(game)
        if isinstance(proc, Touchback):
            return self.touchback(game)
        if isinstance(proc, Turn) and proc.quick_snap:
            return self.quick_snap(game)
        if isinstance(proc, Turn) and proc.blitz:
            return self.blitz(game)
        if isinstance(proc, Turn):
            return self.turn(game)
        if isinstance(proc, PlayerAction):
            return self.player_action(game)
        if isinstance(proc, Block):
            return self.block(game)
        if isinstance(proc, Push):
            return self.push(game)
        if isinstance(proc, FollowUp):
            return self.follow_up(game)
        if isinstance(proc, Apothecary):
            return self.apothecary(game)
        if isinstance(proc, PassAction):
            return self.pass_action(game)
        if isinstance(proc, Catch):
            return self.catch(game)
        if isinstance(proc, Interception):
            return self.interception(game)
        if isinstance(proc, GFI):
            return self.gfi(game)
        if isinstance(proc, Dodge):
            return self.dodge(game)
        if isinstance(proc, Pickup):
            return self.pickup(game)

        raise Exception("Unknown procedure")

    def coin_toss_flip(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def coin_toss_kick_receive(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def setup(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def place_ball(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def high_kick(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def touchback(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def turn(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def quick_snap(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def blitz(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def player_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def block(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def push(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def follow_up(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def apothecary(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def pass_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def catch(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def interception(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def gfi(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def dodge(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def pickup(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")
