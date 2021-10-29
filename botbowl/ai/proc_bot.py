"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains an example bot that takes random actions.
"""
from botbowl.core.model import Agent
from botbowl.core.procedure import *


class ProcBot(Agent):

    def __init__(self, name):
        super().__init__(name)

    def act(self, game):

        # Get current procedure
        proc = game.get_procedure()
        # print(type(proc))

        # Call private function
        if isinstance(proc, CoinTossFlip):
            return self.coin_toss_flip(game)
        if isinstance(proc, CoinTossKickReceive):
            return self.coin_toss_kick_receive(game)
        if isinstance(proc, Setup):
            return self.setup(game)
        if isinstance(proc, Ejection):
            return self.use_bribe(game)
        if isinstance(proc, Reroll):
            if proc.can_use_pro:
                return self.use_pro(game)
            return self.reroll(game)
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
        if isinstance(proc, MoveAction):
            return self.player_action(game)
        if isinstance(proc, MoveAction):
            return self.player_action(game)
        if isinstance(proc, BlockAction):
            return self.player_action(game)
        if isinstance(proc, PassAction):
            return self.player_action(game)
        if isinstance(proc, HandoffAction):
            return self.player_action(game)
        if isinstance(proc, BlitzAction):
            return self.player_action(game)
        if isinstance(proc, FoulAction):
            return self.player_action(game)
        if isinstance(proc, ThrowBombAction):
            return self.player_action(game)
        if isinstance(proc, Block):
            if proc.waiting_juggernaut:
                return self.use_juggernaut(game)
            if proc.waiting_wrestle_attacker or proc.waiting_wrestle_defender:
                return self.use_wrestle(game)
            return self.block(game)
        if isinstance(proc, Push):
            if proc.waiting_stand_firm:
                return self.use_stand_firm(game)
            return self.push(game)
        if isinstance(proc, FollowUp):
            return self.follow_up(game)
        if isinstance(proc, Apothecary):
            return self.apothecary(game)
        if isinstance(proc, Interception):
            return self.interception(game)
        if isinstance(proc, BloodLustBlockOrMove):
            return self.blood_lust_block_or_move(game)
        if isinstance(proc, EatThrall):
            return self.eat_thrall(game)

        raise Exception("Unknown procedure")

    def use_pro(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_juggernaut(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_wrestle(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_stand_firm(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def coin_toss_flip(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def coin_toss_kick_receive(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def setup(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def reroll(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_bribe(self, game):
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

    def interception(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def gfi(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def dodge(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def pickup(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def blood_lust_block_or_move(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def eat_thrall(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")
