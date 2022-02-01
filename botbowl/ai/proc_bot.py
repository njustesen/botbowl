"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains an example bot that takes random actions.
"""
from botbowl.core.model import Agent, Action
from botbowl.core.game import Game, InvalidActionError
from botbowl.core.procedure import *


class ProcBot(Agent):

    def __init__(self, name):
        super().__init__(name)

    def act(self, game: Game):

        # Get current procedure
        proc = game.get_procedure()
        # print(type(proc))
        
        action: Optional[Action] = None
        
        # Call private function
        if isinstance(proc, CoinTossFlip):
            action = self.coin_toss_flip(game)
        elif isinstance(proc, CoinTossKickReceive):
            action = self.coin_toss_kick_receive(game)
        elif isinstance(proc, Setup):
            if proc.reorganize:
                action = self.perfect_defense(game)
            else:
                action = self.setup(game)
        elif isinstance(proc, Ejection):
            action = self.use_bribe(game)
        elif isinstance(proc, Reroll):
            if proc.can_use_pro:
                action = self.use_pro(game)
            else:
                action = self.reroll(game)
        elif isinstance(proc, PlaceBall):
            action = self.place_ball(game)
        elif isinstance(proc, HighKick):
            action = self.high_kick(game)
        elif isinstance(proc, Touchback):
            action = self.touchback(game)
        elif isinstance(proc, Turn) and proc.quick_snap:
            action = self.quick_snap(game)
        elif isinstance(proc, Turn) and proc.blitz:
            action = self.blitz(game)
        elif isinstance(proc, Turn):
            action = self.turn(game)
        elif isinstance(proc, MoveAction):
            action = self.player_action(game)
        elif isinstance(proc, MoveAction):
            action = self.player_action(game)
        elif isinstance(proc, BlockAction):
            action = self.player_action(game)
        elif isinstance(proc, PassAction):
            action = self.player_action(game)
        elif isinstance(proc, HandoffAction):
            action = self.player_action(game)
        elif isinstance(proc, BlitzAction):
            action = self.player_action(game)
        elif isinstance(proc, FoulAction):
            action = self.player_action(game)
        elif isinstance(proc, ThrowBombAction):
            action = self.player_action(game)
        elif isinstance(proc, Block):
            if proc.waiting_juggernaut:
                action = self.use_juggernaut(game)
            elif proc.waiting_wrestle_attacker or proc.waiting_wrestle_defender:
                action = self.use_wrestle(game)
            else:
                action = self.block(game)
        elif isinstance(proc, Push):
            if proc.waiting_stand_firm:
                action = self.use_stand_firm(game)
            else:
                action = self.push(game)
        elif isinstance(proc, FollowUp):
            action = self.follow_up(game)
        elif isinstance(proc, Apothecary):
            action = self.apothecary(game)
        elif isinstance(proc, Interception):
            action = self.interception(game)
        elif isinstance(proc, BloodLustBlockOrMove):
            action = self.blood_lust_block_or_move(game)
        elif isinstance(proc, EatThrall):
            action = self.eat_thrall(game)
        else:
            raise Exception("Unknown procedure")

        if action is None:
            assert action is not None

        # handle illegal action. if handle_illegal_action() returns a new legal action, return that.
        if not game._is_action_allowed(action):
            new_action = self.handle_illegal_action(game, action)
            assert isinstance(new_action, Action)
            assert game._is_action_allowed(new_action)
            return new_action

        return action

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

    def perfect_defense(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def handle_illegal_action(self, game: Game, action: Action) -> Optional[Action]:
        """
        :param game: current game object
        :param action: action that was illegal
        :return: this particular implementation does not return, it raises an error with some
        helpful debug information. But if you override it in a subclass you may return a safe action
        e.g. pick a random action for the list of available action or used game._forced_action()
        """

        error_message = f"{action} is not allowed. "
        allowed_action_types = {action_choice.action_type for action_choice in game.get_available_actions()}
        if action.action_type not in allowed_action_types:
            allowed_action_types_names = ", ".join(action_type.name for action_type in allowed_action_types)
            error_message += f"Allowed action types are: {allowed_action_types_names}"
            set_trace()
            raise InvalidActionError(error_message)

        # Find the relevant action choice
        target_action_choice: Optional[ActionChoice] = None
        for action_choice in game.get_available_actions():
            if action_choice.action_type == action.action_type:
                target_action_choice = action_choice
                break

        if len(target_action_choice.positions) > 0:
            allowed_positions = ', '.join(map(str, target_action_choice.positions))
            if action.position is None and None not in target_action_choice.positions:
                error_message += f"position=None not allowed! Available positions are {allowed_positions}"
            elif action.position not in target_action_choice.positions:
                error_message += f"wrong position. Available positions are {allowed_positions}"
        else:
            error_message += "Other error, no details, sorry"

        raise InvalidActionError(error_message)


