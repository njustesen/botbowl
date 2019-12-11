#!/usr/bin/env python3

from ffai import Action, ActionType, Square
from ffai import ProcBot
from ffai.core.load import *
from ffai.ai.registry import register_bot, make_bot
from ffai.core.game import Game
import time


class MyScriptedBot(ProcBot):

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.opp_team = None

    def new_game(self, game, team):
        """
        Called when a new game starts.
        """
        self.my_team = team
        self.opp_team = game.get_opp_team(team)

    def coin_toss_flip(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.TAILS)
        # return Action(ActionType.HEADS)

    def coin_toss_kick_receive(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game):
        """
        Move players from the reserves to the pitch
        """
        i = len(game.get_players_on_pitch(self.my_team))
        reserves = game.get_reserves(self.my_team)
        if i == 11 or len(reserves) == 0:
            return Action(ActionType.END_SETUP)
        player = reserves[0]
        y = 3
        x = 13 if game.is_team_side(Square(13, 3), self.my_team) else 14
        return Action(ActionType.PLACE_PLAYER, player=player, position=Square(x, y + i))

    def place_ball(self, game):
        """
        Place the ball when kicking.
        """
        left_center = Square(7, 8)
        right_center = Square(20, 8)
        if game.is_team_side(left_center, self.opp_team):
            return Action(ActionType.PLACE_BALL, position=left_center)
        return Action(ActionType.PLACE_BALL, position=right_center)

    def high_kick(self, game):
        """
        Select player to move under the ball.
        """
        ball_pos = game.get_ball_position()
        if game.is_team_side(game.get_ball_position(), self.my_team) and \
                game.get_player_at(game.get_ball_position()) is None:
            for player in game.get_players_on_pitch(self.my_team, up=True):
                if Skill.BLOCK in player.get_skills():
                    return Action(ActionType.PLACE_PLAYER, player=player, position=ball_pos)
        return Action(ActionType.SELECT_NONE)

    def touchback(self, game):
        """
        Select player to give the ball to.
        """
        p = None
        for player in game.get_players_on_pitch(self.my_team, up=True):
            if Skill.BLOCK in player.get_skills():
                return Action(ActionType.SELECT_PLAYER, player=player)
            p = player
        return Action(ActionType.SELECT_PLAYER, player=p)

    def turn(self, game):
        """
        Start a new player action.
        """
        # Start a new player turn
        players_available = set()
        for action_choice in game.state.available_actions:
            for player in action_choice.players:
                players_available.add(player)

        # Loop available players
        for player in players_available:

            # Start blitz action
            if game.is_blitz_available():
                return Action(ActionType.START_BLITZ, player=player)

            # Start block action
            if not game.is_blitz() and not game.is_quick_snap():
                adjacent_players = game.get_adjacent_opponents(player, down=False)
                if player.state.up and len(adjacent_players) > 0:
                    opp_player = adjacent_players[0]
                    dice = game.num_block_dice(player, opp_player)
                    favor = player.team if dice > 0 else opp_player.team
                    if favor == self.my_team and dice > 1:
                        return Action(ActionType.START_BLOCK, player=player)

            # Start pass action
            ball_pos = game.get_ball_position()
            player_with_ball = game.get_player_at(ball_pos)
            if player_with_ball is not None and player_with_ball.team == self.my_team and game.is_pass_available():
                return Action(ActionType.START_PASS, player=player)

            # Start handoff action
            if player_with_ball is not None and player_with_ball.team == self.my_team and game.is_handoff_available():
                return Action(ActionType.START_HANDOFF, player=player)

            # Start foul action
            adjacent_player_squares = game.get_adjacent_opponents(player, down=False)
            if game.is_foul_available() and len(adjacent_player_squares) > 0:
                return Action(ActionType.START_FOUL, player=player)

            # Start movement action
            return Action(ActionType.START_MOVE, player=player)

        # End turn
        return Action(ActionType.END_TURN)

    def quick_snap(self, game):
        return Action(ActionType.END_TURN)

    def blitz(self, game):
        return Action(ActionType.END_TURN)

    def player_action(self, game):
        """
        Move, block, pass, handoff or foul with the active player.
        """
        player = game.state.active_player

        # Loop through available actions
        for action_choice in game.state.available_actions:

            # Move action?
            if action_choice.action_type == ActionType.MOVE:

                # Loop through adjacent empty squares
                for position, agi_rolls in zip(action_choice.positions, action_choice.agi_rolls):
                    if len(agi_rolls) == 0:
                        return Action(ActionType.MOVE, position=position)

            # Block action?
            if action_choice.action_type == ActionType.BLOCK:

                # Loop through available blocks
                for position, block_rolls in zip(action_choice.positions, action_choice.block_rolls):

                    # Only do blocks with 1 die if attacker has block
                    if block_rolls >= 2 or (block_rolls == 1 and Skill.BLOCK in player.get_skills()):
                        return Action(ActionType.BLOCK, position=position)

            # Pass action?
            if action_choice.action_type == ActionType.PASS:

                # Loop through players to pass to
                for position, agi_rolls in zip(action_choice.positions, action_choice.agi_rolls):

                    catcher = game.get_player_at(position)
                    pass_distance = game.get_pass_distance(player, position)

                    # Don't pass to empty squares or long bombs
                    if catcher is not None and pass_distance != PassDistance.LONG_BOMB:
                        return Action(ActionType.PASS, position=position)

            # Hand-off action
            if action_choice.action_type == ActionType.HANDOFF:

                # Loop through players to hand-off to
                for position, agi_rolls in zip(action_choice.positions, action_choice.agi_rolls):
                    return Action(ActionType.HANDOFF, position=position)

            # Foul action
            if action_choice.action_type == ActionType.FOUL:

                # Loop through players to foul
                for position, block_rolls in zip(action_choice.positions, action_choice.block_rolls):
                    return Action(ActionType.FOUL, position=position)

        return Action(ActionType.END_PLAYER_TURN)

    def block(self, game):
        """
        Select block die or reroll.
        """
        # Loop through available dice results
        actions = set()
        for action_choice in game.state.available_actions:
            actions.add(action_choice.action_type)

        if ActionType.SELECT_DEFENDER_DOWN in actions:
            return Action(ActionType.SELECT_DEFENDER_DOWN)

        if ActionType.SELECT_DEFENDER_STUMBLES in actions:
            return Action(ActionType.SELECT_DEFENDER_STUMBLES)

        if ActionType.SELECT_PUSH in actions:
            return Action(ActionType.SELECT_PUSH)

        if ActionType.SELECT_BOTH_DOWN in actions:
            return Action(ActionType.SELECT_BOTH_DOWN)

        if ActionType.USE_REROLL in actions:
            return Action(ActionType.USE_REROLL)

        if ActionType.SELECT_ATTACKER_DOWN in actions:
            return Action(ActionType.SELECT_ATTACKER_DOWN)

    def push(self, game):
        """
        Select square to push to.
        """
        # Loop through available squares
        for position in game.state.available_actions[0].positions:
            return Action(ActionType.PUSH, position=position)

    def follow_up(self, game):
        """
        Follow up or not. ActionType.FOLLOW_UP must be used together with a position.
        """
        player = game.state.active_player
        for position in game.state.available_actions[0].positions:
            # Always follow up
            if player.position != position:
                return Action(ActionType.FOLLOW_UP, position=position)

    def apothecary(self, game):
        """
        Use apothecary?
        """
        return Action(ActionType.USE_APOTHECARY)
        # return Action(ActionType.DONT_USE_APOTHECARY)

    def interception(self, game):
        """
        Select interceptor.
        """
        for action in game.state.available_actions:
            if action.action_type == ActionType.INTERCEPTION:
                for player, agi_rolls in zip(action.players, action.agi_rolls):
                    return Action(ActionType.INTERCEPTION, player=player)
        return Action(ActionType.SELECT_NONE)

    def pass_action(self, game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def catch(self, game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def gfi(self, game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def dodge(self, game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def pickup(self, game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def end_game(self, game):
        """
        Called when a game endw.
        """
        winner = game.get_winning_team()
        print("Casualties: ", game.num_casualties())
        if winner is None:
            print("It's a draw")
        elif winner == self.my_team:
            print("I ({}) won".format(self.name))
        else:
            print("I ({}) lost".format(self.name))


# Register MyScriptedBot
register_bot('scripted', MyScriptedBot)

if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = load_config("bot-bowl-ii")
    config.competition_mode = True
    # config = get_config("ff-7.json")
    # config = get_config("ff-5.json")
    # config = get_config("ff-3.json")
    ruleset = load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = load_arena(config.arena)
    home = load_team_by_filename("human", ruleset)
    away = load_team_by_filename("human", ruleset)

    # Play 100 games
    for i in range(10):
        home_agent = make_bot('scripted')
        home_agent.name = "Scripted Bot 1"
        away_agent = make_bot('scripted')
        away_agent.name = "Scripted Bot 3"
        config.debug_mode = False
        game = Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)
