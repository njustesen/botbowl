import ffai.web.api as api
import ffai.ai.bots as bot
import ffai.core.model as m
import ffai.core.procedure as proc
import ffai.util.pathfinding as pf
import time


class GrodBot(bot.ProcBot):

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

    def start_game(self, game):
        """
        Just start the game.
        """
        return m.Action(m.ActionType.START_GAME)

    def coin_toss_flip(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return m.Action(m.ActionType.TAILS)
        # return Action(ActionType.HEADS)

    def coin_toss_kick_receive(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return m.Action(m.ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game):
        """
        Move players from the reserves to the pitch
        """
        i = len(game.get_players_on_pitch(self.my_team))
        reserves = game.get_reserves(self.my_team)
        if i == 11 or len(reserves) == 0:
            return m.Action(m.ActionType.END_SETUP)
        player = reserves[0]
        y = 3
        x = 13 if game.is_team_side(m.Square(13, 3), self.my_team) else 14
        return m.Action(m.ActionType.PLACE_PLAYER, player=player, pos=m.Square(x, y + i))

    def place_ball(self, game):
        """
        Place the ball when kicking.
        """
        left_center = m.Square(7, 8)
        right_center = m.Square(20, 8)
        if game.is_team_side(left_center, self.opp_team):
            return m.Action(m.ActionType.PLACE_BALL, pos=left_center)
        return m.Action(m.ActionType.PLACE_BALL, pos=right_center)

    def high_kick(self, game):
        """
        Select player to move under the ball.
        """
        ball_pos = game.get_ball_position()
        if game.is_team_side(game.get_ball_position(), self.my_team) and \
                game.get_player_at(game.get_ball_position()) is None:
            for player in game.get_players_on_pitch(self.my_team, up=True):
                if m.Skill.BLOCK in player.skills:
                    return m.Action(m.ActionType.PLACE_PLAYER, player=player, pos=ball_pos)
        return m.Action(m.ActionType.SELECT_NONE)

    def touchback(self, game):
        """
        Select player to give the ball to.
        """
        for player in game.get_players_on_pitch(self.my_team, up=True):
            if m.Skill.BLOCK in player.skills:
                return m.Action(m.ActionType.SELECT_PLAYER, player=player)
        return m.Action(m.ActionType.SELECT_NONE)

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
            player_square: m.Square = player.position
            player_mover = pf.FfMover(player)
            ff_map = pf.FfTileMap(game.state)
            finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True, heuristic=pf.BruteForceHeuristic())
            paths = finder.find_paths(player_mover, player_square.x, player_square.y)

            # Start blitz action
            if game.is_blitz_available():
                return m.Action(m.ActionType.START_BLITZ, player=player)

            # Start block action
            if not game.is_blitz() and not game.is_quick_snap():
                adjacent_player_squares = game.adjacent_player_squares(player, include_own=False, include_opp=True, only_blockable=True)
                if player.state.up and len(adjacent_player_squares) > 0:
                    opp_player = game.get_player_at(adjacent_player_squares[0])
                    dice, favor = proc.Block.dice_and_favor(game, player, opp_player)
                    if favor == self.my_team and dice > 1:
                        return m.Action(m.ActionType.START_BLOCK, player=player)

            # Start pass action
            ball_pos = game.get_ball_position()
            player_with_ball = game.get_player_at(ball_pos)
            if player_with_ball is not None and player_with_ball.team == self.my_team and game.is_pass_available():
                return m.Action(m.ActionType.START_PASS, player=player)

            # Start handoff action
            if player_with_ball is not None and player_with_ball.team == self.my_team and game.is_handoff_available():
                return m.Action(m.ActionType.START_HANDOFF, player=player)

            # Start foul action
            adjacent_player_squares = game.adjacent_player_squares(player, include_own=False, include_opp=True,
                                                                   only_foulable=True)
            if game.is_foul_available() and len(adjacent_player_squares) > 0:
                return m.Action(m.ActionType.START_FOUL, player=player)

            # Start movement action
            return m.Action(m.ActionType.START_MOVE, player=player)

        # End turn
        return m.Action(m.ActionType.END_TURN)

    def player_action(self, game):
        """
        Move, block, pass, handoff or foul with the active player.
        """
        player = game.state.active_player

        # Loop through available actions
        for action_choice in game.state.available_actions:

            # Move action?
            if action_choice.action_type == m.ActionType.MOVE:

                # Loop through adjacent empty squares
                for position, agi_rolls in zip(action_choice.positions, action_choice.agi_rolls):
                    if len(agi_rolls) == 0:
                        return m.Action(m.ActionType.MOVE, pos=position)

            # Block action?
            if action_choice.action_type == m.ActionType.BLOCK:

                # Loop through available blocks
                for position, block_rolls in zip(action_choice.positions, action_choice.block_rolls):

                    # Only do blocks with 1 die if attacker has block
                    if block_rolls >= 2 or (block_rolls == 1 and m.Skill.BLOCK in player.skills):
                        opp_player = game.get_player_at(position)
                        return m.Action(m.ActionType.BLOCK, pos=position)

            # Pass action?
            if action_choice.action_type == m.ActionType.PASS:

                # Loop through players to pass to
                for position, agi_rolls in zip(action_choice.positions, action_choice.agi_rolls):

                    catcher = game.get_player_at(position)
                    pass_distance = game.pass_distance(player, position)

                    # Don't pass to empty squares or long bombs
                    if catcher is not None and pass_distance != m.PassDistance.LONG_BOMB:
                        return m.Action(m.ActionType.PASS, pos=position)

            # Hand-off action
            if action_choice.action_type == m.ActionType.HANDOFF:

                # Loop through players to hand-off to
                for position, agi_rolls in zip(action_choice.positions, action_choice.agi_rolls):
                    return m.Action(m.ActionType.HANDOFF, pos=position)

            # Foul action
            if action_choice.action_type == m.ActionType.FOUL:

                # Loop through players to foul
                for position, block_rolls in zip(action_choice.positions, action_choice.block_rolls):
                    return m.Action(m.ActionType.FOUL, pos=position)

        return m.Action(m.ActionType.END_PLAYER_TURN)

    def block(self, game):
        """
        Select block die or reroll.
        """
        # Loop through available dice results
        actions = set()
        for action_choice in game.state.available_actions:
            actions.add(action_choice.action_type)

        if m.ActionType.SELECT_DEFENDER_DOWN in actions:
            return m.Action(m.ActionType.SELECT_DEFENDER_DOWN)

        if m.ActionType.SELECT_DEFENDER_STUMBLES in actions:
            return m.Action(m.ActionType.SELECT_DEFENDER_STUMBLES)

        if m.ActionType.SELECT_PUSH in actions:
            return m.Action(m.ActionType.SELECT_PUSH)

        if m.ActionType.SELECT_BOTH_DOWN in actions:
            return m.Action(m.ActionType.SELECT_BOTH_DOWN)

        if m.ActionType.USE_REROLL in actions:
            return m.Action(m.ActionType.USE_REROLL)

        if m.ActionType.SELECT_ATTACKER_DOWN in actions:
            return m.Action(m.ActionType.SELECT_ATTACKER_DOWN)

    def push(self, game):
        """
        Select square to push to.
        """
        # Loop through available squares
        for position in game.state.available_actions[0].positions:
            return m.Action(m.ActionType.PUSH, pos=position)

    def follow_up(self, game):
        """
        Follow up or not. ActionType.FOLLOW_UP must be used together with a position.
        """
        player = game.state.active_player
        for position in game.state.available_actions[0].positions:
            # Always follow up
            if player.position != position:
                return m.Action(m.ActionType.FOLLOW_UP, pos=position)

    def apothecary(self, game):
        """
        Use apothecary?
        """
        return m.Action(m.ActionType.USE_APOTHECARY)
        # return Action(ActionType.DONT_USE_APOTHECARY)

    def interception(self, game):
        """
        Select interceptor.
        """
        for action in game.state.available_actions:
            if action.action_type == m.ActionType.INTERCEPTION:
                for player, agi_rolls in zip(action.players, action.agi_rolls):
                    return m.Action(m.ActionType.INTERCEPTION, player=player)
        return m.Action(m.ActionType.SELECT_NONE)

    def pass_action(self, game):
        """
        Reroll or not.
        """
        return m.Action(m.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def catch(self, game):
        """
        Reroll or not.
        """
        return m.Action(m.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def gfi(self, game):
        """
        Reroll or not.
        """
        return m.Action(m.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def dodge(self, game):
        """
        Reroll or not.
        """
        return m.Action(m.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def pickup(self, game):
        """
        Reroll or not.
        """
        return m.Action(m.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def end_game(self, game):
        """
        Called when a game endw.
        """
        winner = game.get_winner()
        print("Casualties: ", game.num_casualties())
        if winner is None:
            print("It's a draw")
        elif winner == self.my_team:
            print("I ({}) won".format(self.name))
        else:
            print("I ({}) lost".format(self.name))


# Register MyScriptedBot
api.register_bot('GrodBot', GrodBot)

if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = api.get_config("ff-11.json")
    config.competition_mode = False
    # config = get_config("ff-7.json")
    # config = get_config("ff-5.json")
    # config = get_config("ff-3.json")
    ruleset = api.get_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = api.get_arena(config.arena)
    home = api.get_team_by_id("human-1", ruleset)
    away = api.get_team_by_id("human-2", ruleset)

    # Play 100 games
    for i in range(100):
        away_agent = GrodBot("Scripted Bot 1")
        home_agent = GrodBot("Scripted Bot 2")
        config.debug_mode = False
        game = api.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i + 1))
        start = time.time()
        game.init()
        game.step()
        end = time.time()
        print(end - start)
