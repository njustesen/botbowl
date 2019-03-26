import ffai.web.api as api
import ffai.ai.bots as bot
import ffai.core.model as m
import ffai.core.table as t
import ffai.core.procedure as proc
import ffai.util.pathfinding as pf
from typing import Optional, List
import time
import ffai.core.game as g
import random
import math
import ffai.util.bothelper as helper
from ffai.util.bothelper import ActionSequence
from enum import Enum
from operator import itemgetter


class GrodBot(bot.ProcBot):
    ''' A Bot that uses path finding to evaluate all possibilities.

    WIP!!! Hand-offs and Pass actions going a bit funny.

    '''

    BASE_SCORE_BLITZ = 55.0
    BASE_SCORE_FOUL = -50.0
    BASE_SCORE_BLOCK = 65       # For a two dice block
    BASE_SCORE_HANDOFF = 40.0
    BASE_SCORE_MOVE_TO_OPPONENT = 45.0
    BASE_SCORE_MOVE_BALL = 45.0
    BASE_SCORE_MOVE_TOWARD_BALL = 15.0
    BASE_SCORE_MOVE_TO_SWEEP = 0.0
    BASE_SCORE_CAGE_BALL = 70.0
    BASE_SCORE_MOVE_TO_BALL = 60.0   #60.0
    BASE_SCORE_BALL_AND_CHAIN = 75.0
    BASE_SCORE_DEFENSIVE_SCREEN = 0.0
    ADDITIONAL_SCORE_DODGE = 0.0  # Lower this value to dodge more.

    def __init__(self, name, verbose=True):
        super().__init__(name)
        self.my_team = None
        self.opp_team = None
        self.current_move: Optional[ActionSequence, None] = None
        self.verbose = verbose

    def new_game(self, game: g.Game, team):
        """
        Called when a new game starts.
        """
        self.my_team = team
        self.opp_team = game.get_opp_team(team)

    def start_game(self, game: g.Game):
        """
        Just start the game.
        """
        return m.Action(t.ActionType.START_GAME)

    def coin_toss_flip(self, game: g.Game):
        """
        Select heads/tails and/or kick/receive
        """
        return m.Action(t.ActionType.TAILS)
        # return Action(ActionType.HEADS)

    def coin_toss_kick_receive(self, game: g.Game):
        """
        Select heads/tails and/or kick/receive
        """
        return m.Action(t.ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game: g.Game) -> m.Action:
        """
        Move players from the reserves to the pitch
        """
        if not helper.get_players(game, self.my_team, include_own=True, include_opp=False, include_off_pitch=False):
            # If no players are on the pitch yet, create a new ActionSequence for the setup.
            action_steps: List[m.Action] = []

            turn = game.state.round
            half = game.state.half
            opp_score = 0
            for team in game.state.teams:
                if team != self.my_team:
                    opp_score = max(opp_score, team.state.score)
            score_diff = self.my_team.state.score - opp_score

            # Choose 11 best players to field
            players_available: List[m.Player] = []
            for available_action in game.state.available_actions:
                if available_action.action_type == t.ActionType.PLACE_PLAYER:
                    players_available = available_action.players

            players_sorted_value = sorted(players_available, key=lambda x: player_value(game, x), reverse=True)
            n_keep: int = min(11, len(players_sorted_value))
            players_available = players_sorted_value[:n_keep]

            # Are we kicking or receiving?
            if game.state.receiving_this_drive:
                place_squares: List[m.Square] = []

                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 7))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 8))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 9))
                # Receiver next
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 8), 8))
                # Support line players
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 10))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 11))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 5))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 13))
                # A bit wide semi-defensive
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 11), 4))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 11), 12))
                # Extra help at the back
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 10), 8))

                players_sorted_bash = sorted(players_available, key=lambda x: player_bash_ability(game, x), reverse=True)
                players_sorted_blitz = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)

            else:
                place_squares: List[m.Square] = []

                # LOS squares first
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 7))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 8))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 13), 9))

                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 12), 3))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 12), 13))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 11), 2))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 11), 14))

                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 12), 5))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 12), 10))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 11), 11))
                place_squares.append(game.state.pitch.get_square(helper.reverse_x_for_right(game, self.my_team, 11), 5))

                players_sorted_bash = sorted(players_available, key=lambda x: player_bash_ability(game, x), reverse=True)
                players_sorted_blitz = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)

            for i in range(len(players_available)):
                action_steps.append(m.Action(t.ActionType.PLACE_PLAYER, player=players_sorted_bash[i], pos=place_squares[i]))

            action_steps.append(m.Action(t.ActionType.END_SETUP))

            self.current_move = ActionSequence(action_steps, description='Setup')

        # We must have initialised the action sequence, lets execute it
        next_action: m.Action = self.current_move.popleft()
        return next_action

    def place_ball(self, game: g.Game):
        """
        Place the ball when kicking.
        """
        left_center = m.Square(7, 8)
        right_center = m.Square(20, 8)
        if game.is_team_side(left_center, self.opp_team):
            return m.Action(t.ActionType.PLACE_BALL, pos=left_center)
        return m.Action(t.ActionType.PLACE_BALL, pos=right_center)

    def high_kick(self, game: g.Game):
        """
        Select player to move under the ball.
        """
        ball_pos = game.get_ball_position()
        if game.is_team_side(game.get_ball_position(), self.my_team) and \
                game.get_player_at(game.get_ball_position()) is None:
            for player in game.get_players_on_pitch(self.my_team, up=True):
                if t.Skill.BLOCK in player.skills:
                    return m.Action(t.ActionType.PLACE_PLAYER, player=player, pos=ball_pos)
        return m.Action(t.ActionType.SELECT_NONE)

    def touchback(self, game: g.Game):
        """
        Select player to give the ball to.
        """
        for player in game.get_players_on_pitch(self.my_team, up=True):
            if t.Skill.BLOCK in player.skills:
                return m.Action(t.ActionType.SELECT_PLAYER, player=player)
        return m.Action(t.ActionType.SELECT_NONE)

    def set_next_move(self, game: g.Game):
        ''' Set self.current_move

        :param game:
        '''
        self.current_move = None

        # Calculate path-finders

        #heatMap = new BloodBowlHeatMap(gameInspector);
        #heatMap.AddUnitByPathFinders(curPossibleOppositionMoves);
        #heatMap.AddUnitByPathFinders(curPossibleTeamMovesThisTurn);
        #heatMap.AddPlayersMoved(gameInspector.GetListFriendlyPlayersMoved());


        paths_other = dict()
        paths_blitz = dict()    # Blitz paths need at least 1 square of movement left for the block at the end
        ff_map = pf.FfTileMap(game.state)
        for action_choice in game.state.available_actions:
            for player in action_choice.players:
                if action_choice.action_type == t.ActionType.START_BLITZ:
                    if player not in paths_blitz:
                        player_square: m.Square = player.position
                        player_mover = pf.FfMover(player)
                        finder = pf.AStarPathFinder(ff_map, player.move_allowed()-1, allow_diag_movement=True, heuristic=pf.BruteForceHeuristic(), probability_costs=True)
                        paths = finder.find_paths(player_mover, player_square.x, player_square.y)
                        paths_blitz[player] = paths
                elif action_choice.action_type in (t.ActionType.START_MOVE, t.ActionType.START_PASS, t.ActionType.START_FOUL, t.ActionType.START_HANDOFF):
                    if player not in paths_other:
                        player_square: m.Square = player.position
                        player_mover = pf.FfMover(player)
                        finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True,heuristic=pf.BruteForceHeuristic(), probability_costs=True)
                        paths = finder.find_paths(player_mover, player_square.x, player_square.y)
                        paths_other[player] = paths

        all_actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            if action_choice.action_type == t.ActionType.START_MOVE:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = paths_other[player]
                    all_actions.extend(potential_move_actions(game, player, paths))
            elif action_choice.action_type == t.ActionType.START_BLITZ:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = paths_blitz[player]
                    all_actions.extend(potential_blitz_actions(game, player, paths))
            elif action_choice.action_type == t.ActionType.START_FOUL:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = paths_other[player]
                    all_actions.extend(potential_foul_actions(game, player, paths))
            elif action_choice.action_type == t.ActionType.START_BLOCK:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    all_actions.extend(potential_block_actions(game, player))
            elif action_choice.action_type == t.ActionType.START_PASS:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.state.pitch.get_ball_position() == player_square:
                        paths = paths_other[player]
                        all_actions.extend(potential_pass_actions(game, player, paths))
            elif action_choice.action_type == t.ActionType.START_HANDOFF:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.state.pitch.get_ball_position() == player_square:
                        paths = paths_other[player]
                        all_actions.extend(potential_handoff_actions(game, player, paths))
            elif action_choice.action_type == t.ActionType.END_TURN:
                all_actions.extend(potential_end_turn_action(game))

        if all_actions:
            all_actions.sort(key = lambda x: x.score, reverse=True)
            self.current_move = all_actions[0]

            if self.verbose:
                print('   Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + game.state.current_team.name + ', Action=' + self.current_move.description + ', Score=' + str(self.current_move.score))

    def turn(self, game: g.Game) -> m.Action:
        """
        Start a new player action / turn.
        """

        # Simple algorithm: 
        #   Loop through all available (yet to move) players.
        #   Compute all possible moves for all players.
        #   Assign a score to each action for each player.
        #   The player/play with the highest score is the one the Bot will attempt to use.
        #   Store a representation of this turn internally (for use by player-action) and return the action to begin.

        self.set_next_move(game)
        next_action: m.Action = self.current_move.popleft()
        return next_action

    def quick_snap(self, game: g.Game):

        self.current_move = None
        return m.Action(t.ActionType.END_TURN)

    def blitz(self, game: g.Game):

        self.current_move = None
        return m.Action(t.ActionType.END_TURN)

    def player_action(self, game: g.Game):
        """
        Take the next action from the current stack and execute
        """

        action_step = self.current_move.popleft()
        return action_step

    def block(self, game: g.Game):
        """
        Select block die or reroll.
        """
        # Loop through available dice results
        active_player: m.Player = game.state.active_player
        attacker: m.Player = game.state.stack.items[-1].attacker
        defender: m.Player = game.state.stack.items[-1].defender

        actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(action_choice.action_type))

            score = block_favourability(action_choice.action_type, active_player, attacker, defender)
            actions.append(ActionSequence(action_steps, score=score, description='Block die choice'))

        actions.sort(key = lambda x: x.score, reverse=True)
        current_move = actions[0]
        return current_move.action_steps[0]

    def push(self, game: g.Game):
        """
        Select square to push to.
        """
        # Loop through available squares
        for position in game.state.available_actions[0].positions:
            return m.Action(t.ActionType.PUSH, pos=position)

    def follow_up(self, game: g.Game):
        """
        Follow up or not. ActionType.FOLLOW_UP must be used together with a position.
        """
        player = game.state.active_player
        for position in game.state.available_actions[0].positions:
            # Always follow up
            if player.position != position:
                return m.Action(t.ActionType.FOLLOW_UP, pos=position)

    def apothecary(self, game: g.Game):
        """
        Use apothecary?
        """
        return m.Action(t.ActionType.USE_APOTHECARY)
        # return Action(ActionType.DONT_USE_APOTHECARY)

    def interception(self, game: g.Game):
        """
        Select interceptor.
        """
        for action in game.state.available_actions:
            if action.action_type == t.ActionType.INTERCEPTION:
                for player, agi_rolls in zip(action.players, action.agi_rolls):
                    return m.Action(t.ActionType.INTERCEPTION, player=player)
        return m.Action(t.ActionType.SELECT_NONE)

    def pass_action(self, game: g.Game):
        """
        Reroll or not.
        """
        return m.Action(t.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def catch(self, game: g.Game):
        """
        Reroll or not.
        """
        return m.Action(t.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def gfi(self, game: g.Game):
        """
        Reroll or not.
        """
        return m.Action(t.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def dodge(self, game: g.Game):
        """
        Reroll or not.
        """
        return m.Action(t.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def pickup(self, game: g.Game):
        """
        Reroll or not.
        """
        return m.Action(t.ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def end_game(self, game: g.Game):
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


def block_favourability(block_result: m.ActionType, active_player: m.Player, attacker: m.Player, defender: m.Player) -> int:

    if attacker.team == active_player.team:
        if block_result == t.ActionType.SELECT_DEFENDER_DOWN: return 6
        elif block_result == t.ActionType.SELECT_DEFENDER_STUMBLES:
            if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE): return 4       # push back
            else: return 6
        elif block_result == t.ActionType.SELECT_PUSH:
                return 4
        elif block_result == t.ActionType.SELECT_BOTH_DOWN:
            if defender.has_skill(t.Skill.BLOCK) and not attacker.has_skill(t.Skill.BLOCK): return 1        # skull
            elif not attacker.has_skill(t.Skill.BLOCK): return 2;                                           # both down
            elif attacker.has_skill(t.Skill.BLOCK) and defender.has_skill(t.Skill.BLOCK): return 3          # nothing happens
            else: return 5                                                                                  # only defender is down
        elif block_result == t.ActionType.SELECT_ATTACKER_DOWN:
            return 1                                                                                        # skull
    else:
        if block_result == t.ActionType.SELECT_DEFENDER_DOWN:
            return 1                                                                                        # least favourable
        elif block_result == t.ActionType.SELECT_DEFENDER_STUMBLES:
            if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE): return 3       # not going down, so I like this.
            else: return 1                                                                                  # splat.  No good.
        elif block_result == t.ActionType.SELECT_PUSH:
            return 3
        elif block_result == t.ActionType.SELECT_BOTH_DOWN:
            if not attacker.has_skill(t.Skill.BLOCK) and defender.has_skill(t.Skill.BLOCK): return 6        # Attacker down, I am not.
            if not attacker.has_skill(t.Skill.BLOCK) and not defender.has_skill(t.Skill.BLOCK): return 5    # Both down is pretty good.
            if attacker.has_skill(t.Skill.BLOCK) and not defender.has_skill(t.Skill.BLOCK): return 2        # Just I splat
            else: return 4                                                                                  # Nothing happens (both have block).
        elif block_result == t.ActionType.SELECT_ATTACKER_DOWN:
            return 6                                                                                        # most favourable!

    return 0


def potential_end_turn_action(game: g.Game) -> List[ActionSequence]:
    actions: List[ActionSequence] = []
    action_steps: List[m.Action] = []
    action_steps.append(m.Action(t.ActionType.END_TURN))
    # End turn happens on a score of 1.0.  Any actions with a lower score are never selected.
    actions.append(ActionSequence(action_steps, score=1.0, description='End Turn'))
    return actions


def potential_block_actions(game: g.Game, player: m.Player) -> List[ActionSequence]:

    # Note to self: need a "stand up and end move option.
    move_actions: List[ActionSequence] = []
    if not player.state.up:
        # There is currently a bug in the controlling logic.  Prone players shouldn't be able to block
        return move_actions
    blockable_players: List[m.Player] = game.state.pitch.adjacent_players(player, include_own=False, include_opp=True, manhattan=False, only_blockable=True, only_foulable=False)
    for blockable_player in blockable_players:
        action_steps: List[m.Action] = []
        action_steps.append(m.Action(t.ActionType.START_BLOCK, player=player))
        action_steps.append(m.Action(t.ActionType.BLOCK, pos=blockable_player.position, player=player))
        action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))

        action_score = score_block(game, player, blockable_player)
        score = action_score

        move_actions.append(ActionSequence(action_steps, score=score, description='Block ' + player.name + ' to (' + str(blockable_player.position.x) + ',' + str(blockable_player.position.y) + ')'))
        # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_blitz_actions(game: g.Game, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: m.Square = game.state.pitch.get_square(path.steps[-1].x, path.steps[-1].y)
        blockable_squares = game.state.pitch.adjacent_player_squares_at(player, end_square, include_own=False, include_opp=True, manhattan=False, only_blockable=True, only_foulable=False)
        for blockable_square in blockable_squares:
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(t.ActionType.START_BLITZ, player=player))
            if not player.state.up:
                action_steps.append(m.Action(t.ActionType.STAND_UP, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(m.Action(t.ActionType.MOVE, pos=game.state.pitch.get_square(step.x, step.y), player=player))
            action_steps.append(m.Action(t.ActionType.BLOCK, pos=blockable_square, player=player))
            action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))

            action_score = score_blitz(game, player, end_square, game.state.pitch.get_player_at(blockable_square))
            path_score = path_cost_to_score(game, path) # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Blitz ' + player.name + ' to ' + str(blockable_square.x) + ',' + str(blockable_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_pass_actions(game: g.Game, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: m.Square = game.state.pitch.get_square(path.steps[-1].x, path.steps[-1].y)
        # Need possible receving players
        to_squares, distances = game.state.pitch.passes_at(player, game.state.weather, end_square)
        for to_square in to_squares:
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(t.ActionType.START_PASS, player=player))

            receiver: Optional[m.Player, None] = game.state.pitch.get_player_at(to_square)

            if not player.state.up:
                action_steps.append(m.Action(t.ActionType.STAND_UP, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(m.Action(t.ActionType.MOVE, pos=game.state.pitch.get_square(step.x, step.y), player=player))
            action_steps.append(m.Action(t.ActionType.PASS, pos=to_square, player=player))
            action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))

            action_score = score_pass(game, player, end_square, to_square)
            path_score = path_cost_to_score(game, path) # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Pass ' + player.name + ' to ' + str(to_square.x) + ',' + str(to_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_handoff_actions(game: g.Game, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: m.Square = game.state.pitch.get_square(path.steps[-1].x, path.steps[-1].y)
        handoffable_squares = game.state.pitch.adjacent_player_squares_at(player, end_square, include_own=True, include_opp=False, manhattan=False, only_blockable=True, only_foulable=False)
        for handoffable_square in handoffable_squares:
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(t.ActionType.START_HANDOFF, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(m.Action(t.ActionType.MOVE, pos=game.state.pitch.get_square(step.x, step.y), player=player))
            action_steps.append(m.Action(t.ActionType.HANDOFF, pos=handoffable_square, player=player))
            action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))

            action_score = score_handoff(game, player, game.state.pitch.get_player_at(handoffable_square), end_square)
            path_score = path_cost_to_score(game, path) # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Handoff ' + player.name + ' to ' + str(handoffable_square.x) + ',' + str(handoffable_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_foul_actions(game: g.Game, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: m.Square = game.state.pitch.get_square(path.steps[-1].x, path.steps[-1].y)
        foulable_squares = game.state.pitch.adjacent_player_squares_at(player, end_square, include_own=False, include_opp=True, manhattan=False, only_blockable=False, only_foulable=True)
        for foulable_square in foulable_squares:
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(t.ActionType.START_FOUL, player=player))
            if not player.state.up:
                action_steps.append(m.Action(t.ActionType.STAND_UP, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(m.Action(t.ActionType.MOVE, pos=game.state.pitch.get_square(step.x, step.y)))
            action_steps.append(m.Action(t.ActionType.FOUL, foulable_square, player=player))
            action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))

            action_score = score_foul(game, player, game.state.pitch.get_player_at(foulable_square), end_square)
            path_score = path_cost_to_score(game, path) # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Foul ' + player.name + ' to ' + str(foulable_square.x) + ',' + str(foulable_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_move_actions(game: g.Game, player: m.Player, paths: List[pf.Path]) -> List[ActionSequence]:
    ''' Return set of all scored possible "MOVE" actions for given player

    :param player:
    :param game:
    :param paths:
    :return:
    '''
    move_actions: List[ActionSequence] = []
    ball_square: m.Square = game.state.pitch.get_ball_position()
    for path in paths:
        path_steps = path.steps
        action_steps: List[m.Action] = []
        action_steps.append(m.Action(t.ActionType.START_MOVE, player=player))
        if not player.state.up:
            action_steps.append(m.Action(t.ActionType.STAND_UP, player=player))
        for step in path_steps:
            # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
            action_steps.append(m.Action(t.ActionType.MOVE, pos=game.state.pitch.get_square(step.x, step.y), player=player))
        action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))

        to_square: m.Square = game.state.pitch.get_square(path_steps[-1].x, path_steps[-1].y)
        if to_square == ball_square:
            x=1
        action_score, description = score_move(game, player, to_square)
        path_score = path_cost_to_score(game, path)  # If an extra GFI required for block, should increase here.  To do.
        score = action_score + path_score

        move_actions.append(ActionSequence(action_steps, score=score, description='Move: ' + description + ' ' + player.name + ' to ' + str(path_steps[-1].x) + ',' + str(path_steps[-1].y)))
        # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def score_blitz(game: g.Game, attacker: m.Player, block_from_square: m.Square, defender: m.Player) -> float:
    score: float = GrodBot.BASE_SCORE_BLITZ
    num_block_dice: int = game.state.pitch.num_block_dice_at(attacker, defender, block_from_square, blitz=True, dauntless_success=False)
    ball_position: m.Player = game.state.pitch.get_ball_position()
    if num_block_dice == 3: score += 30.0
    if num_block_dice == 2: score += 10.0
    if num_block_dice == 1: score += -30.0
    if num_block_dice == -2: score += -75.0
    if num_block_dice == -3: score += -100.0
    if attacker.has_skill(t.Skill.BLOCK): score += 20.0
    if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE): score -= 10.0
    if defender.has_skill(t.Skill.BLOCK): score += -10.0
    if ball_position == attacker.position:
        if attacker.position.is_adjacent(defender.position) and block_from_square == attacker.position:
            score += 20.0
        else:
            score += -40.0
    if defender.position == ball_position: score += 50.0              # Blitzing ball carrier
    if defender.position.is_adjacent(ball_position): score += 20.0    # Blitzing someone adjacent to ball carrier
    if helper.direct_surf_squares(game, block_from_square, defender.position): score += 25.0 # A surf
    score -= len(game.state.pitch.adjacent_players_at(attacker, defender.position, include_own=False, include_opp=True)) * 5.0
    if attacker.position == block_from_square: score -= 20.0      # A Blitz where the block is the starting square is unattractive
    if helper.in_scoring_range(game, defender): score+=10.0       # Blitzing players closer to the endzone is attractive
    return score


def score_foul(game: g.Game, attacker: m.Player, defender: m.Player, to_square : m.Square) -> float:
    score = GrodBot.BASE_SCORE_FOUL
    ball_carrier: Optional[m.Player, None] = game.state.pitch.get_ball_carrier()

    if ball_carrier == attacker: score = score - 30.0
    if attacker.has_skill(t.Skill.DIRTY_PLAYER): score = score + 10.0
    if attacker.has_skill(t.Skill.SNEAKY_GIT): score = score + 10.0
    if defender.state.stunned: score = score - 15.0

    assists_for, assists_against = game.state.pitch.num_assists_at(attacker, defender, to_square, ignore_guard=True)
    score = score + (assists_for-assists_against) * 15.0

    if attacker.team.state.bribes>0: score += 40.0
    if attacker.has_skill(t.Skill.CHAINSAW): score += 30.0
    #TVdiff = defender.GetBaseTV() - attacker.GetBaseTV()
    TVdiff = 10.0
    score = score + TVdiff

    return score


def score_move(game: g.Game, player: m.Player, to_square: m.Square) -> (float, str):

    scores: List[(float, str)] = []

    scores.append((score_receiving_position(game, player, to_square), 'move to receiver'))
    scores.append((score_move_towards_ball(game, player, to_square), 'move toward ball'))
    scores.append((score_move_to_ball(game, player, to_square), 'move to ball'))
    scores.append((score_move_ball(game, player, to_square), 'move ball'))
    scores.append((score_sweep(game, player, to_square), 'move to sweep'))
    scores.append((score_defensive_screen(game, player, to_square), 'move to defensive screen'))
    scores.append((score_offensive_screen(game, player, to_square), 'move to offsensive screen'))
    scores.append((score_caging(game, player, to_square), 'move to cage'))
    scores.append((score_mark_opponent(game, player, to_square), 'move to mark opponent'))

    scores.sort(key=lambda tup: tup[0], reverse=True)
    score, description = scores[0]

    # All moves should avoid the sideline
    if helper.distance_to_sideline(game, to_square) == 0: score -= 20.0
    if helper.distance_to_sideline(game, to_square) == 1: score -= 10.0

    return (score, description)


def score_receiving_position(game: g.Game, player: m.Player, to_square : m.Square) -> float:
    if player.team != game.state.pitch.get_ball_team() or player == game.state.pitch.get_ball_carrier(): return 0.0

    receivingness = player_receiver_ability(game, player)
    score = receivingness - 30.0
    if helper.in_scoring_endzone(game, player.team, to_square):
        numInRange = len(helper.players_in_scoring_endzone(game, player.team, include_own=True, include_opp=False))
        if player.team.state.turn == 8: score += 40   # Pretty damned urgent to get to end zone!
        score -= numInRange * numInRange * 40  # Don't want too many catchers in the endzone ...

    score += 5.0 * (max(helper.distance_to_scoring_endzone(game, player.team, player.position), player.get_ma()) - max(helper.distance_to_scoring_endzone(game, player.team, to_square), player.get_ma()))
    # Above score doesn't push players to go closer than their MA from the endzone.

    if helper.distance_to_scoring_endzone(game, player.team, to_square) > player.get_ma() + 2: score -= 30.0
    opps: List[m.Player] = game.state.pitch.adjacent_players_at(player, to_square, include_own=False, include_opp=True, include_stunned=False)
    if opps: score -= 40.0 + 20.0 * len(opps)
    score -= 10.0 * len(game.state.pitch.adjacent_players_at(player, to_square, include_own=False, include_opp=True))
    numInRange = len(helper.players_in_scoring_distance(game, player.team, include_own=True, include_opp=False))
    score -= numInRange * numInRange * 20.0     # Lower the score if we already have some receivers.
    if helper.contains_a_player(game, player.team, helper.squares_within(game, to_square, 2), include_opp=False, include_own=True): score -= 20.0

    return score


def score_move_towards_ball(game: g.Game, player: m.Player, to_square : m.Square) -> float:
    ball_square: m.Square = game.state.pitch.get_ball_position()
    ball_carrier = game.state.pitch.get_ball_carrier()

    if (player == ball_carrier) or (ball_square is None) or (to_square == ball_square): return 0.0

    score = GrodBot.BASE_SCORE_MOVE_TOWARD_BALL
    if ball_carrier is None: score += 20.0
    distance_to_ball = ball_square.distance(to_square)

    # Cancel the penalty for being near the sideline if the ball is on the sideline
    if helper.distance_to_sideline(game, ball_square) <= 1:
        if helper.distance_to_sideline(game, to_square): score += 10.0

    #if player.get_ma() < distance_to_ball + 3:
    score -= distance_to_ball*2

    return score


def score_move_to_ball(game: g.Game, player: m.Player, to_square: m.Square) -> float:
    ball_square: m.Square = game.state.pitch.get_ball_position()
    ball_carrier = game.state.pitch.get_ball_carrier()
    if ball_square is None: return -20.0
    if ball_carrier == player: return 0.0
    if ball_square != to_square : return 0.0
    if ball_carrier is not None: return 0.0


    score = GrodBot.BASE_SCORE_MOVE_TO_BALL
    if player.has_skill(t.Skill.SURE_HANDS) or not player.team.state.reroll_used: score += 15.0
    if player.get_ag() < 2: score += -10.0
    if player.get_ag() == 3: score += 5.0
    if player.get_ag() > 3: score += 10.0
    num_tz = game.state.pitch.num_tackle_zones_at(player, ball_square)
    score += - 10 * num_tz    # Lower score if lots of tackle zones on ball.

    # Cancel the penalty for being near the sideline if the ball is on/near the sideline
    if helper.distance_to_sideline(game, ball_square) == 1: score+=10.0
    if helper.distance_to_sideline(game, ball_square) == 0: score+=20.0

    return score


def score_move_ball(game: g.Game, player: m.Player, to_square: m.Square) -> float:
    ball_square: m.Square = game.state.pitch.get_ball_position()
    ball_carrier = game.state.pitch.get_ball_carrier()
    if not player == ball_carrier: return 0.0

    score = GrodBot.BASE_SCORE_MOVE_BALL
    if helper.in_scoring_endzone(game, player.team, to_square): score += 30.0
    elif player.team.state.turn == 8: score -= 70.0
    score -= 5.0 * (helper.distance_to_scoring_endzone(game, player.team, to_square) - helper.distance_to_scoring_endzone(game, player.team, player.position))
    opps: List[m.Player] = game.state.pitch.adjacent_players_at(player, to_square, include_own=False, include_opp=True, include_stunned=False)
    if opps: score -= 40.0  + 20.0  * len(opps)
    if helper.contains_a_player(game, player.team, helper.squares_within(game, to_square, 2), include_opp=True, include_own=False, include_stunned=False): score -= 10.0
    if not helper.blitz_used(game): score -= 50.0
    #score += heatMap.GetBallMoveSquareSafetyScore(to_square)
    return score


def score_sweep(game: g.Game, player: m.Player, to_square: m.Square) -> float:
    if game.state.pitch.get_ball_team() != player.team: return 0.0  # Don't sweep unless the other team has the ball
    if helper.distance_to_defending_endzone(game, player.team, game.state.pitch.get_ball_position()) < 9: return 0.0  # Don't sweep when the ball is close to the endzone
    if helper.players_in_scoring_distance(game, player.team, include_own=False, include_opp=True): return 0.0  # Don't sweep when there are opponent units in scoring range

    score = GrodBot.BASE_SCORE_MOVE_TO_SWEEP
    blitziness = player_blitz_ability(game, player)
    score += blitziness - 60.0
    score -= 30.0 * len(game.state.pitch.adjacent_player_squares(player, include_own=False, only_blockable=True))

    # Now to evaluate ideal square for Sweeping:

    x_preferred = int(helper.reverse_x_for_left(game, player.team, (game.state.pitch.width-2) / 4))
    y_preferred = int((game.state.pitch.height-2) / 2)
    score -= abs(y_preferred - to_square .y) * 10.0

    # subtract 5 points for every square away from the preferred sweep location.
    score -= abs(x_preferred - to_square .x) * 5.0

    # Check if a player is already sweeping:
    for i in range(-2,3):
        for j in range(-2,3):
            cur: m.Square = game.state.pitch.get_square(x_preferred + i, y_preferred + j)
            player: Optional[m.Player, None] = game.state.pitch.get_player_at(cur)
            if player is not None and player.team == player.team: score -= 90.0

    return score


def score_defensive_screen(game: g.Game, player: m.Player, to_square: m.Square) -> float:
    if game.state.pitch.get_ball_team() is None or game.state.pitch.get_ball_team() == player.team: return 0.0
    ''' This one is a bit trickier by nature, because it involves combinations of two or more
     * players...
     *
     * ok this is actaully rather simple:
      * Increase score if square is close to ball carrier.
      * Decrease if far away.
      * Decrease if square is behind ball carrier.
      * Increase slightly if square is 1 away from sideline.
      * Decrease if close to a player on the same team WHO IS ALREADY screening.
      * Increase slightly if most of the players movement must be used to arrive at the screening square.
      *
      * This isn't working yet... Perhaps I need a list of all nearby players to understand square.
     '''
    score = GrodBot.BASE_SCORE_DEFENSIVE_SCREEN

    ball_carrier = game.state.pitch.get_ball_carrier()
    ball_square = game.state.pitch.get_ball_position()
    distanceBallCarrierToEnd = helper.distance_to_defending_endzone(game, player.team, ball_square)
    distanceSquareToEnd = helper.distance_to_defending_endzone(game, player.team, to_square )

    if distanceSquareToEnd + 1.0 < distanceBallCarrierToEnd: score += 30.0  # Increase score defending on correct side of field.

    distance_to_ball = ball_square.distance(to_square)
    score += 4.0*max(5.0 - distance_to_ball, 0.0) # Increase score defending in front of ball carrier
    score += distanceSquareToEnd/10.0  # Increase score a small amount to screen closer to opponents.
    distance_to_closest_opponent = helper.distance_to_nearest_player(game, player.team, to_square, include_own=False, include_opp=True, include_stunned=False)
    if distance_to_closest_opponent <= 1.5: score -= 30.0
    elif distance_to_closest_opponent <= 2.95: score += 10.0
    elif distance_to_closest_opponent > 2.95: score += 5.0
    if helper.distance_to_sideline(game, to_square) == 1: score += 10.0  # Cancel the negative score of being 1 from sideline.
                        # I should make it a tiny bit more attractive if ball is on this side of field.  Lets do in a new update.
    distance_to_closest_friendly_used = helper.distance_to_nearest_player(game, player.team, to_square, include_own=True, include_opp=False, only_used=True)
    if distance_to_closest_friendly_used >= 4: score += 2.0
    elif distance_to_closest_friendly_used >= 3: score += 40.0  # Increase score if the square links with another friendly (hopefully also screening)
    elif distance_to_closest_friendly_used > 2: score += 10.0   # Descrease score if very close to another defender
    else: score -= 10.0  # Decrease score if too close to another defender.

    distance_to_closest_friendly_unused = helper.distance_to_nearest_player(game, player.team, to_square, include_own=True, include_opp=False, include_used=True)
    if distance_to_closest_friendly_unused >= 4: score += 3.0
    elif distance_to_closest_friendly_unused >= 3: score += 8.0 # Increase score if the square links with another friendly (hopefully also screening)
    elif distance_to_closest_friendly_unused > 2: score += 3.0  # Descrease score if very close to another defender
    else: score -= 10.0  # Decrease score if too close to another defender.

    return score


def score_offensive_screen(game: g.Game, player: m.Player, to_square: m.Square) -> float:
    ''' Another subtle one.  Basically if the ball carrier "breaks out", I want to screen him from
     * behind, rather than cage him.  I may even want to do this with an important receiver.
     *
     * Want my players to be 2 squares from each other, not counting direct diagonals.
     * want my players to be hampering the movement of opponent ball or players.  Ideally
     * want my players in a line between goal line and opponent.
     *
     '''

    ball_carrier: m.Player = game.state.pitch.get_ball_carrier()
    ball_square: m.Player = game.state.pitch.get_ball_position()
    if ball_carrier is None or ball_carrier.team == player.team: return 0.0

    score = 0.0

    return score


def score_caging(game: g.Game, player: m.Player, to_square: m.Square) -> float:
    ball_carrier: m.Player = game.state.pitch.get_ball_carrier()
    if ball_carrier is None or ball_carrier.team != player.team or ball_carrier == player: return 0.0          # Noone has the ball.  Don't try to cage.
    ball_square: m.Square = game.state.pitch.get_ball_position()

    cage_square_groups: List[List[m.Square]] = []
    cage_square_groups.append(helper.caging_squares_north_east(game, ball_square))
    cage_square_groups.append(helper.caging_squares_north_west(game, ball_square))
    cage_square_groups.append(helper.caging_squares_south_east(game, ball_square))
    cage_square_groups.append(helper.caging_squares_south_west(game, ball_square))

    distOppToBall = helper.distance_to_nearest_player(game, player.team, ball_square, include_own=False, include_opp=True, include_stunned=False)
    avgMAopps = average_ma(game, helper.get_players(game, player.team, include_own=False, include_opp=True, include_stunned=False))
    score = 0.0

    for curGroup in cage_square_groups:
        if not helper.contains_a_player(game, player.team, curGroup, include_opp=False, include_own=True, only_blockable=True):
            if to_square in curGroup: score += GrodBot.BASE_SCORE_CAGE_BALL
            dist = helper.distance_to_nearest_player(game, player.team, to_square, include_own=False, include_stunned=False, include_opp=True)
            score += distOppToBall - dist
            if distOppToBall > avgMAopps: score -= 30.0
            if not ball_carrier.state.used: score -= 30.0
            if to_square.is_adjacent(game.state.pitch.get_ball_position()): score += 5
            if helper.is_bishop_position_of(game, player, ball_carrier): score -= 2
            #score += heatMap.GetCageNecessityScore(to_square )

    if not player.state.up: score += 5.0
    if not ball_carrier.state.used: score = max(0.0, score - GrodBot.BASE_SCORE_CAGE_BALL)
    return score


def score_mark_opponent(game: g.Game, player: m.Player, to_square: m.Square) -> float:

    ball_carrier: m.Player = game.state.pitch.get_ball_carrier()
    team_with_ball = game.state.pitch.get_ball_team()
    ball_square = game.state.pitch.get_ball_position()
    if ball_carrier == player: return 0.0
    allOpps: List[m.Player] = game.state.pitch.adjacent_players_at(player, to_square, include_opp=True, include_own=False)
    if not allOpps: return 0.0

    score = GrodBot.BASE_SCORE_MOVE_TO_OPPONENT
    if to_square .is_adjacent(game.state.pitch.get_ball_position()):
        if team_with_ball == player.team: score += 20.0
        else: score += 30.0

    for opp in allOpps:
        if helper.distance_to_scoring_endzone(game, opp.team, to_square ) < opp.get_ma() + 2:
            score += 10.0  # Mark opponents in scoring range first.
            break         # Only add score once.

    if len(allOpps) == 1:
        score += 20.0
        numFriendlyNextTo = game.state.pitch.num_tackle_zones_in(allOpps[0])
        if allOpps[0].state.up:
            if numFriendlyNextTo == 1: score += 5.0
            else: score -= 10.0 * numFriendlyNextTo

        if not allOpps[0].state.up:
            if numFriendlyNextTo == 0: score += 5.0
            else: score -= 10.0 * numFriendlyNextTo  # Unless we want to start fouling ...

    if not player.state.up: score += 25.0
    if not player.has_skill(t.Skill.GUARD): score -= len(allOpps) * 10.0
    else: score += len(allOpps) * 10.0

    ballIsNear = False
    for curOpp in allOpps:
        if curOpp.position.is_adjacent(game.state.pitch.get_ball_position()):
            ballIsNear = True

    if ballIsNear: score += 8.0
    if player.position != to_square  and game.state.pitch.num_tackle_zones_in(player)>0:
        score -= 40.0

    if ball_square is not None:
        distance_to_ball = ball_square.distance(to_square)
        score -= distance_to_ball / 5.0   # Mark opponents closer to ball when possible

    if team_with_ball is not None and team_with_ball != player.team:
        distToOtherEndzone = helper.distance_to_scoring_endzone(game, player.team, to_square )
        score += distToOtherEndzone / 5.0   # if opponent has ball, mark closer to defending zone.
        # This way there is a preference for most advanced (distance wise) units.
    return score


def score_handoff(game: g.Game, ball_carrier: m.Player, receiver: m.Player, from_square: m.Square) -> float:
    if receiver == ball_carrier: return 0.0

    score = GrodBot.BASE_SCORE_HANDOFF
    score += probability_fail_to_score(probability_catch_fail(game, receiver))
    if not ball_carrier.team.state.reroll_used: score += +10.0
    score -= 5.0 * (helper.distance_to_scoring_endzone(game, ball_carrier.team, receiver.position) - helper.distance_to_scoring_endzone(game, ball_carrier.team, ball_carrier.position))
    if receiver.state.used: score -= 30.0
    if (game.state.pitch.num_tackle_zones_in(ball_carrier) > 0 or game.state.pitch.num_tackle_zones_in(receiver) > 0) and not helper.blitz_used(game): score -= 50.0 # Don't try a risky hand-off if we haven't blitzed yet
    if helper.in_scoring_range(game, receiver): score += 40.0
    # score += heatMap.GetBallMoveSquareSafetyScore(to_square)
    return score


def score_pass(game: g.Game, passer: m.Player, from_square: m.Square, to_square: m.Square) -> float:

    receiver = game.state.pitch.get_player_at(to_square)

    if receiver is None: return -50.0
    if receiver.team != passer.team: return -200.0
    if receiver == passer: return -100.0

    score = 40.0
    score += probability_fail_to_score(probability_catch_fail(game, receiver))
    dist: t.PassDistance = game.state.pitch.pass_distance(from_square, receiver.position)
    score += probability_fail_to_score(probability_pass_fail(game, passer, from_square, dist))
    if not passer.team.state.reroll_used: score += +10.0
    score = score - 5.0 * (helper.distance_to_scoring_endzone(game, receiver.team, receiver.position) - helper.distance_to_scoring_endzone(game, passer.team, passer.position))
    if receiver.state.used: score -= 30.0
    if game.state.pitch.num_tackle_zones_in(passer) > 0 or game.state.pitch.num_tackle_zones_in(receiver) > 0  and not helper.blitz_used(game): score -= 50.0
    if helper.in_scoring_range(game, receiver): score+=40.0
    return score


def score_block(game: g.Game, attacker: m.Player, defender: m.Player) -> float:
    score = GrodBot.BASE_SCORE_BLOCK
    ball_carrier = game.state.pitch.get_ball_carrier()
    ball_square = game.state.pitch.get_ball_position()
    if attacker.has_skill(t.Skill.CHAINSAW):
        score += 15.0
        score += 20.0 - 2 * defender.get_av()
        # Add something in case the defender is really valuable?
    else:
        num_block_dice = game.state.pitch.num_block_dice(attacker, defender)
        if num_block_dice == 3: score += 15.0
        if num_block_dice == 2: score += 0.0
        if num_block_dice == 1: score += -66.0  # score is close to zero.
        if num_block_dice == -2: score += -95.0
        if num_block_dice == -3: score += -150.0

        if not attacker.team.state.reroll_used and not attacker.has_skill(t.Skill.LONER): score += 10.0
        if attacker.has_skill(t.Skill.BLOCK) or attacker.has_skill(t.Skill.WRESTLE): score += 20.0
        if defender.has_skill(t.Skill.DODGE) and not attacker.has_skill(t.Skill.TACKLE): score += -10.0
        if defender.has_skill(t.Skill.BLOCK): score += -10.0
        if helper.attacker_would_surf(game, attacker, defender): score += 32.0
        if attacker.has_skill(t.Skill.LONER): score -= 10.0

    if attacker == ball_carrier: score += -45.0
    if defender == ball_carrier: score += 35.0
    if defender.position.is_adjacent(ball_square): score += 15.0

    return score


def score_push(game: g.Game, from_square: m.Square, to_square: m.Square) -> int:
    score = 0.0
    ball_square = game.state.pitch.get_ball_position()
    if helper.distance_to_sideline(game, to_square) == 0: score = score + 10.0    # Push towards sideline
    if ball_square is not None and to_square .is_adjacent(ball_square): score = score - 15.0    # Push away from ball
    if helper.direct_surf_squares(game, from_square, to_square): score = score + 10.0
    return score


def scoring_urgency_score (player: m.Player) -> float:
    if player.team.state.turn == 8: return 40
    return 0


def path_cost_to_score(game: g.Game, path: pf.Path) -> float:
    cost: float = path.cost

    #assert 0 <= cost <= 1

    score = -(cost * cost * (250.0 + GrodBot.ADDITIONAL_SCORE_DODGE))
    return score


def probability_fail_to_score(probability: float) -> float:
    score = -(probability * probability * (250.0 + GrodBot.ADDITIONAL_SCORE_DODGE))
    return score


def probability_catch_fail (game: g.Game, receiver: m.Player) -> float:
    num_tz = 0.0
    if not receiver.has_skill(t.Skill.NERVES_OF_STEEL): num_tz = game.state.pitch.num_tackle_zones_in(receiver)
    probSuccess = min(5.0, receiver.get_ag()+1.0-num_tz)/6.0
    if receiver.has_skill(t.Skill.CATCH): probSuccess += (1.0-probSuccess)*probSuccess
    probability = 1.0 - probSuccess
    return probability


def probability_pass_fail (game: g.Game, passer: m.Player, from_square: m.Square, dist: t.PassDistance) -> float:
    nTZ = 0.0
    if not passer.has_skill(t.Skill.NERVES_OF_STEEL): nTZ = game.state.pitch.num_tackle_zones_at(passer, from_square)
    probSuccess = 0.0
    if passer.has_skill(t.Skill.ACCURATE): nTZ -= 1
    if passer.has_skill(t.Skill.STRONG_ARM and dist!=t.PassDistance.QUICK_PASS): nTZ -= 1
    if dist==t.PassDistance.HAIL_MARY: return -100.0
    if dist==t.PassDistance.QUICK_PASS: nTZ -= 1
    if dist==t.PassDistance.SHORT_PASS: nTZ -= 0
    if dist==t.PassDistance.LONG_PASS: nTZ += 1
    if dist==t.PassDistance.LONG_BOMB: nTZ += 2
    probSuccess = min(5.0, passer.get_ag()-nTZ)/6.0
    if passer.has_skill(t.Skill.PASS): probSuccess += (1.0-probSuccess)*probSuccess
    probability = 1.0 - probSuccess
    return probability


def choose_gaze_victim(game: g.Game, player: m.Player) -> m.Player:
    bestUnit: m.Player = None
    bestScore = 0.0
    ball_square: m.Square = game.state.pitch.get_ball_position()
    potentials: List[m.Player] = game.state.pitch.adjacent_players(player, include_own=False, include_opp=True, only_blockable=True)
    for unit in potentials:
        curScore = 5.0
        curScore += 6.0 - unit.get_ag()
        if unit.position.is_adjacent(ball_square): curScore += 5.0
        if curScore > bestScore:
            bestScore = curScore
            bestUnit = unit
    return bestUnit


def average_st(game: g.Game, players: List[m.Player]) -> float:
    avSt = 0.0
    num = 0.0
    for player in players:
        avSt = avSt + player.get_st()
        num += 1.0
    avSt = avSt / num
    return avSt


def average_av(game: g.Game, players: List[m.Player]) -> float:
    avAV = 0.0
    num = 0.0
    for player in players:
        avAV = avAV + player.get_av()
        num += 1.0
    avAV = avAV / num
    return avAV


def average_ma(game: g.Game, players: List[m.Player]) -> float:
    avMA = 0.0
    num = 0.0
    for player in players:
        avMA = avMA + player.get_ma()
        num += 1.0
    avMA = avMA / num
    return avMA


def player_bash_ability(game: g.Game, player: m.Player) -> float:
    bashiness: float = 0.0
    bashiness += 10.0 * player.get_st()
    bashiness += 5.0 * player.get_av()
    if player.has_skill(t.Skill.BLOCK): bashiness += 10.0
    if player.has_skill(t.Skill.WRESTLE): bashiness += 10.0
    if player.has_skill(t.Skill.MIGHTY_BLOW): bashiness += 5.0
    if player.has_skill(t.Skill.CLAWS): bashiness += 5.0
    if player.has_skill(t.Skill.PILING_ON): bashiness += 5.0
    if player.has_skill(t.Skill.GUARD): bashiness += 15.0
    if player.has_skill(t.Skill.DAUNTLESS): bashiness += 10.0
    if player.has_skill(t.Skill.FOUL_APPEARANCE): bashiness += 5.0
    if player.has_skill(t.Skill.TENTACLES): bashiness += 5.0
    if player.has_skill(t.Skill.STUNTY): bashiness -= 10.0
    if player.has_skill(t.Skill.REGENERATION): bashiness += 10.0
    if player.has_skill(t.Skill.THICK_SKULL): bashiness += 3.0
    return bashiness


def team_bash_ability(game: g.Game, players: List[m.Player]) -> float:
    sum = 0.0
    for player in players: sum += player_bash_ability(game, player)
    return sum


def player_pass_ability(game: g.Game, player: m.Player) -> float:
    passingAbility = 0.0
    passingAbility += player.get_ag() * 15.0    # Agility most important.
    passingAbility += player.get_ma() * 2.0     # Fast movements make better ball throwers.
    if player.has_skill(t.Skill.PASS): passingAbility += 10.0
    if player.has_skill(t.Skill.SURE_HANDS): passingAbility += 5.0
    if player.has_skill(t.Skill.EXTRA_ARMS): passingAbility += 3.0
    if player.has_skill(t.Skill.NERVES_OF_STEEL): passingAbility += 3.0
    if player.has_skill(t.Skill.ACCURATE): passingAbility += 5.0
    if player.has_skill(t.Skill.STRONG_ARM): passingAbility += 5.0
    if player.has_skill(t.Skill.BONE_HEAD): passingAbility -= 15.0
    if player.has_skill(t.Skill.REALLY_STUPID): passingAbility -= 15.0
    if player.has_skill(t.Skill.WILD_ANIMAL): passingAbility -= 15.0
    if player.has_skill(t.Skill.ANIMOSITY): passingAbility -= 10.0
    if player.has_skill(t.Skill.LONER): passingAbility -= 15.0
    if player.has_skill(t.Skill.DUMP_OFF): passingAbility += 5.0
    if player.has_skill(t.Skill.SAFE_THROW): passingAbility += 5.0
    if player.has_skill(t.Skill.NO_HANDS): passingAbility -= 100.0
    return passingAbility


def player_blitz_ability(game: g.Game, player: m.Player) -> float:
    blitzingAbility = player_bash_ability(game, player)
    blitzingAbility += player.get_ma() * 10.0
    if player.has_skill(t.Skill.TACKLE): blitzingAbility += 5.0
    if player.has_skill(t.Skill.SPRINT): blitzingAbility += 5.0
    if player.has_skill(t.Skill.SURE_FEET): blitzingAbility += 5.0
    if player.has_skill(t.Skill.STRIP_BALL): blitzingAbility += 5.0
    if player.has_skill(t.Skill.DIVING_TACKLE): blitzingAbility += 5.0
    if player.has_skill(t.Skill.MIGHTY_BLOW): blitzingAbility += 5.0
    if player.has_skill(t.Skill.CLAWS): blitzingAbility += 5.0
    if player.has_skill(t.Skill.PILING_ON): blitzingAbility += 5.0
    if player.has_skill(t.Skill.BONE_HEAD): blitzingAbility -= 15.0
    if player.has_skill(t.Skill.REALLY_STUPID): blitzingAbility -= 15.0
    if player.has_skill(t.Skill.WILD_ANIMAL): blitzingAbility -= 10.0
    if player.has_skill(t.Skill.LONER): blitzingAbility -= 15.0
    if player.has_skill(t.Skill.SIDE_STEP): blitzingAbility += 5.0
    if player.has_skill(t.Skill.JUMP_UP): blitzingAbility += 5.0
    if player.has_skill(t.Skill.HORNS): blitzingAbility += 10.0
    if player.has_skill(t.Skill.JUGGERNAUT): blitzingAbility += 10.0
    if player.has_skill(t.Skill.LEAP): blitzingAbility += 5.0
    return blitzingAbility


def player_receiver_ability(game: g.Game, player: m.Player) -> float:
    receivingAbility = 0.0
    receivingAbility += player.get_ma() * 5.0
    receivingAbility += player.get_ag() * 10.0
    if player.has_skill(t.Skill.CATCH): receivingAbility += 15.0
    if player.has_skill(t.Skill.EXTRA_ARMS): receivingAbility += 10.0
    if player.has_skill(t.Skill.NERVES_OF_STEEL): receivingAbility += 5.0
    if player.has_skill(t.Skill.DIVING_CATCH): receivingAbility += 5.0
    if player.has_skill(t.Skill.DODGE): receivingAbility += 10.0
    if player.has_skill(t.Skill.SIDE_STEP): receivingAbility += 5.0
    if player.has_skill(t.Skill.BONE_HEAD): receivingAbility -= 15.0
    if player.has_skill(t.Skill.REALLY_STUPID): receivingAbility -= 15.0
    if player.has_skill(t.Skill.WILD_ANIMAL): receivingAbility -= 15.0
    if player.has_skill(t.Skill.LONER): receivingAbility -= 15.0
    if player.has_skill(t.Skill.NO_HANDS): receivingAbility -= 100.0
    return receivingAbility


def player_run_ability(game: g.Game, player: m.Player) -> float:
    runningAbility = 0.0
    runningAbility += player.get_ma() * 10.0    # Really favour fast units
    runningAbility += player.get_ag() * 10.0    # Agility to be prized
    runningAbility += player.get_st() * 5.0     # Doesn't hurt to be strong!
    if player.has_skill(t.Skill.SURE_HANDS): runningAbility += 10.0
    if player.has_skill(t.Skill.BLOCK): runningAbility += 10.0
    if player.has_skill(t.Skill.EXTRA_ARMS): runningAbility += 5.0
    if player.has_skill(t.Skill.DODGE): runningAbility += 10.0
    if player.has_skill(t.Skill.SIDE_STEP): runningAbility += 5.0
    if player.has_skill(t.Skill.STAND_FIRM): runningAbility += 3.0
    if player.has_skill(t.Skill.BONE_HEAD): runningAbility -= 15.0
    if player.has_skill(t.Skill.REALLY_STUPID): runningAbility -= 15.0
    if player.has_skill(t.Skill.WILD_ANIMAL): runningAbility -= 15.0
    if player.has_skill(t.Skill.LONER): runningAbility -= 15.0
    if player.has_skill(t.Skill.ANIMOSITY): runningAbility -= 5.0
    if player.has_skill(t.Skill.DUMP_OFF): runningAbility += 5.0
    if player.has_skill(t.Skill.NO_HANDS): runningAbility -= 100.0
    return runningAbility


def player_value(game: g.Game, player: m.Player) -> float:
    value = player.get_ag()*40 + player.get_av()*30 + player.get_ma()*30 + player.get_st()*50 + len(player.skills)*20
    return value

'''
def GetBlockFavourability(game: g.Game, BlockResult blockDice, attacker: m.Player, defender: m.Player) -> int:
    #gameInspector.Debug("GrodBot: getBlockFavourability")
    if (attacker.GetSide() == gameInspector.GetTeam().GetSide()) {
        switch (blockDice) {
            case DefenderDown:
                return 6           # pow
            case DefenderStumbles:
                if (defender.has_skill(t.Skill.Dodge) and not attacker.has_skill(t.Skill.Tackle)) return 4       # push back
                else return 6       # pow
            case Pushed:
                return 4
            case BothDown:
                if (defender.has_skill(t.Skill.Block) and not attacker.has_skill(t.Skill.Block)) return 1       # skull
                elif (not attacker.has_skill(t.Skill.Block)) return 2       # both down
                elif (attacker.has_skill(t.Skill.Block) and defender.has_skill(t.Skill.Block)) return 3       # nothing happens
                else return 5       # only defender is down
            case AttackerDown:
                return 1           # skull
        }
    } else {
        switch (blockDice) {
            case DefenderDown:
                return 1   # least favourable
            case DefenderStumbles:
                if (defender.has_skill(t.Skill.Dodge) and not attacker.has_skill(t.Skill.Tackle)) return 3  # not going down, so I like this.
                else return 1  # splat.  No good.
            case Pushed:
                return 3
            case BothDown:
                if (not attacker.has_skill(t.Skill.Block) and defender.has_skill(t.Skill.Block)) return 6   # Attacker down, I am not.
                if (not attacker.has_skill(t.Skill.Block) and not defender.has_skill(t.Skill.Block)) return 5   # Both down is pretty good.
                if (attacker.has_skill(t.Skill.Block) and not defender.has_skill(t.Skill.Block)) return 2   # Just I splat
                else  return 4   # Nothing happens (both have block).
            case AttackerDown:
                return 6   # most favourable!
        }
    }
    return 0
'''

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
        away_agent = GrodBot("GrodBot 1")
        home_agent = GrodBot("GrodBot 2")
        config.debug_mode = False
        game = api.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i + 1))
        start = time.time()
        game.init()
        game.step()
        end = time.time()
        print(end - start)
