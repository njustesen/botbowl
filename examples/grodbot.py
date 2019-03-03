from __future__ import annotations
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
from enum import Enum


class GrodBot(bot.ProcBot):

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.opp_team = None
        self.current_move: Optional[ActionSequence, None] = None

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

    def setup(self, game: g.Game):
        """
        Move players from the reserves to the pitch
        """
        i = len(game.get_players_on_pitch(self.my_team))
        reserves = game.get_reserves(self.my_team)
        if i == 11 or len(reserves) == 0:
            return m.Action(t.ActionType.END_SETUP)
        player = reserves[0]
        y = 3
        x = 13 if game.is_team_side(m.Square(13, 3), self.my_team) else 14
        return m.Action(t.ActionType.PLACE_PLAYER, player=player, pos=m.Square(x, y + i))

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
                if m.Skill.BLOCK in player.skills:
                    return m.Action(t.ActionType.PLACE_PLAYER, player=player, pos=ball_pos)
        return m.Action(t.ActionType.SELECT_NONE)

    def touchback(self, game: g.Game):
        """
        Select player to give the ball to.
        """
        for player in game.get_players_on_pitch(self.my_team, up=True):
            if m.Skill.BLOCK in player.skills:
                return m.Action(t.ActionType.SELECT_PLAYER, player=player)
        return m.Action(t.ActionType.SELECT_NONE)

    def set_next_move(self, game: g.Game):
        ''' Set self.current_move

        :param game:
        '''
        self.current_move = None
        all_actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            if action_choice.action_type == t.ActionType.START_MOVE:
                players_available: List[m.Player] = action_choice.players
                ff_map = pf.FfTileMap(game.state)
                for player in players_available:
                    player_square: m.Square = player.position
                    player_mover = pf.FfMover(player)
                    finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True, heuristic=pf.BruteForceHeuristic())
                    paths = finder.find_paths(player_mover, player_square.x, player_square.y)
                    all_actions.extend(potential_move_actions(player, game, paths))
            elif action_choice.action_type == t.ActionType.START_BLITZ:
                players_available: List[m.Player] = action_choice.players
                ff_map = pf.FfTileMap(game.state)
                for player in players_available:
                    player_square: m.Square = player.position
                    player_mover = pf.FfMover(player)
                    # Need 1 square of movement left to execute block after moving
                    finder = pf.AStarPathFinder(ff_map, player.move_allowed()-1, allow_diag_movement=True,heuristic=pf.BruteForceHeuristic())
                    paths = finder.find_paths(player_mover, player_square.x, player_square.y)
                    all_actions.extend(potential_blitz_actions(player, game, paths))
            elif action_choice.action_type == t.ActionType.START_FOUL:
                players_available: List[m.Player] = action_choice.players
                ff_map = pf.FfTileMap(game.state)
                for player in players_available:
                    player_square: m.Square = player.position
                    player_mover = pf.FfMover(player)
                    # Need 1 square of movement left to execute block after moving
                    finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True,heuristic=pf.BruteForceHeuristic())
                    paths = finder.find_paths(player_mover, player_square.x, player_square.y)
                    all_actions.extend(potential_foul_actions(player, game, paths))
            elif action_choice.action_type == t.ActionType.START_BLOCK:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    all_actions.extend(potential_block_actions(player, game))
            elif action_choice.action_type == t.ActionType.START_PASS:
                players_available: List[m.Player] = action_choice.players
                ff_map = pf.FfTileMap(game.state)
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.state.pitch.get_ball_position() == player_square:
                        player_mover = pf.FfMover(player)
                        # Need 1 square of movement left to execute block after moving
                        finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True,
                                                    heuristic=pf.BruteForceHeuristic())
                        paths = finder.find_paths(player_mover, player_square.x, player_square.y)
                        all_actions.extend(potential_pass_actions(player, game, paths))
            elif action_choice.action_type == t.ActionType.START_HANDOFF:
                players_available: List[m.Player] = action_choice.players
                ff_map = pf.FfTileMap(game.state)
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.state.pitch.get_ball_position() == player_square:
                        player_mover = pf.FfMover(player)
                        # Need 1 square of movement left to execute block after moving
                        finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True,heuristic=pf.BruteForceHeuristic())
                        paths = finder.find_paths(player_mover, player_square.x, player_square.y)
                        all_actions.extend(potential_handoff_actions(player, game, paths))
            elif action_choice.action_type == t.ActionType.END_TURN:
                all_actions.extend(potential_end_turn_action(game))

        if all_actions:
            all_actions.sort(key = lambda x: x.score, reverse=True)
            self.current_move = all_actions[0]
            print(self.current_move.description)

    def turn(self, game: g.Game):
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
        actions = set()
        for action_choice in game.state.available_actions:
            actions.add(action_choice.action_type)

        if t.ActionType.SELECT_DEFENDER_DOWN in actions:
            return m.Action(t.ActionType.SELECT_DEFENDER_DOWN)

        if t.ActionType.SELECT_DEFENDER_STUMBLES in actions:
            return m.Action(t.ActionType.SELECT_DEFENDER_STUMBLES)

        if t.ActionType.SELECT_PUSH in actions:
            return m.Action(t.ActionType.SELECT_PUSH)

        if t.ActionType.SELECT_BOTH_DOWN in actions:
            return m.Action(t.ActionType.SELECT_BOTH_DOWN)

        if t.ActionType.USE_REROLL in actions:
            return m.Action(t.ActionType.USE_REROLL)

        if t.ActionType.SELECT_ATTACKER_DOWN in actions:
            return m.Action(t.ActionType.SELECT_ATTACKER_DOWN)

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
        winner = game.get_winner()
        print("Casualties: ", game.num_casualties())
        if winner is None:
            print("It's a draw")
        elif winner == self.my_team:
            print("I ({}) won".format(self.name))
        else:
            print("I ({}) lost".format(self.name))


class ActionSequence():
    ''' Class containing a single possible move of a single player.
    '''

    def __init__(self, action_steps: List[m.Action], score: float = 0, description: str = ''):
        ''' Creates a new ActionSequence - an ordered list of sequential m.Actions to attempt to undertake.
        :param action_steps: Sequence of action steps that form this action.
        :param score: A score representing the attractiveness of the move (default: 0)
        :param description: A debug string (defaul: '')
        '''

        # Note the intention of this object is that when the object is acting, as steps are completed,
        # they are removed from the move_sequence so the next move is always the top of the move_sequence
        # list.

        self.action_steps = action_steps
        self.score = score
        self.description = description

    def is_valid(self, game: g.Game) -> bool:
        ''' Check if move can be executed given current game state.
        Checks if the object represents a valid, executable sequence given current game state.  For example, steps must
        be adjacent, and begin adjacent to players present position.  Player must be available to move (another player
        is not already moving, player has not already moved) etc.
        :param game:
        :return: True if controlling bot program *should* be able to execute the set of steps represented, else False
        '''
        pass

    def popleft(self):
        val = self.action_steps[0]
        del self.action_steps[0]
        return val


def potential_end_turn_action(game: g.Game) -> List[ActionSequence]:
    ''' Returns a scored end-turn action
    :param game:
    :return:
    '''
    actions: List[ActionSequence] = []
    action_steps: List[m.Action] = []
    action_steps.append(m.Action(t.ActionType.END_TURN))
    actions.append(ActionSequence(action_steps, score = -1, description='End Turn'))
    return actions


def potential_block_actions(player: m.Player, game: g.Game) -> List[ActionSequence]:
    ''' Return set of all scored possible "MOVE" actions for given player

    :param player:
    :param game:
    :param paths:
    :return:
    '''
    move_actions: List[ActionSequence] = []
    if not player.state.up:
        # There is currently a bug in the controlling logic.  Prone players shouldn't be able to block
        return move_actions
    blockable_squares: List[m.Player] = game.state.pitch.adjacent_player_squares(player, include_own=False, include_opp=True, manhattan=False, only_blockable=True, only_foulable=False)
    for blockable_square in blockable_squares:
        action_steps: List[m.Action] = []
        action_steps.append(m.Action(t.ActionType.START_BLOCK, player=player))
        action_steps.append(m.Action(t.ActionType.BLOCK, pos=blockable_square, player=player))
        action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))
        move_actions.append(ActionSequence(action_steps, score = random.randint(1, 100)/20, description='Block ' + player.name + ' to ' + str(blockable_square.x) + ',' + str(blockable_square.y)))
        # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_blitz_actions(player: m.Player, game: g.Game, paths: List[pf.Path]) -> List[ActionSequence]:
    ''' Return set of all scored possible "MOVE" actions for given player

    :param player:
    :param game:
    :param paths:
    :return:
    '''
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
            move_actions.append(ActionSequence(action_steps, score = len(path.steps)*(1.0 - path.cost) + random.randint(1,100)/100, description='Blitz ' + player.name + ' to ' + str(blockable_square.x) + ',' + str(blockable_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_pass_actions(player: m.Player, game: g.Game, paths: List[pf.Path]) -> List[ActionSequence]:
    ''' Return set of all scored possible "MOVE" actions for given player

    :param player:
    :param game:
    :param paths:
    :return:
    '''
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: m.Square = game.state.pitch.get_square(path.steps[-1].x, path.steps[-1].y)
        # Need possible receving players
        to_squares = game.state.pitch.passes_at(player, game.state.weather, end_square)
        for to_square in to_squares:
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(t.ActionType.START_PASS, player=player))
            if not player.state.up:
                action_steps.append(m.Action(t.ActionType.STAND_UP, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(m.Action(t.ActionType.MOVE, pos=game.state.pitch.get_square(step.x, step.y), player=player))
            action_steps.append(m.Action(t.ActionType.PASS, pos=to_square, player=player))
            action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))
            to_player: m.Player = game.state.pitch.get_player_at(to_square)
            if player is not None and to_player.state.up and to_player.team == player.team:
                # Favourable score to pass to a standing player on the same team
                cur_score = 10
            else:
                cur_score = -1
            move_actions.append(ActionSequence(action_steps, score = cur_score, description='Pass ' + player.name + ' to ' + str(to_square.x) + ',' + str(to_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_handoff_actions(player: m.Player, game: g.Game, paths: List[pf.Path]) -> List[ActionSequence]:
    ''' Return set of all scored possible "MOVE" actions for given player

    :param player:
    :param game:
    :param paths:
    :return:
    '''
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: m.Square = game.state.pitch.get_square(path.steps[-1].x, path.steps[-1].y)
        handoffable_players = game.state.pitch.adjacent_player_squares_at(player, end_square, include_own=True, include_opp=False, manhattan=False, only_blockable=True, only_foulable=False)
        for handoffable_player in handoffable_players:
            action_steps: List[m.Action] = []
            action_steps.append(m.Action(t.ActionType.START_HANDOFF, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(m.Action(t.ActionType.MOVE, pos=game.state.pitch.get_square(step.x, step.y), player=player))
            action_steps.append(m.Action(t.ActionType.HANDOFF, pos=handoffable_player.position, player=player))
            action_steps.append(m.Action(t.ActionType.END_PLAYER_TURN, player=player))
            move_actions.append(ActionSequence(action_steps, score = len(path.steps)*(1.0 - path.cost) + random.randint(1,100)/1000, description='Handoff ' + player.name + ' to ' + str(handoffable_player.position.x) + ',' + str(handoffable_player.position.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_foul_actions(player: m.Player, game: g.Game, paths: List[pf.Path]) -> List[ActionSequence]:
    ''' Return set of all scored possible "MOVE" actions for given player

    :param player:
    :param game:
    :param paths:
    :return:
    '''
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
            move_actions.append(ActionSequence(action_steps, score = len(path.steps)*(1.0 - path.cost) + random.randint(1,100)/10, description='Foul ' + player.name + ' to ' + str(foulable_square.x) + ',' + str(foulable_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_move_actions(player: m.Player, game: g.Game, paths: List[pf.Path]) -> List[ActionSequence]:
    ''' Return set of all scored possible "MOVE" actions for given player

    :param player:
    :param game:
    :param paths:
    :return:
    '''
    move_actions: List[ActionSequence] = []
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
        move_actions.append(ActionSequence(action_steps, score = len(path.steps)*(1.0 - path.cost) + random.randint(1,100)/1000, description='Move ' + player.name + ' to ' + str(path_steps[-1].x) + ',' + str(path_steps[-1].y)))
        # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


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
