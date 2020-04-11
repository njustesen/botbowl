

#!/usr/bin/env python3

import ffai
from ffai import Action, ActionType, Square, BBDieResult, Skill, PassDistance, Tile, Rules, Formation, ProcBot
import ffai.ai.pathfinding as pf
import ffai.ai.bothelper as helper
import ffai.core.model as m
import ffai.core.table as t

import time

class GrodBotTranslated(ProcBot):

    """
    A Bot that uses path finding to evaluate all possibilities.
    WIP!!! Hand-offs and Pass actions going a bit funny.

    Translated to new ffai API.
    """

    raise NotImplementedError("Not completed! ")

    BASE_SCORE_BLITZ = 55.0
    BASE_SCORE_FOUL = -50.0
    BASE_SCORE_BLOCK = 65*2  # For a two dice block
    BASE_SCORE_HANDOFF = 40.0
    BASE_SCORE_PASS = 40.0
    BASE_SCORE_MOVE_TO_OPPONENT = 45.0
    BASE_SCORE_MOVE_BALL = 45.0
    BASE_SCORE_MOVE_TOWARD_BALL = 0.0
    BASE_SCORE_MOVE_TO_SWEEP = 0.0
    BASE_SCORE_CAGE_BALL = 70.0
    BASE_SCORE_MOVE_TO_BALL = 60.0  # 60.0
    BASE_SCORE_BALL_AND_CHAIN = 75.0
    BASE_SCORE_DEFENSIVE_SCREEN = 0.0
    ADDITIONAL_SCORE_DODGE = 0.0  # Lower this value to dodge more.

    def __init__(self, name):
        # From GrodBot
        self.current_move: Optional[ActionSequence] = None
        self.verbose = verbose
        self.heat_map: Optional[helper.FfHeatMap] = None

        #From scripted bot (some same as grodbot)

        super().__init__(name)
        self.my_team = None
        self.opp_team = None
        self.actions = []
        self.last_turn = 0
        self.last_half = 0

        self.off_formation = [
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
            ["-", "-", "-", "-", "-", "s", "-", "-", "-", "0", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "S"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "m", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        self.def_formation = [
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "0"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "S", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "b", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
            ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        ]

        self.off_formation = Formation("Wedge offense", self.off_formation)
        self.def_formation = Formation("Zone defense", self.def_formation)
        self.setup_actions = []

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

    def coin_toss_kick_receive(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game):
        """
        TODO:   Add GrodBot's sort by bashiness here. For now use scripted bot's
                setup.

        Use either a Wedge offensive formation or zone defensive formation.
        """
        # Update teams
        self.my_team = game.get_team_by_id(self.my_team.team_id)
        self.opp_team = game.get_opp_team(self.my_team)

        if self.setup_actions:
            action = self.setup_actions.pop(0)
            return action
        else:
            if game.get_receiving_team() == self.my_team:
                self.setup_actions = self.off_formation.actions(game, self.my_team)
                self.setup_actions.append(Action(ActionType.END_SETUP))
            else:
                self.setup_actions = self.def_formation.actions(game, self.my_team)
                self.setup_actions.append(Action(ActionType.END_SETUP))

    def place_ball(self, game):
        """
        GrodBot has same implementation as scripted bot
        """
        left_center = Square(7, 8)
        right_center = Square(20, 8)
        if game.is_team_side(left_center, self.opp_team):
            return Action(ActionType.PLACE_BALL, position=left_center)
        return Action(ActionType.PLACE_BALL, position=right_center)

    def high_kick(self, game):
        """
        Select player to move under the ball.
        GRODBOT selects the blitziest player to put under the ball
        TODO: implement player_blitz_ability()
        """
        ball_pos = game.get_ball_position() #OK

        if game.is_team_side(game.get_ball_position(), self.my_team) and game.get_player_at(game.get_ball_position()) is None:
            players_available = game.get_players_on_pitch(self.my_team, up=True)
            if players_available:
                players_sorted = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)
                player = players_sorted[0]
                return Action(ActionType.SELECT_PLAYER, player=player, position=ball_pos)
        return Action(ActionType.SELECT_NONE)

    def touchback(self, game):
        """
        Select player to give the ball to.
        GRODBOT selects the blitziest player
        """
        players_available = game.get_players_on_pitch(self.my_team, up=True)
        if players_available:
            players_sorted = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)
            player = players_sorted[0]
            return Action(ActionType.SELECT_PLAYER, player=player)
        return Action(ActionType.SELECT_PLAYER, player=None)

    def turn(self, game):
        """
        Start a new player action / turn.
        GRODBOT implements a list of actions and pops the left side
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

    def set_next_move(self, game):
        """
        Not superloaded from ProcBot
        This function adds action(s) to self.current_move, later used in self.turn()
        """
        self.current_move = None

        players_moved: List[m.Player] = helper.get_players(game, self.my_team, include_own=True, include_opp=False, include_used=True, only_used=False)

        players_to_move: List[m.Player] = helper.get_players(game, self.my_team, include_own=True, include_opp=False, include_used=False)
        paths_own: Dict[m.Player, List[pf.Path]] = dict()
        ff_map = pf.FfTileMap(game.state)
        for player in players_to_move:
            player_mover = pf.FfMover(player)
            finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True, heuristic=pf.BruteForceHeuristic(), probability_costs=True)
            paths = finder.find_paths(player_mover, player.position.x, player.position.y)
            paths_own[player] = paths

        players_opponent: List[m.Player] = helper.get_players(game, self.my_team, include_own=False, include_opp=True, include_stunned=False)
        paths_opposition: Dict[m.Player, List[pf.Path]] = dict()
        for player in players_opponent:
            player_mover = pf.FfMover(player)
            finder = pf.AStarPathFinder(ff_map, player.move_allowed(), allow_diag_movement=True, heuristic=pf.BruteForceHeuristic(), probability_costs=True)
            paths = finder.find_paths(player_mover, player.position.x, player.position.y)
            paths_opposition[player] = paths

        # Create a heat-map of control zones
        heat_map: helper.FfHeatMap = helper.FfHeatMap(game, self.my_team)
        heat_map.add_unit_by_paths(game, paths_opposition)
        heat_map.add_unit_by_paths(game, paths_own)
        heat_map.add_players_moved(game, helper.get_players(game, self.my_team, include_own=True, include_opp=False, only_used=True))
        self.heat_map = heat_map

        all_actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            if action_choice.action_type == t.ActionType.START_MOVE:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = paths_own[player]
                    all_actions.extend(potential_move_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.START_BLITZ:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    player_mover = pf.FfMover(player)
                    finder = pf.AStarPathFinder(ff_map, player.move_allowed() - 1, allow_diag_movement=True, heuristic=pf.BruteForceHeuristic(), probability_costs=True)
                    paths = finder.find_paths(player_mover, player.position.x, player.position.y)
                    all_actions.extend(potential_blitz_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.START_FOUL:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    paths = paths_own[player]
                    all_actions.extend(potential_foul_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.START_BLOCK:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    all_actions.extend(potential_block_actions(game, heat_map, player))
            elif action_choice.action_type == t.ActionType.START_PASS:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.state.pitch.get_ball_position() == player_square:
                        paths = paths_own[player]
                        all_actions.extend(potential_pass_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.START_HANDOFF:
                players_available: List[m.Player] = action_choice.players
                for player in players_available:
                    player_square: m.Square = player.position
                    if game.state.pitch.get_ball_position() == player_square:
                        paths = paths_own[player]
                        all_actions.extend(potential_handoff_actions(game, heat_map, player, paths))
            elif action_choice.action_type == t.ActionType.END_TURN:
                all_actions.extend(potential_end_turn_action(game))

        if all_actions:
            all_actions.sort(key=lambda x: x.score, reverse=True)
            self.current_move = all_actions[0]

            if self.verbose:
                print('   Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + game.state.current_team.name + ', Action=' + self.current_move.description + ', Score=' + str(self.current_move.score))
