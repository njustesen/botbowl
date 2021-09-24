#!/usr/bin/env python3

import numpy as np
import ffai
from typing import List
from ffai import Action, ActionType, Square, BBDieResult, Skill, Team, Formation, ProcBot, Game
import ffai.ai.pathfinding as pf
import time


class Script:

    def plan(self, game: Game, team: Team) -> List[Action]:
        raise NotImplementedError

    def unmarked_players(self, game, team):
        players = []
        for player in team.players:
            if player.position is not None and not player.state.used and game.num_tackle_zones_in(player) == 0:
                players.append(player)
        return players

    def offensive_assist_positions(self, game, team):
        opp_team = game.get_opp_team(team)
        ball_carrer = game.get_ball_carrier()
        assist_positions = []
        for opp_player in game.get_players_on_pitch(opp_team, up=True):
            for teammate in game.get_adjacent_opponents(opp_player, down=False):
                if not teammate.state.used:
                    own_str, opp_str = game.get_block_strengths(teammate, opp_player)
                    score = 0
                    if own_str < opp_str:
                        score += 1
                    elif own_str == opp_str:
                        score += 2
                    elif own_str > opp_str:
                        score += 0.5
                    if opp_player == ball_carrer:
                        score *= 2
                    if opp_player.has_skill(Skill.BLOCK):
                        score *= 0.75
                    for open_position in game.get_adjacent_squares(opp_player.position, occupied=False):
                        if len(game.get_adjacent_players(open_position, team=opp_team, down=False)) == 1:
                            assist_positions.append((open_position, score))
        assist_positions.sort(key=lambda x: x[1], reverse=True)
        if len(assist_positions) == 0:
            return []
        return [assist_position[0] for assist_position in assist_positions]

    def defensive_assist_positions(self, game, team):
        opp_team = game.get_opp_team(team)
        deny_2d_assist_positions = []
        deny_1d_assist_positions = []
        for opp_player in game.get_players_on_pitch(opp_team, up=True):
            for teammate in game.get_adjacent_opponents(opp_player, down=False):
                opp_str, own_str = game.get_block_strengths(opp_player, teammate)
                if opp_str > own_str:
                    for open_position in game.get_adjacent_squares(opp_player.position, occupied=False):
                        if len(game.get_adjacent_players(open_position, team=opp_team, down=False)) == 1:
                            deny_2d_assist_positions.append(open_position)
                elif opp_str == own_str:
                    for open_position in game.get_adjacent_squares(opp_player.position, occupied=False):
                        if len(game.get_adjacent_players(open_position, team=opp_team, down=False)) == 1:
                            deny_1d_assist_positions.append(open_position)
        return deny_2d_assist_positions + deny_1d_assist_positions


class StandUpScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        for player in game.get_players_on_pitch(team, up=False, used=False):
            if not player.state.stunned and game.num_tackle_zones_in(player) > 0:
                actions.append(Action(ActionType.START_MOVE, player=player))
                actions.append(Action(ActionType.STAND_UP))
                actions.append(Action(ActionType.END_PLAYER_TURN))
        return actions


class TouchdownScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        ball_carrier = game.get_ball_carrier()
        if ball_carrier is not None and ball_carrier.team == team and not ball_carrier.state.used:
            # Can ball carrier score with high probability
            td_path = pf.get_safest_path_to_endzone(game, ball_carrier, allow_team_reroll=True)
            if td_path is not None:
                actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
                actions.append(Action(ActionType.MOVE, position=td_path.steps[-1]))
        return actions


class HandoffTouchdownScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        ball_carrier = game.get_ball_carrier()
        # Move ball carrier to endzone
        if ball_carrier is not None and ball_carrier.team == team and not ball_carrier.state.used:
            # Hand-off action to scoring player
            if game.is_handoff_available():
                # Get players in scoring range
                unused_teammates = []
                for player in team.players:
                    if player.position is not None and player != ball_carrier and not player.state.used and player.state.up:
                        unused_teammates.append(player)
                # Find other players in scoring range
                handoff_p = None
                handoff_path = None
                for player in unused_teammates:
                    if game.get_distance_to_endzone(player) > player.num_moves_left():
                        continue
                    td_path = pf.get_safest_path_to_endzone(game, player, allow_team_reroll=True)
                    if td_path is None:
                        continue
                    handoff_path = pf.get_safest_path(game, ball_carrier, player.position, allow_team_reroll=True)
                    if handoff_path is None:
                        continue
                    p_catch = game.get_catch_prob(player, handoff=True, allow_catch_reroll=True, allow_team_reroll=True)
                    p = td_path.prob * handoff_path.prob * p_catch
                    if handoff_p is None or p > handoff_p:
                        handoff_p = p
                        handoff_path = handoff_path

                # Hand-off if high probability or last turn
                if handoff_path is not None and (handoff_p >= 0.7 or team.state.turn == 8):
                    actions = [Action(ActionType.START_HANDOFF, player=ball_carrier),
                               Action(ActionType.MOVE, handoff_path.steps[-1])]
        return actions


class EndzoneScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        ball_carrier = game.get_ball_carrier()
        # Move ball carrier to endzone
        if ball_carrier is not None and ball_carrier.team == team and not ball_carrier.state.used:

            # Move safely towards the endzone
            if game.num_tackle_zones_in(ball_carrier) == 0:
                paths = pf.get_all_paths(game, ball_carrier)
                best_path = None
                best_distance = 100
                target_x = game.get_opp_endzone_x(team)
                for path in paths:
                    distance_to_endzone = abs(target_x - path.steps[-1].x)
                    if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance):
                        best_path = path
                        best_distance = distance_to_endzone
                if best_path is not None:
                    steps = []
                    for step in best_path.steps:
                        if game.num_tackle_zones_at(ball_carrier, step) > 0:
                            break
                        if len(steps) >= ball_carrier.num_moves_left():
                            break
                        steps.append(step)
                    if len(steps) > 0:
                        actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
                        for step in steps:
                            actions.append(Action(ActionType.MOVE, position=step))

        return actions


class SafeBlockScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        if game.is_quick_snap():
            return actions
        attacker, defender, p_self_up, p_opp_down, block_p_fumble_self, block_p_fumble_opp = self._get_safest_block(game, team)
        if attacker is not None:
            actions.append(Action(ActionType.START_BLOCK, player=attacker))
            actions.append(Action(ActionType.BLOCK, position=defender.position))
        return actions

    def _get_safest_block(self, game, team):
        block_attacker = None
        block_defender = None
        block_p_self_up = None
        block_p_opp_down = None
        block_p_fumble_self = None
        block_p_fumble_opp = None
        for attacker in team.players:
            if attacker.position is not None and not attacker.state.used and attacker.state.up:
                for defender in game.get_adjacent_opponents(attacker, down=False):
                    p_self, p_opp, p_fumble_self, p_fumble_opp = game.get_block_probs(attacker, defender)
                    p_self_up = (1-p_self)
                    if block_p_self_up is None or (p_self_up > block_p_self_up and p_opp >= p_fumble_self):
                        block_p_self_up = p_self_up
                        block_p_opp_down = p_opp
                        block_attacker = attacker
                        block_defender = defender
                        block_p_fumble_self = p_fumble_self
                        block_p_fumble_opp = p_fumble_opp
        return block_attacker, block_defender, block_p_self_up, block_p_opp_down, block_p_fumble_self, block_p_fumble_opp


class PickupScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        ball_carrier = game.get_ball_carrier()
        # Pickup ball
        if game.get_ball_carrier() is None:
            pickup_p = None
            pickup_player = None
            pickup_path = None
            for player in team.players:
                if player.position is not None and not player.state.used:
                    if player.position.distance(game.get_ball_position()) <= player.get_ma() + 2:
                        path = pf.get_safest_path(game, player, game.get_ball_position())
                        if path is not None:
                            p = path.prob
                            if pickup_p is None or p > pickup_p:
                                pickup_p = p
                                pickup_player = player
                                pickup_path = path
            if pickup_player is not None:
                actions.append(Action(ActionType.START_MOVE, player=pickup_player))
                if not pickup_player.state.up:
                    actions.append(Action(ActionType.STAND_UP))
                for step in pickup_path.steps:
                    actions.append(Action(ActionType.MOVE, position=step))
                # Pick up the ball with {pickup_player.role.name}, p={pickup_p}
                # Find safest path towards endzone
                if game.num_tackle_zones_at(pickup_player, game.get_ball_position()) == 0:
                    paths = pf.get_all_paths(game, pickup_player, from_position=game.get_ball_position(),
                                             num_moves_used=len(pickup_path))
                    best_path = None
                    best_distance = 100
                    target_x = game.get_opp_endzone_x(team)
                    for path in paths:
                        distance_to_endzone = abs(target_x - path.steps[-1].x)
                        if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance):
                            best_path = path
                            best_distance = distance_to_endzone
                    if best_path is not None:
                        steps = []
                        for step in best_path.steps:
                            if game.num_tackle_zones_at(pickup_player, step) > 0:
                                break
                            if len(steps) + len(pickup_path.steps) >= pickup_player.get_ma():
                                break
                            steps.append(step)
                        if len(steps) > 0:
                            actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
                            for step in steps:
                                actions.append(Action(ActionType.MOVE, position=step))
        return actions


class ReceiversScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        ball_carrier = game.get_ball_carrier()
        # Move receivers into scoring distance if not already
        for player in self.unmarked_players(game, team):
            if player.has_skill(Skill.CATCH) and player != ball_carrier:
                if game.get_distance_to_endzone(player) > player.num_moves_left():
                    continue
                paths = pf.get_all_paths(game, player)
                best_path = None
                best_distance = 100
                target_x = game.get_opp_endzone_x(team)
                for path in paths:
                    distance_to_endzone = abs(target_x - path.steps[-1].x)
                    if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance):
                        best_path = path
                        best_distance = distance_to_endzone
                if best_path is not None:
                    steps = []
                    for step in best_path.steps:
                        if len(steps) >= player.get_ma() + (3 if not player.state.up else 0):
                            break
                        if game.num_tackle_zones_at(player, step) > 0:
                            break
                        if step.distance(best_path.steps[-1]) < player.get_ma():
                            break
                        steps.append(step)
                    if len(steps) > 0:
                        actions.append(Action(ActionType.START_MOVE, player=player))
                        if not player.state.up:
                            actions.append(Action(ActionType.STAND_UP))
                        for step in steps:
                            actions.append(Action(ActionType.MOVE, position=step))
        return actions


class BlitzScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        # Blitz with open block players
        if game.is_blitz_available():
            best_blitz_attacker = None
            best_blitz_defender = None
            best_blitz_score = None
            best_blitz_path = None
            for blitzer in game.get_players_on_pitch(team, used=False):
                paths = pf.get_all_paths(game, blitzer, blitz=blitzer.has_skill(Skill.BLOCK))
                for path in paths:
                    defender = game.get_player_at(path.steps[-1])
                    if defender is None:
                        continue
                    final_position = path.steps[-2] if len(path.steps) > 1 else blitzer.position
                    p_self, p_opp, p_fumble_self, p_fumble_opp = game.get_blitz_probs(blitzer,
                                                                                      final_position,
                                                                                      defender)
                    p_self_up = path.prob * (1 - p_self)
                    p_opp = path.prob * p_opp
                    p_fumble_opp = p_fumble_opp * path.prob
                    if blitzer == game.get_ball_carrier():
                        p_fumble_self = path.prob + (1 - path.prob) * p_fumble_self
                    score = p_self_up + p_opp + p_fumble_opp - p_fumble_self
                    if best_blitz_score is None or score > best_blitz_score:
                        best_blitz_attacker = blitzer
                        best_blitz_defender = defender
                        best_blitz_score = score
                        best_blitz_path = path
            if best_blitz_path is not None:
                actions.append(Action(ActionType.START_BLITZ, player=best_blitz_attacker))
                actions.append(Action(ActionType.MOVE, position=best_blitz_path.steps[-1]))
        return actions


class CageScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        ball_carrier = game.get_ball_carrier()
        # Make cage around ball carrier
        cage_positions = [
            Square(game.get_ball_position().x - 1, game.get_ball_position().y - 1),
            Square(game.get_ball_position().x + 1, game.get_ball_position().y - 1),
            Square(game.get_ball_position().x - 1, game.get_ball_position().y + 1),
            Square(game.get_ball_position().x + 1, game.get_ball_position().y + 1)
        ]
        if ball_carrier is not None:
            for cage_position in cage_positions:
                if game.get_player_at(cage_position) is None and not game.is_out_of_bounds(cage_position):
                    for player in self.unmarked_players(game, team):
                        if player == ball_carrier or player.position in cage_positions:
                            continue
                        if player.position.distance(cage_position) > player.num_moves_left():
                            continue
                        if game.num_tackle_zones_in(player) > 0:
                            continue
                        path = pf.get_safest_path(game, player, cage_position)
                        if path is not None and path.prob == 1:
                            actions.append(Action(ActionType.START_MOVE, player=player))
                            actions.append(Action(ActionType.MOVE, position=path.steps[-1]))
                            return actions
        return actions


class AssistScript(Script):

    def __init__(self, defensive=False):
        self.defensive = defensive

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        if self.defensive:
            assists_positions = self.defensive_assist_positions(game, team)
        else:
            assists_positions = self.offensive_assist_positions(game, team)

        for assist_position in assists_positions:
            for player in self.unmarked_players(game, team):
                if not player.position.distance(assist_position) <= player.get_ma():
                    continue
                path = pf.get_safest_path(game, player, assist_position)
                if path is not None and path.prob == 1:
                    actions.append(Action(ActionType.START_MOVE, player=player))
                    actions.append(Action(ActionType.MOVE, position=path.steps[-1]))
                    return actions
            return actions
        return []


class MoveToBallScript(Script):

    def plan(self, game: Game, team: Team) -> List[Action]:
        actions = []
        ball_carrier = game.get_ball_carrier()
        # Move towards the ball
        for player in self.unmarked_players(game, team):
            if player == ball_carrier:
                continue
            if game.num_tackle_zones_in(player) > 0:
                continue
            if ball_carrier is None and player.position != game.get_ball_position():
                paths = pf.get_all_paths(game, player)
                shortest_distance = None
                path = None
                for p in paths:
                    distance = p.steps[-1].distance(game.get_ball_position())
                    if p.prob == 1 and (shortest_distance is None or 0 < distance < shortest_distance):
                        shortest_distance = distance
                        path = p
                if path is not None:
                    actions.append(Action(ActionType.START_MOVE, player=player))
                    actions.append(Action(ActionType.MOVE, position=path.steps[-1]))
                    return actions
            elif ball_carrier.team != team:
                paths = pf.get_all_paths(game, player)
                shortest_distance = None
                path = None
                for p in paths:
                    distance = p.steps[-1].distance(ball_carrier.position)
                    if p.prob == 1 and (shortest_distance is None or 0 < distance < shortest_distance):
                        shortest_distance = distance
                        path = p
                if path is not None:
                    actions.append(Action(ActionType.START_MOVE, player=player))
                    actions.append(Action(ActionType.MOVE, position=path.steps[-1]))
                    return actions
        return actions


class PortfolioBot(ProcBot):

    def __init__(self, name, policy="ordered", seed=None):
        super().__init__(name)
        self.rnd = np.random.RandomState(seed)
        self.stats = {}
        self.my_team = None
        self.opp_team = None
        self.actions = []
        self.last_turn = 0
        self.last_half = 0
        self._script = None
        self.policy = policy
        self.turn_scripts = [
            StandUpScript(),
            TouchdownScript(),
            AssistScript(),
            AssistScript(defensive=True),
            SafeBlockScript(),
            BlitzScript(),
            HandoffTouchdownScript(),
            EndzoneScript(),
            PickupScript(),
            ReceiversScript(),
            CageScript(),
            MoveToBallScript(),
        ]

        self.blitz_scripts = [
            AssistScript(),
            AssistScript(defensive=True),
            BlitzScript(),
            PickupScript(),
            ReceiversScript(),
            CageScript(),
            MoveToBallScript()
        ]

        self.quick_snap_scripts = [

        ]

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

    def set_script(self, script: Script):
        self._script = script

    def new_game(self, game, team):
        """
        Called when a new game starts.
        """
        self.my_team = team
        self.opp_team = game.get_opp_team(team)
        self.last_turn = 0
        self.last_half = 0

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

    def setup(self, game):
        """
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

    def reroll(self, game):
        """
        Select between USE_REROLL and DONT_USE_REROLL
        """
        reroll_proc = game.get_procedure()
        context = reroll_proc.context
        if type(context) == ffai.Dodge:
            return Action(ActionType.USE_REROLL)
        if type(context) == ffai.Pickup:
            return Action(ActionType.USE_REROLL)
        if type(context) == ffai.PassAttempt:
            return Action(ActionType.USE_REROLL)
        if type(context) == ffai.Catch:
            return Action(ActionType.USE_REROLL)
        if type(context) == ffai.GFI:
            return Action(ActionType.USE_REROLL)
        if type(context) == ffai.Block:
            attacker = context.attacker
            attackers_down = 0
            for die in context.roll.dice:
                if die.get_value() == BBDieResult.ATTACKER_DOWN:
                    attackers_down += 1
                elif die.get_value() == BBDieResult.BOTH_DOWN and not attacker.has_skill(Skill.BLOCK) and not attacker.has_skill(Skill.WRESTLE):
                    attackers_down += 1
            if attackers_down > 0 and context.favor != self.my_team:
                return Action(ActionType.USE_REROLL)
            if attackers_down == len(context.roll.dice) and context.favor != self.opp_team:
                return Action(ActionType.USE_REROLL)
            return Action(ActionType.DONT_USE_REROLL)

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
                if Skill.BLOCK in player.get_skills() and game.num_tackle_zones_in(player) == 0:
                    return Action(ActionType.SELECT_PLAYER, player=player, position=ball_pos)
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
        # Update teams
        self.my_team = game.get_team_by_id(self.my_team.team_id)
        self.opp_team = game.get_opp_team(self.my_team)

        # Reset actions if new turn
        turn = game.get_agent_team(self).state.turn
        half = game.state.half
        if half > self.last_half or turn > self.last_turn:
            self.actions.clear()
            self.last_turn = turn
            self.last_half = half
            self.actions = []

        # End turn if only action left
        if len(game.state.available_actions) == 1:
            if game.state.available_actions[0].action_type == ActionType.END_TURN:
                self.actions = [Action(ActionType.END_TURN)]

        # Execute planned actions if any
        if len(self.actions) > 0:
            action = self._get_next_action()
            return action

        # Plan actions
        proc = game.get_procedure()
        if proc.blitz:
            self._make_plan(game, self.blitz_scripts)
        elif proc.quick_snap:
            self._make_plan(game, self.quick_snap_scripts)
        else:
            self._make_plan(game, self.turn_scripts)
        action = self._get_next_action()
        return action

    def _get_next_action(self):
        if len(self.actions) == 0:
            return Action(ActionType.END_TURN)
        action = self.actions[0]
        self.actions = self.actions[1:]
        return action

    def _make_plan(self, game: Game, scripts: List[Script]):
        actions = []
        available_scripts = scripts.copy()
        script = None
        while len(actions) == 0 and len(available_scripts) > 0:
            if self._script is not None:
                script = self._script
                self._script = None
            elif self.policy == "random":
                script = self.rnd.choice(list(available_scripts))
            elif self.policy == "ordered":
                script = available_scripts[0]
            available_scripts.remove(script)
            script_name = type(script).__name__
            print("Script", script_name)
            start = time.time()
            actions = script.plan(game, team=self.my_team)
            end = time.time()
            print("Actions:",actions)
            if script_name not in self.stats.keys():
                self.stats[script_name] = []
            self.stats[script_name].append(end - start)
        if script is None:
            print("No applicable script")
        else:
            print(script_name, ":", ", ".join([action.action_type.name for action in actions]))
        self.actions = actions

    def quick_snap(self, game):
        return Action(ActionType.END_TURN)

    def blitz(self, game):
        return self.turn(game)

    def player_action(self, game):
        # Execute planned actions if any
        if len(self.actions) > 0:
            action = self._get_next_action()
            return action
        ball_carrier = game.get_ball_carrier()
        if ball_carrier == game.get_active_player():
            td_path = pf.get_safest_path_to_endzone(game, ball_carrier)
            if td_path is not None and td_path.prob <= 0.9:
                self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
                for step in td_path.steps:
                    self.actions.append(Action(ActionType.MOVE, position=step))
                return
        return Action(ActionType.END_PLAYER_TURN)

    def block(self, game):
        """
        Select block die or reroll.
        """
        # Get attacker and defender
        attacker = game.get_procedure().attacker
        defender = game.get_procedure().defender
        is_blitz = game.get_procedure().blitz
        dice = game.num_block_dice(attacker, defender, blitz=is_blitz)

        # Loop through available dice results
        actions = set()
        for action_choice in game.state.available_actions:
            actions.add(action_choice.action_type)

        # 1. DEFENDER DOWN
        if ActionType.SELECT_DEFENDER_DOWN in actions:
            return Action(ActionType.SELECT_DEFENDER_DOWN)

        if ActionType.SELECT_DEFENDER_STUMBLES in actions and not (defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE)):
            return Action(ActionType.SELECT_DEFENDER_STUMBLES)

        if ActionType.SELECT_BOTH_DOWN in actions and not defender.has_skill(Skill.BLOCK) and attacker.has_skill(Skill.BLOCK):
            return Action(ActionType.SELECT_BOTH_DOWN)

        # 2. BOTH DOWN if opponent carries the ball and doesn't have block
        if ActionType.SELECT_BOTH_DOWN in actions and game.get_ball_carrier() == defender and not defender.has_skill(Skill.BLOCK):
            return Action(ActionType.SELECT_BOTH_DOWN)

        # 3. USE REROLL if defender carries the ball
        if ActionType.USE_REROLL in actions and game.get_ball_carrier() == defender:
            return Action(ActionType.USE_REROLL)

        # 4. PUSH
        if ActionType.SELECT_DEFENDER_STUMBLES in actions:
            return Action(ActionType.SELECT_DEFENDER_STUMBLES)

        if ActionType.SELECT_PUSH in actions:
            return Action(ActionType.SELECT_PUSH)

        # 5. BOTH DOWN
        if ActionType.SELECT_BOTH_DOWN in actions:
            return Action(ActionType.SELECT_BOTH_DOWN)

        # 6. USE REROLL to avoid attacker down unless a one-die block
        if ActionType.USE_REROLL in actions and dice > 1:
            return Action(ActionType.USE_REROLL)

        # 7. ATTACKER DOWN
        if ActionType.SELECT_ATTACKER_DOWN in actions:
            return Action(ActionType.SELECT_ATTACKER_DOWN)

    def push(self, game):
        """
        Select square to push to.
        """
        for position in game.state.available_actions[0].positions:
            return Action(ActionType.PUSH, position=position)

    def follow_up(self, game):
        """
        Follow up or not. ActionType.FOLLOW_UP must be used together with a position.
        """
        player = game.state.active_player
        for position in game.state.available_actions[0].positions:
            if player.position == position and game.get_ball_carrier() == player:
                return Action(ActionType.FOLLOW_UP, position=position)
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
            if action.action_type == ActionType.SELECT_PLAYER:
                for player, rolls in zip(action.players, action.rolls):
                    return Action(ActionType.SELECT_PLAYER, player=player)
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

    def pickup(self, game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)

    def use_juggernaut(self, game):
        return Action(ActionType.USE_SKILL)

    def use_wrestle(self, game):
        return Action(ActionType.USE_SKILL)

    def use_stand_firm(self, game):
        return Action(ActionType.USE_SKILL)

    def use_pro(self, game):
        return Action(ActionType.USE_SKILL)

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
            print(self.my_team.state.score, "-", self.opp_team.state.score)
        else:
            print("I ({}) lost".format(self.name))
            print(self.my_team.state.score, "-", self.opp_team.state.score)


# Register Portfolio
ffai.register_bot('portfolio-bot', PortfolioBot)


if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = ffai.load_config("bot-bowl-iii")
    #config = ffai.load_config("gym-1")
    config.competition_mode = False
    config.pathfinding_enabled = True
    # config = get_config("gym-7.json")
    # config = get_config("gym-5.json")
    # config = get_config("gym-3.json")
    ruleset = ffai.load_rule_set(config.ruleset)  # We don't need all the rules
    arena = ffai.load_arena(config.arena)
    home = ffai.load_team_by_filename("human", ruleset)
    away = ffai.load_team_by_filename("human", ruleset)

    # Play 10 games
    for i in range(1):
        home_agent = ffai.make_bot('portfolio-bot')
        away_agent = ffai.make_bot('random')
        config.debug_mode = False
        game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)

        print("---- STATS ----")

        by_time = {}
        for script_name, times in home_agent.stats.items():
            mean_time = np.mean(times)
            if mean_time not in times:
                by_time[mean_time] = []
            by_time[mean_time].append(script_name)

        for t in sorted(by_time.keys()):
            for script_name in by_time[t]:
                print(t, ":", script_name)
