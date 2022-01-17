#!/usr/bin/env python3
from typing import List

import botbowl
from botbowl import Action, ActionType, Square, BBDieResult, Skill, Formation, ProcBot
import botbowl.core.pathfinding as pf
import time

from botbowl.core.pathfinding.python_pathfinding import Path  # Only used for type checker

class MyScriptedBot(ProcBot):

    def __init__(self, name):
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
        self.last_turn = 0
        self.last_half = 0

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
            action = self.setup_actions.pop(0)
            return action


    def reroll(self, game):
        """
        Select between USE_REROLL and DONT_USE_REROLL
        """
        reroll_proc = game.get_procedure()
        context = reroll_proc.context
        if type(context) == botbowl.Dodge:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.Pickup:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.PassAttempt:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.Catch:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.GFI:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.BloodLust:
            return Action(ActionType.USE_REROLL)
        if type(context) == botbowl.Block:
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
            #print(f"Half: {half}")
            #print(f"Turn: {turn}")

        # End turn if only action left
        if len(game.state.available_actions) == 1:
            if game.state.available_actions[0].action_type == ActionType.END_TURN:
                self.actions = [Action(ActionType.END_TURN)]

        # Execute planned actions if any
        if len(self.actions) > 0:
            action = self._get_next_action()
            return action

        # Split logic depending on offense, defense, and loose ball - and plan actions
        ball_carrier = game.get_ball_carrier()
        self._make_plan(game, ball_carrier)
        action = self._get_next_action()
        return action

    def _get_next_action(self):
        action = self.actions[0]
        self.actions = self.actions[1:]
        #print(f"Action: {action.to_json()}")
        return action

    def _make_plan(self, game, ball_carrier):
        #print("1. Stand up marked players")
        for player in self.my_team.players:
            if player.position is not None and not player.state.up and not player.state.stunned and not player.state.used:
                if game.num_tackle_zones_in(player) > 0:
                    self.actions.append(Action(ActionType.START_MOVE, player=player))
                    self.actions.append(Action(ActionType.STAND_UP))
                    #print(f"Stand up marked player {player.role.name}")
                    return

        #print("2. Move ball carrier to endzone")
        if ball_carrier is not None and ball_carrier.team == self.my_team and not ball_carrier.state.used:
            #print("2.1 Can ball carrier score with high probability")
            td_path = pf.get_safest_path_to_endzone(game, ball_carrier, allow_team_reroll=True)
            if td_path is not None and td_path.prob >= 0.7:
                self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
                self.actions.extend(path_to_move_actions(game, td_path))
                #print(f"Score with ball carrier, p={td_path.prob}")
                return

            #print("2.2 Hand-off action to scoring player")
            if game.is_handoff_available():

                # Get players in scoring range
                unused_teammates = []
                for player in self.my_team.players:
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
                if handoff_path is not None and (handoff_p >= 0.7 or self.my_team.state.turn == 8):
                    self.actions = [Action(ActionType.START_HANDOFF, player=ball_carrier),
                                    Action(ActionType.HANDOFF, handoff_path.steps[-1])]
                    return

            #print("2.3 Move safely towards the endzone")
            if game.num_tackle_zones_in(ball_carrier) == 0:
                paths = pf.get_all_paths(game, ball_carrier)
                best_path = None
                best_distance = 100
                target_x = game.get_opp_endzone_x(self.my_team)
                for path in paths:
                    distance_to_endzone = abs(target_x - path.steps[-1].x)
                    if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance) and game.num_tackle_zones_at(ball_carrier, path.get_last_step()) == 0:
                        best_path = path
                        best_distance = distance_to_endzone
                if best_path is not None:
                    self.actions.append(Action(ActionType.START_MOVE, player=ball_carrier))
                    self.actions.extend(path_to_move_actions(game, best_path))
                    #print(f"Move ball carrier {ball_carrier.role.name}")
                    return

        #print("3. Safe blocks")
        attacker, defender, p_self_up, p_opp_down, block_p_fumble_self, block_p_fumble_opp = self._get_safest_block(game)
        if attacker is not None and p_self_up > 0.94 and block_p_fumble_self == 0:
            self.actions.append(Action(ActionType.START_BLOCK, player=attacker))
            self.actions.append(Action(ActionType.BLOCK, position=defender.position))
            #print(f"Safe block with {attacker.role.name} -> {defender.role.name}, p_self_up={p_self_up}, p_opp_down={p_opp_down}")
            return

        #print("4. Pickup ball")
        if game.get_ball_carrier() is None:
            pickup_p = None
            pickup_player = None
            pickup_path = None
            for player in self.my_team.players:
                if player.position is not None and not player.state.used:
                    if player.position.distance(game.get_ball_position()) <= player.get_ma() + 2:
                        path = pf.get_safest_path(game, player, game.get_ball_position())
                        if path is not None:
                            p = path.prob
                            if pickup_p is None or p > pickup_p:
                                pickup_p = p
                                pickup_player = player
                                pickup_path = path
            if pickup_player is not None and pickup_p > 0.33:
                self.actions.append(Action(ActionType.START_MOVE, player=pickup_player))
                if not pickup_player.state.up:
                    self.actions.append(Action(ActionType.STAND_UP))
                self.actions.extend(path_to_move_actions(game, pickup_path))
                #print(f"Pick up the ball with {pickup_player.role.name}, p={pickup_p}")
                # Find safest path towards endzone
                if game.num_tackle_zones_at(pickup_player, game.get_ball_position()) == 0:
                    paths = pf.get_all_paths(game, pickup_player, from_position=game.get_ball_position(), num_moves_used=len(pickup_path))
                    best_path = None
                    best_distance = 100
                    target_x = game.get_opp_endzone_x(self.my_team)
                    for path in paths:
                        distance_to_endzone = abs(target_x - path.steps[-1].x)
                        if path.prob == 1 and (best_path is None or distance_to_endzone < best_distance) and game.num_tackle_zones_at(pickup_player, path.get_last_step()) == 0:
                            best_path = path
                            best_distance = distance_to_endzone
                    if best_path is not None:
                        self.actions.extend(path_to_move_actions(game, best_path))
                        #print(f"- Move ball carrier {pickup_player.role.name}")
                return

        # Scan for unused players that are not marked
        open_players = []
        for player in self.my_team.players:
            if player.position is not None and not player.state.used and game.num_tackle_zones_in(player) == 0:
                open_players.append(player)

        #print("5. Move receivers into scoring distance if not already")
        for player in open_players:
            if player.has_skill(Skill.CATCH) and player != ball_carrier:
                if game.get_distance_to_endzone(player) > player.num_moves_left():
                    continue
                paths = pf.get_all_paths(game, ball_carrier)
                best_path = None
                best_distance = 100
                target_x = game.get_opp_endzone_x(self.my_team)
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
                        self.actions.append(Action(ActionType.START_MOVE, player=player))
                        if not player.state.up:
                            self.actions.append(Action(ActionType.STAND_UP))
                        for step in steps:
                            self.actions.append(Action(ActionType.MOVE, position=step))
                        print(f"Move receiver {player.role.name}")
                        return

        #print("6. Blitz with open block players")
        if game.is_blitz_available():

            best_blitz_attacker = None
            best_blitz_defender = None
            best_blitz_score = None
            best_blitz_path = None
            for blitzer in open_players:
                if blitzer.position is not None and not blitzer.state.used and blitzer.has_skill(Skill.BLOCK):
                    blitz_paths = pf.get_all_paths(game, blitzer, blitz=True)
                    for path in blitz_paths:
                        final_position = path.steps[-2] if len(path.steps) > 1 else blitzer.position
                        for defender in game.get_adjacent_players(final_position, team=game.get_opp_team(blitzer.team)):
                            p_self, p_opp, p_fumble_self, p_fumble_opp = game.get_blitz_probs(blitzer, final_position, defender)
                            p_self_up = path.prob * (1-p_self)
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
            if best_blitz_attacker is not None and best_blitz_score >= 1.25:
                self.actions.append(Action(ActionType.START_BLITZ, player=best_blitz_attacker))
                self.actions.extend(path_to_move_actions(game, best_blitz_path))
                #print(f"Blitz with {best_blitz_attacker.role.name}, score={best_blitz_score}")
                return

        #print("7. Make cage around ball carrier")
        cage_positions = [
            Square(game.get_ball_position().x - 1, game.get_ball_position().y - 1),
            Square(game.get_ball_position().x + 1, game.get_ball_position().y - 1),
            Square(game.get_ball_position().x - 1, game.get_ball_position().y + 1),
            Square(game.get_ball_position().x + 1, game.get_ball_position().y + 1)
        ]
        if ball_carrier is not None:
            for cage_position in cage_positions:
                if game.get_player_at(cage_position) is None and not game.is_out_of_bounds(cage_position):
                    for player in open_players:
                        if player == ball_carrier or player.position in cage_positions:
                            continue
                        if player.position.distance(cage_position) > player.num_moves_left():
                            continue
                        if game.num_tackle_zones_in(player) > 0:
                            continue
                        path = pf.get_safest_path(game, player, cage_position)
                        if path is not None and path.prob > 0.94:
                            self.actions.append(Action(ActionType.START_MOVE, player=player))
                            self.actions.extend(path_to_move_actions(game, path))
                            #print(f"Make cage around towards ball carrier {player.role.name}")
                            return

        # Scan for assist positions
        assist_positions = []
        for player in game.get_opp_team(self.my_team).players:
            if player.position is None or not player.state.up:
                continue
            opponents = game.get_adjacent_opponents(player, down=False)
            for opponent in opponents:
                att_str, def_str = game.get_block_strengths(player, opponent)
                if def_str >= att_str:
                    for open_position in game.get_adjacent_squares(player.position, occupied=False):
                        if len(game.get_adjacent_players(open_position, team=self.opp_team, down=False)) == 1:
                            assist_positions.append(open_position)

        #print("8. Move non-marked players to assist")
        for player in open_players:
            paths = pf.get_all_paths(game, player)
            for assist_position in assist_positions:
                assist_path = None
                for path in paths:
                    if path.steps[-1] == assist_position:
                        if path.prob == 1:
                            self.actions.append(Action(ActionType.START_MOVE, player=player))
                            self.actions.extend(path_to_move_actions(game, path))
                            #print(f"Move assister {player.role.name} to {assist_position.to_json}")
                            return

        #print("9. Move towards the ball")
        for player in open_players:
            if player == ball_carrier:
                continue
            if game.num_tackle_zones_in(player) > 0:
                continue
            if ball_carrier is None:
                paths = pf.get_all_paths(game, player)
                shortest_distance = None
                path = None
                for p in paths:
                    distance = p.steps[-1].distance(game.get_ball_position())
                    if shortest_distance is None or (p.prob == 1 and distance < shortest_distance):
                        shortest_distance = distance
                        path = p
            elif ball_carrier.team != self.my_team:
                paths = pf.get_all_paths(game, player)
                shortest_distance = None
                path = None
                for p in paths:
                    distance = p.steps[-1].distance(ball_carrier.position)
                    if shortest_distance is None or (p.prob == 1 and distance < shortest_distance):
                        shortest_distance = distance
                        path = p
            else:
                continue
            if path is not None:
                if len(path.steps) > 0:
                    self.actions.append(Action(ActionType.START_MOVE, player=player))
                    self.actions.extend(path_to_move_actions(game, path))
                    #print(f"Move towards ball {player.role.name}")
                    return

        #print("10. Risky blocks")
        attacker, defender, p_self_up, p_opp_down, block_p_fumble_self, block_p_fumble_opp = self._get_safest_block(game)
        if attacker is not None and (p_opp_down > (1-p_self_up) or block_p_fumble_opp > 0):
            self.actions.append(Action(ActionType.START_BLOCK, player=attacker))
            self.actions.append(Action(ActionType.BLOCK, position=defender.position))
            #print(f"Block with {player.role.name} -> {defender.role.name}, p_self_up={p_self_up}, p_opp_down={p_opp_down}")
            return

        #print("11. End turn")
        self.actions.append(Action(ActionType.END_TURN))

    def _get_safest_block(self, game):
        block_attacker = None
        block_defender = None
        block_p_self_up = None
        block_p_opp_down = None
        block_p_fumble_self = None
        block_p_fumble_opp = None
        for attacker in self.my_team.players:
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

    def quick_snap(self, game):
        return Action(ActionType.END_TURN)

    def blitz(self, game):
        return Action(ActionType.END_TURN)

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
                self.actions.extend(path_to_move_actions(game, td_path))
                #print(f"Scoring with {ball_carrier.role.name}, p={td_path.prob}")
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
        # return Action(ActionType.DONT_USE_REROLL)

    def pickup(self, game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def use_juggernaut(self, game):
        return Action(ActionType.USE_SKILL)
        # return Action(ActionType.DONT_USE_SKILL)

    def use_wrestle(self, game):
        return Action(ActionType.USE_SKILL)
        # return Action(ActionType.DONT_USE_SKILL)

    def use_stand_firm(self, game):
        return Action(ActionType.USE_SKILL)
        # return Action(ActionType.DONT_USE_SKILL)

    def use_pro(self, game):
        return Action(ActionType.USE_SKILL)
        # return Action(ActionType.DONT_USE_SKILL)

    def use_bribe(self, game):
        return Action(ActionType.USE_BRIBE)

    def blood_lust_block_or_move(self, game):
        return Action(ActionType.START_BLOCK)

    def eat_thrall(self, game):
        position = game.get_available_actions()[0].positions[0]
        return Action(ActionType.SELECT_PLAYER, position)

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


def path_to_move_actions(game: botbowl.Game, path: Path) -> List[Action]:
    final_action_type = ActionType.MOVE
    active_team = game.state.available_actions[0].team
    player_at_target = game.get_player_at(path.get_last_step())

    if player_at_target is not None:
        if active_team is player_at_target.team and player_at_target.state.up:
            final_action_type = ActionType.HANDOFF
        elif game.get_opp_team(active_team) is player_at_target.team:
            if player_at_target.state.up:
                final_action_type = ActionType.BLOCK
            else:
                final_action_type = ActionType.FOUL
        else:
            raise Exception(f'Cant target {player_at_target} with this {Path}')

    final_action = Action(final_action_type, position=path.get_last_step())
    if game._is_action_allowed(final_action):
        return [final_action]
    else:
        return [Action(ActionType.MOVE, position=sq) for sq in path.steps[:-1]] + [final_action]


# Register MyScriptedBot
botbowl.register_bot('scripted', MyScriptedBot)

if __name__ == "__main__":

    # Uncomment to this to evaluate the bot against the random baseline

    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl-iii")
    config.competition_mode = False
    config.pathfinding_enabled = True
    # config = get_config("gym-7.json")
    # config = get_config("gym-5.json")
    # config = get_config("gym-3.json")
    ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)  # We don't need all the rules
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)

    # Play 10 games
    for i in range(10):
        home_agent = botbowl.make_bot('scripted')
        home_agent.name = "Scripted Bot"
        away_agent = botbowl.make_bot('random')
        away_agent.name = "Random Bot"
        config.debug_mode = False
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)
