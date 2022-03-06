"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains most of the model classes.
"""

from botbowl.core.util import get_data_path, Stack, compare_iterable
from botbowl.core.table import *
from botbowl.core.forward_model import Immutable, Reversible, CallableStep, treat_as_immutable, immutable_after_init

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import List, Optional, Set, Dict

import numpy as np
import uuid
import time
import pickle
from math import sqrt
import os


class ReplayStep:

    def __init__(self, game, num_reports):
        self.game = game
        self.num_reports = num_reports


class Replay:

    def __init__(self, replay_id, load=False):
        self.replay_id = replay_id
        self.steps = {}
        self.actions = {}
        self.reports = []
        self.idx = 0
        if load:
            filename = get_data_path('replays') + "/" + replay_id + ".rep"
            replay = pickle.load(open(filename, "rb"))
            self.steps = replay.steps
            self.actions = replay.actions
            self.reports = replay.reports
            self.idx = 0
            # Construct step reports
            for idx, step in self.steps.items():
                if step.num_reports == 0:
                    step.game['state']['reports'] = []
                else:
                    step.game['state']['reports'] = [report.to_json() for report in self.reports[:step.num_reports]]

    def record_step(self, game):
        state = game.to_json(ignore_reports=True)
        self.steps[self.idx] = ReplayStep(state, len(game.state.reports))
        self.idx += 1

    def record_action(self, action):
        self.actions[self.idx] = action.to_json() if action is not None else None
        self.idx += 1

    def dump(self, game):
        replay_id = game.game_id
        self.reports = game.state.reports
        name = self.steps[0].game['home_agent']['name'] + "_VS_" + self.steps[0].game['away_agent']['name'] + "_" + str(
            replay_id)
        directory = get_data_path('replays')
        if not os.path.exists(directory):
            os.mkdir(directory)
        filename = os.path.join(directory, f"{name}.rep")
        print(f"Saving replay to {filename}")
        pickle.dump(self, open(filename, "wb"))
        print(f"Replay saved to {filename}")

    def next(self):
        if len(self.steps) == 0 or self.idx + 1 >= len(self.steps):
            return None
        self.idx += 1
        while self.idx not in self.steps:
            # print(self.actions[self.idx])
            self.idx += 1
        return self.steps[self.idx]

    def prev(self):
        if len(self.steps) == 0 or self.idx - 1 < 0:
            return None
        self.idx -= 1
        while self.idx not in self.steps:
            # print(self.actions[self.idx])
            self.idx -= 1
        return self.steps[self.idx]

    def first(self):
        if len(self.steps) == 0:
            return None
        self.idx = 0
        return self.steps[self.idx]

    def last(self):
        if len(self.steps) == 0:
            return None
        self.idx = max(self.steps.keys())
        return self.steps[self.idx]

    def to_json(self):
        return {
            'replay_id': self.replay_id,
            'steps': {idx: step.game for idx, step in self.steps.items()},
            'actions': {idx: action for idx, action in self.actions.items()}
        }

    def to_simple_json(self):
        return {
            'game': self.steps[0]
        }


class TimeLimits:

    def __init__(self, turn, secondary, init, end):
        self.turn = turn
        self.secondary = secondary
        self.init = init
        self.end = end

    def to_json(self):
        return {
            'turn': self.turn,
            'secondary': self.secondary,
            'init': self.init,
            'end': self.end
        }


class Configuration:
    name: str
    arena: Optional['TwoPlayerArena']
    ruleset: Optional['RuleSet']
    roster_size: int
    pitch_max: int
    pitch_min: int
    scrimmage_min: int
    wing_max: int
    rounds: int
    kick_off_table: bool
    fast_mode: bool
    debug_mode: bool
    competition_mode: bool
    kick_scatter_distance: str
    offensive_formations: List['Formation']
    defensive_formations: List['Formation']
    time_limits: Optional[TimeLimits]
    pathfinding_enabled: bool
    pathfinding_directly_to_adjacent: bool

    def __init__(self):
        self.name = "Default"
        self.arena = None
        self.ruleset = None
        self.roster_size = 16
        self.pitch_max = 11
        self.pitch_min = 3
        self.scrimmage_min = 3
        self.wing_max = 2
        self.rounds = 8
        self.kick_off_table = True
        self.fast_mode = False
        self.debug_mode = False
        self.competition_mode = False
        self.kick_scatter_distance = "d6"
        self.offensive_formations = []
        self.defensive_formations = []
        self.time_limits = None
        self.pathfinding_enabled = True
        self.pathfinding_directly_to_adjacent = True


class PlayerState(Reversible):
    up: bool
    in_air: bool
    used: bool
    spp_earned: int
    moves: int
    stunned: bool
    bone_headed: bool
    hypnotized: bool
    really_stupid: bool
    heated: bool
    knocked_out: bool
    ejected: bool
    injuries_gained: List
    wild_animal: bool
    taken_root: bool
    blood_lust: bool
    picked_up: bool
    used_skills: Set[Skill]
    squares_moved: List['Square']
    has_blocked: bool

    def __init__(self):
        super().__init__()
        self.up = True
        self.in_air = False
        self.used = False
        self.spp_earned = 0
        self.moves = 0
        self.stunned = False
        self.bone_headed = False
        self.hypnotized = False
        self.really_stupid = False
        self.heated = False
        self.knocked_out = False
        self.ejected = False
        self.injuries_gained = []
        self.wild_animal = False
        self.taken_root = False
        self.blood_lust = False
        self.picked_up = False
        self.used_skills = set()
        self.squares_moved = []
        self.has_blocked = False
        self.failed_nega_trait_this_turn = False

    def to_json(self):
        return {
            'up': self.up,
            'in_air': self.in_air,
            'used': self.used,
            'stunned': self.stunned,
            'knocked_out': self.knocked_out,
            'bone_headed': self.bone_headed,
            'hypnotized': self.hypnotized,
            'really_stupid': self.really_stupid,
            'heated': self.heated,
            'ejected': self.ejected,
            'spp_earned': self.spp_earned,
            'moves': self.moves,
            'injuries_gained': [injury.name for injury in self.injuries_gained],
            'squares_moved': [square.to_json() for square in self.squares_moved],
            'wild_animal': self.wild_animal,
            'taken_root': self.taken_root,
            'blood_lust': self.blood_lust,
            'has_blocked': self.has_blocked,
            'failed_nega_trait_this_turn': self.failed_nega_trait_this_turn
        }

    def reset(self):
        self.up = True
        self.used = False
        self.in_air = False
        self.stunned = False
        self.bone_headed = False
        self.wild_animal = False
        self.taken_root = False
        self.hypnotized = False
        self.really_stupid = False
        self.heated = False
        self.blood_lust = False
        self.picked_up = False
        self.used_skills.clear()
        self.squares_moved.clear()
        self.failed_nega_trait_this_turn = False

    def reset_turn(self):
        self.moves = 0
        self.used = False
        self.used_skills.clear()
        self.failed_nega_trait_this_turn = False
        self.squares_moved.clear()

    always_show_attr = ['up']
    show_if_true_attr = ['used', 'stunned', 'bone_headed', 'hypnotized', 'really_stupid', 'heated', 'knocked_out',
                         'ejected', 'wild_animal', 'taken_root', 'blood_lust', 'picked_up', 'has_blocked']

    def __repr__(self):
        states_to_show = [f"{attr}={getattr(self, attr)}" for attr in PlayerState.always_show_attr] + \
                         [f"{attr}=True" for attr in PlayerState.show_if_true_attr if getattr(self, attr)]
        return f'PlayerState({", ".join(states_to_show)})'


class Agent:
    name: str
    human: bool
    agent_id: str

    def __init__(self, name, human=False, agent_id=None):
        if agent_id is not None:
            self.agent_id = agent_id
        else:
            self.agent_id = str(uuid.uuid1())
        self.name = name
        self.human = human

    def to_json(self):
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'human': self.human
        }

    def __eq__(self, other):
        if other is None or self is None:
            return False
        return self.agent_id == other.agent_id

    def __hash__(self):
        return self.agent_id

    def new_game(self, game, team):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def act(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def end_game(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")


class TeamState(Reversible):
    bribes: int
    babes: int
    apothecaries: int
    wizard_available: bool
    masterchef: bool
    score: int
    turn: int
    rerolls_start: int
    rerolls: int
    ass_coaches: int
    cheerleaders: int
    fame: int
    reroll_used: bool
    time_violation: int

    def __init__(self, team):
        super().__init__()
        self.bribes = 0
        self.babes = 0
        self.apothecaries = team.apothecaries
        self.wizard_available = False
        self.masterchef = False
        self.score = 0
        self.turn = 0
        self.rerolls_start = team.rerolls
        self.rerolls = team.rerolls
        self.ass_coaches = team.ass_coaches
        self.cheerleaders = team.cheerleaders
        self.fame = 0
        self.reroll_used = False
        self.time_violation = 0

    def to_json(self):
        return {
            'bribes': self.bribes,
            'babes': self.babes,
            'apothecaries': self.apothecaries,
            'masterchef': self.masterchef,
            'score': self.score,
            'turn': self.turn,
            'rerolls_start': self.rerolls_start,
            'rerolls': self.rerolls,
            'ass_coaches': self.ass_coaches,
            'cheerleaders': self.cheerleaders,
            'fame': self.fame,
            'reroll_used': self.reroll_used,
            'time_violation': self.time_violation
        }

    def reset_turn(self):
        self.reroll_used = False

    def use_reroll(self):
        self.rerolls -= 1
        self.reroll_used = True


class Clock:
    seconds: int
    started_at: float
    paused_at: float
    paused_seconds: int
    is_primary: bool
    team: 'Team'

    def __init__(self, team, seconds, is_primary=False):
        self.seconds = seconds
        self.started_at = time.time()
        self.paused_at = None
        self.paused_seconds = 0
        self.is_primary = is_primary
        self.team = team

    def is_running(self) -> bool:
        return self.paused_at is None

    def pause(self) -> None:
        assert self.paused_at is None
        self.paused_at = time.time()

    def resume(self) -> None :
        assert self.paused_at is not None
        now = time.time()
        self.paused_seconds += now - self.paused_at
        self.paused_at = None

    def get_running_time(self) -> float:
        now = time.time()
        if self.is_running():
            return now - self.started_at - self.paused_seconds
        else:
            return self.paused_at - self.started_at - self.paused_seconds

    def get_ratio_done(self):
        return self.get_running_time() / self.seconds

    def get_seconds_left(self):
        return self.seconds - self.get_running_time()

    def is_done(self):
        return self.is_running() and self.get_running_time() > self.seconds

    def to_json(self):
        return {
            'is_running': self.is_running(),
            'running_time': self.get_running_time(),
            'ratio_done': self.get_ratio_done(),
            'is_done': self.is_done(),
            'seconds': self.seconds,
            'seconds_left': self.get_seconds_left(),
            'is_primary': self.is_primary,
            'team_id': self.team.team_id,
            'paused_seconds': self.paused_seconds,
            'started_at': self.started_at
        }


class GameState(Reversible):
    stack: Stack
    reports: List['Outcome']
    half: int
    round: int
    coin_toss_winner: Optional['Team']
    kicking_first_half: Optional['Team']
    receiving_first_half: Optional['Team']
    kicking_this_drive: Optional['Team']
    receiving_this_drive: Optional['Team']
    current_team: Optional['Team']
    teams: List['Team']
    home_team: 'Team'
    away_team: 'Team'
    team_by_id: Dict[str, 'Team']
    player_by_id: Dict[str, 'Player']
    team_by_player_id: Dict[str, 'Team']

    pitch: 'Pitch'
    dugouts: Dict[str, 'Dugout']
    weather: WeatherType
    gentle_gust: bool
    turn_order: List['Team']
    spectators: int
    active_player: Optional['Player']
    game_over: bool
    available_actions: List['ActionChoice']
    clocks: List[Clock]
    rerolled_procs: Set['Procedure']
    player_action_type: Optional[ActionType]

    def __init__(self, game, home_team, away_team):
        super().__init__(ignored_keys=["clocks"])
        self.stack = Stack()
        self.reports = []
        self.half = 1
        self.round = 0
        self.coin_toss_winner = None
        self.kicking_first_half = None
        self.receiving_first_half = None
        self.kicking_this_drive = None
        self.receiving_this_drive = None
        self.current_team = None
        self.teams = [home_team, away_team]
        self.home_team = home_team
        self.away_team = away_team
        self.team_by_id = {team.team_id: team for team in self.teams}
        self.player_by_id = {}
        self.team_by_player_id = {}
        for team in self.teams:
            for player in team.players:
                self.team_by_player_id[player.player_id] = team
                self.player_by_id[player.player_id] = player
        self.pitch = Pitch(game.arena.width, game.arena.height)
        self.dugouts = {team.team_id: Dugout(team) for team in self.teams}
        self.weather = WeatherType.NICE
        self.gentle_gust = False
        self.turn_order = []
        self.spectators = 0
        self.active_player = None
        self.game_over = False
        self.available_actions = []
        self.clocks = []
        self.rerolled_procs = set()
        self.player_action_type = None

    def compare(self, other):
        """
        :param other, another GameState
        Compares all relevant values of the two game states and returns a list strings containing all the
        differences. Empty list if all relevant values are equal.
        """
        errors = compare_iterable(self.to_json(ignore_clocks=True),
                                  other.to_json(ignore_clocks=True), path="state")

        errors.extend(compare_iterable(self.stack.items, other.stack.items, path="state.stack.items"))

        return errors

    def to_json(self, ignore_reports=False, ignore_clocks=False):
        return {
            'half': self.half,
            'kicking_first_half': self.kicking_first_half.team_id if self.kicking_first_half is not None else None,
            'receiving_first_half': self.receiving_first_half.team_id if self.receiving_first_half is not None else None,
            'kicking_this_drive': self.kicking_this_drive.team_id if self.kicking_this_drive is not None else None,
            'receiving_this_drive': self.receiving_this_drive.team_id if self.receiving_this_drive is not None else None,
            'pitch': self.pitch.to_json(),
            'home_dugout': self.dugouts[self.home_team.team_id].to_json(),
            'away_dugout': self.dugouts[self.away_team.team_id].to_json(),
            'home_team': self.home_team.to_json(),
            'away_team': self.away_team.to_json(),
            'game_over': self.game_over,
            'weather': self.weather.name,
            'gentle_gust': self.gentle_gust,
            'available_actions': [action.to_json() for action in self.available_actions],
            'reports': [report.to_json() for report in self.reports] if not ignore_reports else [],
            'current_team_id': self.current_team.team_id if self.current_team is not None else None,
            'round': self.round,
            'spectators': self.spectators,
            'active_player_id': self.active_player.player_id if self.active_player is not None else None,
            'clocks': [clock.to_json() for clock in self.clocks] if not ignore_clocks else None,
            'player_action_type': self.player_action_type.name if self.player_action_type is not None else None
        }


class Pitch(Reversible):
    balls: List['Ball']
    bomb: Optional['Bomb']
    board: List[List[Optional['Player']]]
    squares: List[List['Square']]
    height: int
    width: int

    def __init__(self, width, height):
        super().__init__(ignored_keys=["board", "squares"])
        self.balls = []
        self.bomb = None
        self.board = [[None for x in range(width)] for y in range(height)]
        self.squares = [[Square(x, y, x == 0 or x == width-1 or y == 0 or y == height-1)
                         for x in range(width)] for y in range(height)]
        self.height = len(self.board)
        self.width = len(self.board[0])

    def to_json(self):
        board = []
        for y in range(len(self.board)):
            row = []
            for x in range(len(self.board[0])):
                row.append(self.board[y][x].player_id if self.board[y][x] is not None else None)
            board.append(row)
        return {
            'board': board,
            'balls': [ball.to_json() for ball in self.balls],
            'bomb': self.bomb.to_json() if self.bomb else None
        }


@immutable_after_init
class ActionChoice:
    action_type: ActionType
    positions: List['Square']
    players: List['Player']
    team: 'Team'
    rolls: List[int]
    block_dice: List
    disabled: bool
    skill: Optional[Skill]
    paths: List['Path']

    def __init__(self, action_type, team, positions=None, players=None, rolls=None, block_dice=None, skill=None,
                 paths=None, disabled=False):
        self.action_type = action_type
        self.positions = [] if positions is None else positions
        self.players = [] if players is None else players
        self.team = team
        self.rolls = [] if rolls is None else rolls
        self.block_dice = [] if block_dice is None else block_dice
        self.disabled = disabled
        self.skill = skill
        self.paths = [] if paths is None else paths

    def __repr__(self):
        return f"ActionChoice({self.action_type}, len(positions)={len(self.positions)}, " \
               f"len(players)={len(self.players)}, len(paths)={len(self.paths)})"

    def to_json(self):
        return {
            'action_type': self.action_type.name,
            'positions': [position.to_json() if position is not None else None for position in self.positions],
            'team_id': self.team.team_id if self.team is not None else None,
            "rolls": self.rolls,
            "block_dice": self.block_dice,
            'player_ids': [player.player_id for player in self.players],
            "skill": self.skill.name if self.skill is not None else None,
            "disabled": self.disabled,
            "paths": [
                {
                    "steps": [square.to_json() for square in path.steps],
                    "rolls": path.rolls,
                    "prob": path.prob,
                    "block_dice": path.block_dice,
                    "foul_roll": path.foul_roll,
                    "handoff_roll": path.handoff_roll
                } for path in self.paths
            ]
        }


@treat_as_immutable
class Action:
    action_type: ActionType
    position: Optional['Square']
    player: Optional['Player']

    def __init__(self, action_type, position=None, player=None):
        self.action_type = action_type
        self.position = position
        self.player = player

    def __repr__(self):
        pos_str = f", position={self.position}" if self.position is not None else ""
        player_str = f", player={self.player}" if self.player is not None else ""
        return f"Action({self.action_type}{pos_str}{player_str})"

    def to_json(self):
        return {
            'action_type': self.action_type.name,
            'position': self.position.to_json() if self.position is not None else None,
            'player_id': self.player.player_id if self.player is not None else None
        }


class TwoPlayerArena:
    home_tiles = [Tile.HOME, Tile.HOME_TOUCHDOWN, Tile.HOME_WING_LEFT, Tile.HOME_WING_RIGHT, Tile.HOME_SCRIMMAGE]
    away_tiles = [Tile.AWAY, Tile.AWAY_TOUCHDOWN, Tile.AWAY_WING_LEFT, Tile.AWAY_WING_RIGHT, Tile.AWAY_SCRIMMAGE]
    scrimmage_tiles = [Tile.HOME_SCRIMMAGE, Tile.AWAY_SCRIMMAGE]
    wing_right_tiles = [Tile.HOME_WING_RIGHT, Tile.AWAY_WING_RIGHT]
    wing_left_tiles = [Tile.HOME_WING_LEFT, Tile.AWAY_WING_LEFT]
    home_td_tiles = [Tile.HOME_TOUCHDOWN]
    away_td_tiles = [Tile.AWAY_TOUCHDOWN]

    def __init__(self, board):
        self.board = board
        self.width = len(board[0])
        self.height = len(board)
        self.json = None

    def is_in_opp_endzone(self, position, home):
        if home:
            return self.board[position.y][position.x] == Tile.AWAY_TOUCHDOWN
        else:
            return self.board[position.y][position.x] == Tile.HOME_TOUCHDOWN

    def to_json(self):
        if self.json is not None:
            return self.json

        board = []
        for r in self.board:
            row = []
            for tile in r:
                row.append(tile.name if tile is not None else None)
            board.append(row)
        self.json = {
            'board': board
        }
        return self.json


class Die(ABC):
    value: str

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def to_json(self):
        pass


@treat_as_immutable
class DiceRoll:
    dice: List[Die]
    modifiers: int
    target: Optional[int]
    d68: bool
    roll_type: RollType
    target_higher: int
    target_lower: int
    highest_succeed: bool
    lowest_fail: bool

    def __init__(self, dice, modifiers=0, target=None, d68=False, roll_type=RollType.AGILITY_ROLL, target_higher=True,
                 target_lower=False, highest_succeed=True, lowest_fail=True):
        self.dice = dice
        self.sum = 0
        self.d68 = d68
        self.target = target
        self.modifiers = modifiers
        self.roll_type = roll_type
        self.target_higher = target_higher
        self.target_lower = target_lower
        self.highest_succeed = highest_succeed
        self.lowest_fail = lowest_fail
        # Roll dice
        for d in self.dice:
            if not isinstance(d, BBDie):
                if d68 and isinstance(d, D6):
                    self.sum += d.get_value() * 10
                else:
                    self.sum += d.get_value()

    def __repr__(self):
        return f"DiceRoll(dice={self.dice})"

    def to_json(self):
        dice = []
        for die in self.dice:
            dice.append(die.to_json())
        return {
            'dice': dice,
            'sum': self.sum,
            'target': self.target,
            'modifiers': self.modifiers,
            'modified_target': self.modified_target(),
            'result': self.get_result(),
            'roll_type': self.roll_type.name,
            'target_higher': self.target_higher,
            'target_lower': self.target_lower,
            'highest_succeed': self.highest_succeed,
            'lowest_fail': self.lowest_fail
        }

    def modified_target(self):
        if self.target is not None:
            return max(1 * len(self.dice), min(6 * len(self.dice), self.target - self.modifiers))
        return None

    def contains(self, value):
        for die in self.dice:
            if die.get_value() == value:
                return True
        return False

    def get_values(self):
        return [d.get_value() for d in self.dice]

    def get_sum(self):
        return self.sum

    def get_result(self):
        return self.sum + self.modifiers

    def is_d6_success(self):
        if self.lowest_fail and self.sum == 1 * len(self.dice):
            return False

        if self.highest_succeed and self.sum == 6 * len(self.dice):
            return True

        if self.target_higher:
            return self.sum + self.modifiers >= self.target
        elif self.target_lower:
            return self.sum + self.modifiers <= self.target
        else:
            return self.sum + self.modifiers == self.target

    def same(self):
        value = None
        for die in self.dice:
            if value is None or die.get_value() == value:
                value = die.get_value()
                continue
            return False
        return True


class D3(Die):
    FixedRolls = []

    @staticmethod
    def fix(value):
        if 1 <= value <= 3:
            D3.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of D3 must be between 1 and 3")

    def __init__(self, rnd):
        if len(D3.FixedRolls) > 0:
            self.value = D3.FixedRolls.pop(0)
        else:
            self.value = rnd.randint(1, 4)

    def __repr__(self):
        return f"D3({self.value})"

    def get_value(self):
        return self.value

    def to_json(self):
        return {
            'die_type': 'D3',
            'result': self.value
        }


class D6(Die, Immutable):
    FixedRolls = []

    TWO_PROBS = {
        2: (1 / 6 * 1 / 6),
        3: 2 * (1 / 6 * 1 / 6),
        4: 3 * (1 / 6 * 1 / 6),
        5: 4 * (1 / 6 * 1 / 6),
        6: 5 * (1 / 6 * 1 / 6),
        7: 6 * (1 / 6 * 1 / 6),
        8: 5 * (1 / 6 * 1 / 6),
        9: 4 * (1 / 6 * 1 / 6),
        10: 3 * (1 / 6 * 1 / 6),
        11: 2 * (1 / 6 * 1 / 6),
        12: 1 * (1 / 6 * 1 / 6)
    }

    @staticmethod
    def fix(value):
        if 1 <= value <= 6:
            D6.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of D6 must be between 1 and 6")

    def __init__(self, rnd):
        if len(D6.FixedRolls) > 0:
            self.value = D6.FixedRolls.pop(0)
        else:
            self.value = rnd.randint(1, 7)

    def __repr__(self):
        return f"D6({self.value})"

    def get_value(self):
        return self.value

    def to_json(self):
        return {
            'die_type': 'D6',
            'result': self.value
        }


class D8(Die, Immutable):
    FixedRolls = []

    @staticmethod
    def fix(value):
        if 1 <= value <= 8:
            D8.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of D8 must be between 1 and 8")

    def __init__(self, rnd):
        if len(D8.FixedRolls) > 0:
            self.value = D8.FixedRolls.pop(0)
        else:
            self.value = rnd.randint(1, 9)

    def __repr__(self):
        return f"D8({self.value})"

    def get_value(self):
        return self.value

    def to_json(self):
        return {
            'die_type': 'D8',
            'result': self.value
        }


class BBDie(Die, Immutable):
    value: BBDieResult
    FixedRolls = []

    @staticmethod
    def fix(value):
        if type(value) == BBDieResult:
            BBDie.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of BBDie must be a BBDieResult")

    @staticmethod
    def clear_fixes():
        BBDie.FixedRolls.clear()

    def __init__(self, rnd):
        if len(BBDie.FixedRolls) > 0:
            self.value = BBDie.FixedRolls.pop(0)
        else:
            r = rnd.randint(1, 7)
            if r == 6:
                r = 3
            self.value = BBDieResult(r)

    def __repr__(self):
        return f"BBDie({self.value})"

    def get_value(self) -> BBDieResult:
        return self.value

    def to_json(self):
        return {
            'die_type': 'BB',
            'result': self.value.name
        }


class Dugout(Reversible):

    def __init__(self, team):
        super().__init__()
        self.team = team
        self.reserves = []
        self.kod = []
        self.casualties = []
        self.dungeon = []  # Ejected

    def to_json(self):
        return {
            'team_id': self.team.team_id,
            'reserves': [player.player_id for player in self.reserves],
            'kod': [player.player_id for player in self.kod],
            'casualties': [player.player_id for player in self.casualties],
            'dungeon': [player.player_id for player in self.dungeon]
        }


class Role:

    def __init__(self, name, races, ma, st, ag, av, skills, cost, feeder, n_skill_sets=None, d_skill_sets=None,
                 star_player=False):
        self.name = name
        self.races = races
        self.ma = ma
        self.st = st
        self.ag = ag
        self.av = av
        self.skills = skills
        self.cost = cost
        self.feeder = feeder
        self.n_skill_sets = n_skill_sets if n_skill_sets is not None else []
        self.d_skill_sets = d_skill_sets if d_skill_sets is not None else []
        self.star_player = star_player


class Piece:
    position: 'Square'

    def __init__(self, position=None):
        self.position = position

    def is_catchable(self):
        return False


class Catchable(Piece):

    def __init__(self, position, on_ground=True, is_carried=False):
        super().__init__(position)
        self.on_ground = on_ground
        self.is_carried = is_carried

    def move(self, x, y):
        # This is unfortunately way slower than below, but Square is Immutable
        self.position = Square(self.position.x + x, self.position.y + y)
        # self.position.x += x
        # self.position.y += y

    def move_to(self, position):
        self.position = position

    def to_json(self):
        return {
            'position': self.position.to_json() if self.position is not None else None,
            'on_ground': self.on_ground,
            'is_carried': self.is_carried
        }

    def is_catchable(self):
        return True


class Ball(Catchable, Reversible):

    def __init__(self, position, on_ground=True, is_carried=False):
        Reversible.__init__(self, ["position"])
        Catchable.__init__(self, position, on_ground, is_carried)

    def move(self, x, y):
        if self.trajectory_initialized():
            log_entry = CallableStep(self, Catchable.move, (x, y), Catchable.move, (-x, -y))
            self.log_this(log_entry)

        super().move(x, y)

    def move_to(self, position):

        if self.trajectory_initialized():
            log_entry = CallableStep(self, Catchable.move_to, (copy(position),), Catchable.move_to, (self.position,))
            self.log_this(log_entry)

        super().move_to(position)

    def __repr__(self):
        return f"Ball(position={self.position if self.position is not None else 'None'}, " \
               f"on_ground={self.on_ground}, " \
               f"is_carried={self.is_carried})"


class Bomb(Catchable, Reversible):

    def __init__(self, position, on_ground=True, is_carried=False):
        super().__init__(position, on_ground, is_carried)


class Player(Piece, Reversible):
    player_id: str
    role: Role
    nr: int
    team: 'Team'
    extra_ma: int
    extra_st: int
    extra_ag: int
    extra_av: int
    injuries: List

    def __init__(self, player_id, role, name, nr, team, extra_skills=None, extra_ma=0, extra_st=0, extra_ag=0,
                 extra_av=0, niggling_injuries=0, mng=False, spp=0, injuries=None, position=None):
        Reversible.__init__(self, ignored_keys=["position", "role"])
        super().__init__(position)
        self.player_id = player_id
        self.role = role
        self.name = name
        self.nr = nr
        self.team = team
        self.extra_skills = extra_skills if extra_skills is not None else []
        self.extra_ma = extra_ma
        self.extra_st = extra_st
        self.extra_ag = extra_ag
        self.extra_av = extra_av
        self.injuries = [] if injuries is None else injuries
        for _ in range(niggling_injuries):
            self.injuries.append(CasualtyEffect.NIGGLING)
        self.mng = mng
        self.spp = spp
        self.state = PlayerState()

    def to_json(self):
        return {
            'player_id': self.player_id,
            'name': self.name,
            'role': self.role.name,
            'team_id': self.team.team_id,
            'nr': self.nr,
            'role_skills': [skill.name for skill in self.role.skills],
            'extra_skills': [skill.name for skill in self.extra_skills],
            'ma': self.get_ma(),
            'st': self.get_st(),
            'ag': self.get_ag(),
            'av': self.get_av(),
            'injuries': [injury.name for injury in self.injuries],
            'mng': self.mng,
            'spp': self.spp,
            'state': self.state.to_json(),
            'position': self.position.to_json() if self.position is not None else None
        }

    def get_ag(self):
        ag = self.role.ag + self.extra_ag - self.injuries.count(CasualtyEffect.AG) - self.state.injuries_gained.count(
            CasualtyEffect.AG)
        ag = max(self.role.ag - 2, ag)
        ag = max(1, ag)
        ag = min(10, ag)
        return ag

    def get_st(self):
        st = self.role.st + self.extra_st - self.injuries.count(CasualtyEffect.ST) - self.state.injuries_gained.count(
            CasualtyEffect.ST)
        st = max(self.role.st - 2, st)
        st = max(1, st)
        st = min(10, st)
        return st

    def get_ma(self):
        if self.state.taken_root:
            return 0
        else:
            ma = self.role.ma + self.extra_ma - self.injuries.count(
                CasualtyEffect.MA) - self.state.injuries_gained.count(CasualtyEffect.MA)
            ma = max(self.role.ma - 2, ma)
            ma = max(1, ma)
            ma = min(10, ma)
            return ma

    def get_av(self):
        av = self.role.av + self.extra_av - - self.injuries.count(CasualtyEffect.AV) - self.state.injuries_gained.count(
            CasualtyEffect.AV)
        av = max(self.role.av - 2, av)
        av = max(1, av)
        av = min(10, av)
        return av

    def num_niggling_injuries(self):
        return self.injuries.count(CasualtyEffect.NIGGLING)

    def has_skill(self, skill):
        return skill in self.get_skills()

    def has_used_skill(self, skill):
        return skill in self.state.used_skills

    def can_use_skill(self, skill):
        return self.has_skill(skill) and not self.has_used_skill(skill)

    def use_skill(self, skill):
        return self.state.used_skills.add(skill)

    def get_skills(self):
        return self.role.skills + self.extra_skills

    def num_gfis_left(self):
        return self.num_moves_left(include_gfi=True) - self.num_moves_left(include_gfi=False)

    def has_tackle_zone(self):
        if self.has_skill(Skill.TITCHY):
            return False
        if self.state.up and not self.state.bone_headed and not self.state.hypnotized and not self.state.really_stupid:
            return True
        return False

    def can_catch(self):
        return self.state.up and not self.state.bone_headed and not self.state.hypnotized and \
               not self.state.really_stupid and Skill.NO_HANDS not in self.get_skills()

    def can_assist(self):
        return self.state.up and not self.state.bone_headed and not self.state.hypnotized and not self.state.really_stupid

    def num_moves_left(self, include_gfi: bool = False):
        if self.state.taken_root or self.state.used or self.state.stunned:
            return 0
        moves = self.get_ma() - self.state.moves
        if include_gfi:
            if self.has_skill(Skill.SPRINT):
                moves = moves + 3
            else:
                moves = moves + 2
        return max(0, moves)

    def __eq__(self, other):
        return isinstance(other, Player) and other.player_id == self.player_id

    def __hash__(self):
        return self.player_id.__hash__()

    def place_prone(self):
        self.state.up = False
        self.state.taken_root = False

    def __repr__(self):
        return f"Player(position={self.position if self.position is not None else 'None'}, {self.role.name}, state={self.state})"


@immutable_after_init
class Square:
    x: int
    y: int
    _out_of_bounds: Optional[bool]

    def __init__(self, x: int, y: int, _out_of_bounds=None):
        self.x = x
        self.y = y
        self._out_of_bounds = _out_of_bounds

    @property
    def out_of_bounds(self):
        assert self._out_of_bounds is not None  # This assertion can be removed when we trust the unit tests more
        return self._out_of_bounds

    def to_json(self):
        return {
            'x': self.x,
            'y': self.y
        }

    def __eq__(self, other):
        if other is None or self is None:
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self.x * 7 + self.y * 13

    def distance(self, other, manhattan=False, flight=False):
        if manhattan:
            return abs(other.x - self.x) + abs(other.y - self.y)
        elif flight:
            return sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
        else:
            return max(abs(other.x - self.x), abs(other.y - self.y))

    def is_adjacent(self, other, manhattan=False):
        return self.distance(other, manhattan) == 1

    def __repr__(self):
        return f"Square({self.x}, {self.y}, out={self._out_of_bounds if self._out_of_bounds is not None else 'None'})"


class Race:

    def __init__(self, name, roles, reroll_cost, apothecary, stakes):
        self.name = name
        self.roles = roles
        self.reroll_cost = reroll_cost
        self.apothecary = apothecary
        self.stakes = stakes


class Team(Reversible):

    def __init__(self, team_id, name, race, players=None, treasury=0, apothecaries=0, rerolls=0, ass_coaches=0,
                 cheerleaders=0, fan_factor=0):
        super().__init__()
        self.team_id = team_id
        self.name = name
        self.race = race
        self.players = players if players is not None else []
        self.treasury = treasury
        self.apothecaries = apothecaries
        self.rerolls = rerolls
        self.fan_factor = fan_factor
        self.ass_coaches = ass_coaches
        self.cheerleaders = cheerleaders
        self.state = TeamState(self)

    def to_json(self):
        players = []
        players_by_id = {}
        for player in self.players:
            players.append(player.to_json())
            players_by_id[player.player_id] = player.to_json()
        return {
            'team_id': self.team_id,
            'name': self.name,
            'race': self.race,
            'treasury': self.treasury,
            'apothecaries': self.apothecaries,
            'rerolls': self.rerolls,
            'ass_coaches': self.ass_coaches,
            'cheerleaders': self.cheerleaders,
            'fan_factor': self.fan_factor,
            'players_by_id': players_by_id,
            'state': self.state.to_json()
        }

    def __eq__(self, other):
        return other is not None and other.team_id == self.team_id

    def __hash__(self):
        return self.team_id


@immutable_after_init
class Outcome:
    outcome_type: OutcomeType
    position: Optional[Square]
    player: Optional[Player]
    opp_player: Optional[Player]
    rolls: List[DiceRoll]
    team: Optional[Team]
    n: int
    skill: Optional[Skill]

    def __init__(self, outcome_type, position=None, player=None, opp_player=None, rolls=None, team=None, n=0,
                 skill=None):
        self.outcome_type = outcome_type
        self.position = position
        self.player = player
        self.opp_player = opp_player
        self.rolls = rolls if rolls is not None else []
        self.team = team
        self.n = n
        self.skill = skill

    def __repr__(self):
        pos_str = "" if self.position is None else f", position={self.position}"
        return f"Outcome({self.outcome_type}{pos_str}, rolls={self.rolls})"

    def to_json(self):
        rolls = []
        for roll in self.rolls:
            rolls.append(roll.to_json())
        return {
            'outcome_type': self.outcome_type.name,
            'pos': self.position.to_json() if self.position is not None else None,
            'player_id': self.player.player_id if self.player is not None else None,
            'opp_player_id': self.opp_player.player_id if self.opp_player is not None else None,
            'rolls': rolls,
            'team_id': self.team.team_id if self.team is not None else None,
            'n': self.n if self.n is not None else None,
            'skill': self.skill.name if self.skill is not None else None
        }


class Inducement:

    def __init__(self, name, cost, max_num, reduced=0):
        self.name = name
        self.cost = cost
        self.max_num = max_num
        self.reduced = reduced


class RuleSet:

    def __init__(self, name, races=[], star_players=[], inducements=[], spp_actions={}, spp_levels={}, improvements={},
                 se_start=0, se_interval=0, se_pace=0):
        self.name = name
        self.races = races
        self.star_players = star_players
        self.inducements = inducements
        self.spp_actions = spp_actions
        self.spp_levels = spp_levels
        self.improvements = improvements
        self.se_start = se_start
        self.se_interval = se_interval
        self.se_pace = se_pace

    def get_role(self, role, race):
        for r in self.races:
            if r.name == race:
                for p in r.roles:
                    if p.name == role:
                        return p
                raise Exception("Role not found in race: " + race + " -> " + role)
        raise Exception("Race not found: " + race)


class Formation(Immutable):

    def __init__(self, name, formation):
        self.name = name
        self.formation = formation

    def _get_player(self, players, t):
        if t == 'S':
            idx = np.argmax([player.get_st() + (0.5 if player.has_skill(Skill.BLOCK) else 0) - (
                0.5 if player.has_skill(Skill.SURE_HANDS) else 0) for player in players])
            return players[idx]
        if t == 'm':
            idx = np.argmax([player.get_ma() for player in players])
            return players[idx]
        if t == 'a':
            idx = np.argmax([player.get_ag() for player in players])
            return players[idx]
        if t == 'v':
            idx = np.argmax([player.get_av() for player in players])
            return players[idx]
        if t == 's':
            idx = np.argmax([1 if player.has_skill(Skill.SURE_HANDS) else 0 for player in players])
            return players[idx]
        if t == 'p':
            idx = np.argmax([1 if player.has_skill(Skill.PASS) else 0 for player in players])
            return players[idx]
        if t == 'c':
            idx = np.argmax([1 if player.has_skill(Skill.CATCH) else 0 for player in players])
            return players[idx]
        if t == 'b':
            idx = np.argmax([1 if player.has_skill(Skill.BLOCK) else 0 for player in players])
            return players[idx]
        if t == 'd':
            idx = np.argmax([1 if player.has_skill(Skill.DODGE) else 0 for player in players])
            return players[idx]
        if t == '0':
            idx = np.argmin([len(player.get_skills()) for player in players])
            return players[idx]
        if t == 'x':
            idx = np.argmax([1 if player.has_skill(Skill.BLOCK) else (
                0 if player.has_skill(Skill.PASS) or player.has_skill(Skill.CATCH) else 0.5) for player in players])
            return players[idx]
        return players[0]

    def actions(self, game, team):
        reorganize = game.get_procedure().reorganize

        home = team == game.state.home_team
        actions = []
        # Move all player on the pitch back to the reserves
        player_on_pitch = []
        for player in team.players:
            if player.position is not None:
                if not reorganize:
                    actions.append(Action(ActionType.PLACE_PLAYER, position=None, player=player))
                player_on_pitch.append(player)

        # Go through formation from scrimmage to touchdown zone
        players = player_on_pitch
        if not reorganize:
            players += game.get_reserves(team)

        positions_used = set()

        # setup on scrimmage
        for t in ['S', 's', 'p', 'b', 'c', 'm', 'a', 'v', 'd', '0', 'x']:
            for y in range(len(self.formation)):
                if len(players) == 0:
                    return actions
                x = len(self.formation[0]) - 1
                tp = self.formation[y][x]
                if tp == '-' or tp != t:
                    continue
                yy = y + 1
                xx = x + 1 if not home else game.arena.width - x - 2
                position = game.get_square(xx, yy)
                if not game.is_scrimmage(position) or position in positions_used:
                    continue
                player = self._get_player(players, t)
                players.remove(player)
                actions.append(Action(ActionType.PLACE_PLAYER, position=position, player=player))
                positions_used.add(position)

        for t in ['S', 's', 'p', 'b', 'c', 'm', 'a', 'v', 'd', '0', 'x']:
            for y in range(len(self.formation)):
                for x in reversed(range(len(self.formation[0]))):
                    if len(players) == 0:
                        return actions
                    tp = self.formation[y][x]
                    if tp == '-' or tp != t:
                        continue
                    yy = y + 1
                    xx = x + 1 if not home else game.arena.width - x - 2
                    position = game.get_square(xx, yy)
                    if game.is_scrimmage(position) or position in positions_used:
                        continue
                    player = self._get_player(players, t)
                    players.remove(player)
                    actions.append(Action(ActionType.PLACE_PLAYER, position=position, player=player))
                    positions_used.add(position)
        return actions

    def compare(self, other, path):
        """
        For testing purposes only
        """
        diff = []
        if self.name != other.name:
            diff.append(f"{path}.name: {self.name} _NotEqual_ {other.name}")

        formations_equal = all((self_a == other_a).all() for self_a, other_a in zip(self.formation, other.formation))

        if not formations_equal:
            diff.append(f"{path}.formation: <too big to display> _NotEqual_ <too big to display>")

        return diff
