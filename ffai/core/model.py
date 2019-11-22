"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains most of the model classes.
"""

from copy import copy, deepcopy
import numpy as np
import uuid
import time
import json
from math import sqrt
from ffai.core.util import *
from ffai.core.table import *


class Replay:

    def __init__(self, replay_id, load=False, simple=False):
        self.replay_id = replay_id
        self.steps = {}
        self.actions = {}
        self.idx = 0
        if load:
            filename = get_data_path('replays') + "/" + replay_id + ".rep"
            replay = json.load(open(filename, "r"))
            if simple:
                self.steps = replay['steps'][0]
                self.actions = []
            else:
                self.steps = replay['steps']
                self.actions = replay['actions']
            self.idx = 0

    def record_step(self, game):
        self.steps[self.idx] = game.to_json()
        self.idx += 1

    def record_action(self, action):
        self.actions[self.idx] = action.to_json() if action is not None else None
        self.idx += 1

    def dump(self, replay_id):
        name = self.steps[0]['home_agent']['name'] + "_VS_" + self.steps[0]['away_agent']['name'] + "_" + str(replay_id)
        filename = get_data_path('replays') + '/' + name + '.rep'
        json.dump(self.to_json(), open(filename, "w"))

    def next(self):
        if len(self.steps) == 0 or self.idx + 1 >= len(self.steps):
            return None
        self.idx += 1
        while self.idx not in self.steps:
            #print(self.actions[self.idx])
            self.idx += 1
        return self.steps[self.idx]

    def prev(self):
        if len(self.steps) == 0 or self.idx - 1 < 0:
            return None
        self.idx -= 1
        while self.idx not in self.steps:
            #print(self.actions[self.idx])
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
            'steps': {idx : step for idx, step in self.steps.items()},
            'actions': {idx : action for idx, action in self.actions.items()}
        }

    def to_simple_json(self):
        return {
            'game': self.steps[0]
        }


class TimeLimits:

    def __init__(self, game, turn, secondary, disqualification, init, end):
        self.game = game
        self.turn = turn
        self.secondary = secondary
        self.disqualification = disqualification
        self.init = init
        self.end = end

    def to_json(self):
         return {
            'game': self.game,
            'turn': self.turn,
            'secondary': self.secondary,
            'disqualification': self.disqualification,
            'init': self.init,
            'end': self.end
        }


class Configuration:

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


class PlayerState:

    def __init__(self):
        self.up = True
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
        self.casualty_effect = None
        self.casualty_type = None
        self.wild_animal = False
        self.used_skills = []
        self.squares_moved = []

    def to_json(self):
        return {
            'up': self.up,
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
            'casualty_type': self.casualty_type.name if self.casualty_type is not None else None,
            'casualty_effect': self.casualty_effect.name if self.casualty_effect is not None else None,
            'squares_moved': [square.to_json() for square in self.squares_moved],
            'wild_animal':self.wild_animal
        }

    def reset(self):
        self.up = True
        self.used = False
        self.stunned = False
        self.bone_headed = False
        self.hypnotized = False
        self.really_stupid = False
        self.heated = False
        self.used_skills.clear()
        self.squares_moved.clear()

    def reset_turn(self):
        self.moves = 0
        self.used = False
        self.used_skills.clear()
        self.squares_moved.clear()


class Agent:

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


class TeamState:

    def __init__(self, team):
        self.bribes = 0
        self.babes = 0
        self.apothecary_available = team.apothecary
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
            'apothecary_available': self.apothecary_available,
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

    def __init__(self, team, seconds, is_primary=False):
        self.seconds = seconds
        self.started_at = time.time()
        self.paused_at = None
        self.paused_seconds = 0
        self.is_primary = is_primary
        self.team = team

    def is_running(self):
        return self.paused_at is None

    def pause(self):
        assert self.paused_at is None
        self.paused_at = time.time()

    def resume(self):
        assert self.paused_at is not None
        now = time.time()
        self.paused_seconds += now - self.paused_at
        self.paused_at = None

    def get_running_time(self):
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


class GameState:

    def __init__(self, game, home_team, away_team):
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

    def to_json(self):
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
            'reports': [report.to_json() for report in self.reports],
            'current_team_id': self.current_team.team_id if self.current_team is not None else None,
            'round': self.round,
            'spectators': self.spectators,
            'active_player_id': self.active_player.player_id if self.active_player is not None else None,
            'clocks': [ clock.to_json() for clock in self.clocks ]
        }


class Pitch:

    range = [-1, 0, 1]

    def __init__(self, width, height):
        self.balls = []
        self.board = []
        self.squares = []
        for y in range(height):
            self.board.append([])
            self.squares.append([])
            for x in range(width):
                self.board[y].append(None)
                self.squares[y].append(Square(x, y))
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
            'balls': [ball.to_json() for ball in self.balls]
        }


class ActionChoice:

    def __init__(self, action_type, team, positions=None, players=None, rolls=None, block_rolls=None, agi_rolls=None, disabled=False):
        self.action_type = action_type
        self.positions = [] if positions is None else positions
        self.players = [] if players is None else players
        self.team = team
        self.rolls = [] if rolls is None else rolls
        self.block_rolls = [] if block_rolls is None else block_rolls
        self.disabled = disabled
        self.agi_rolls = [] if agi_rolls is None else agi_rolls

    def to_json(self):
        return {
            'action_type': self.action_type.name,
            'positions': [position.to_json() if position is not None else None for position in self.positions],
            'team_id': self.team.team_id if self.team is not None else None,
            "rolls": self.rolls,
            "block_rolls": self.block_rolls,
            "agi_rolls": self.agi_rolls,
            'player_ids': [player.player_id for player in self.players],
            "disabled": self.disabled
        }


class Action:

    def __init__(self, action_type, position=None, player=None):
        self.action_type = action_type
        self.position = position
        self.player = player

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


class Die:

    def get_value(self):
        Exception("Method not implemented")


class DiceRoll:

    def __init__(self, dice, modifiers=0, target=None, d68=False, roll_type=RollType.AGILITY_ROLL):
        self.dice = dice
        self.sum = 0
        self.d68 = d68
        self.target = target
        self.modifiers = modifiers
        self.roll_type = roll_type
        # Roll dice
        for d in self.dice:
            if not isinstance(d, BBDie):
                if d68 and isinstance(d, D6):
                    self.sum += d.get_value() * 10
                else:
                    self.sum += d.get_value()

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
            'roll_type': self.roll_type.name
        }

    def modified_target(self):
        if self.target is not None:
            return max(1*len(self.dice), min(6*len(self.dice), self.target - self.modifiers))
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
        if self.sum == 1:
            return False

        if self.sum == 6:
            return True

        return self.sum + self.modifiers >= self.target

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
    def fix_result(value):
        if 1 <= value <= 3:
            D3.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of D3 must be between 1 and 3")

    def __init__(self, rnd):
        if len(D3.FixedRolls) > 0:
            self.value = D3.FixedRolls.pop(0)
        else:
            self.value = rnd.randint(1, 4)

    def get_value(self):
        return self.value

    def to_json(self):
        return {
            'die_type': 'D3',
            'result': self.value
        }


class D6(Die):

    FixedRolls = []

    @staticmethod
    def fix_result(value):
        if 1 <= value <= 6:
            D6.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of D6 must be between 1 and 6")

    def __init__(self, rnd):
        if len(D6.FixedRolls) > 0:
            self.value = D6.FixedRolls.pop(0)
        else:
            self.value = rnd.randint(1, 7)

    def get_value(self):
        return self.value

    def to_json(self):
        return {
            'die_type': 'D6',
            'result': self.value
        }


class D8(Die):

    FixedRolls = []

    @staticmethod
    def fix_result(value):
        if 1 <= value <= 8:
            D8.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of D8 must be between 1 and 8")

    def __init__(self, rnd):
        if len(D8.FixedRolls) > 0:
            self.value = D8.FixedRolls.pop(0)
        else:
            self.value = rnd.randint(1, 9)

    def get_value(self):
        return self.value

    def to_json(self):
        return {
            'die_type': 'D8',
            'result': self.value
        }


class BBDie(Die):

    FixedRolls = []

    @staticmethod
    def fix_result(value):
        if type(value) == BBDieResult:
            BBDie.FixedRolls.append(value)
        else:
            raise ValueError("Fixed result of BBDie must be a BBDieResult")

    def __init__(self, rnd):
        if len(BBDie.FixedRolls) > 0:
            self.value = BBDie.FixedRolls.pop(0)
        else:
            r = rnd.randint(1, 7)
            if r == 6:
                r = 3
            self.value = BBDieResult(r)

    def get_value(self):
        return self.value

    def to_json(self):
        return {
            'die_type': 'BB',
            'result': self.value.name
        }


class Dugout:

    def __init__(self, team):
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

    def __init__(self, position=None):
        self.position = position


class Ball(Piece):

    def __init__(self, position, on_ground=True, is_carried=False):
        super().__init__(position)
        self.on_ground = on_ground
        self.is_carried = is_carried

    def move(self, x, y):
        self.position.x += x
        self.position.y += y

    def move_to(self, position):
        self.position = Square(position.x, position.y)

    def to_json(self):
        return {
            'position': self.position.to_json() if self.position is not None else None,
            'on_ground': self.on_ground,
            'is_carried': self.is_carried
        }


class Player(Piece):

    def __init__(self, player_id, role, name, nr, team, extra_skills=None, extra_ma=0, extra_st=0, extra_ag=0, extra_av=0,
                 niggling=0, mng=False, spp=0, position=None):
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
        self.niggling = niggling
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
            'skills': [skill.name for skill in self.get_skills()],
            'ma': self.get_ma(),
            'st': self.get_st(),
            'ag': self.get_ag(),
            'av': self.get_av(),
            'niggling': self.niggling,
            'mng': self.mng,
            'spp': self.spp,
            'state': self.state.to_json(),
            'position': self.position.to_json() if self.position is not None else None
        }

    def get_ag(self):
        return self.role.ag + self.extra_ag

    def get_st(self):
        return self.role.st + self.extra_st

    def get_ma(self):
        return self.role.ma + self.extra_ma

    def get_av(self):
        return self.role.av + self.extra_av

    def has_skill(self, skill):
        return skill in self.get_skills()

    def has_used_skill(self, skill):
        return skill in self.state.used_skills

    def get_skills(self):
        return self.role.skills + self.extra_skills

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

    def num_moves_left(self, include_gfi: bool = True):
        if self.state.used or self.state.stunned:
            moves = 0
        else:
            moves = self.get_ma()
            if not self.state.up and not self.has_skill(Skill.JUMP_UP):
                moves = max(0, moves - 3)
            moves = moves - self.state.moves
            if include_gfi:
                if self.has_skill(Skill.SPRINT):
                    moves = moves + 3
                else:
                    moves = moves + 2
        return moves

    def __eq__(self, other):
        return isinstance(other, Player) and other.player_id == self.player_id

    def __hash__(self):
        return self.player_id.__hash__()


class Square:

    def __init__(self, x, y):
        self.x = x
        self.y = y

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
            return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)
        else:
            return max(abs(other.x - self.x), abs(other.y - self.y))

    def is_adjacent(self, other, manhattan=False):
        return self.distance(other, manhattan) == 1


class Race:

    def __init__(self, name, roles, reroll_cost, apothecary, stakes):
        self.name = name
        self.roles = roles
        self.reroll_cost = reroll_cost
        self.apothecary = apothecary
        self.stakes = stakes


class Team:

    def __init__(self, team_id, name, race, players=None, treasury=0, apothecary=False, rerolls=0, ass_coaches=0,
                 cheerleaders=0, fan_factor=0):
        self.team_id = team_id
        self.name = name
        self.race = race
        self.players = players if players is not None else []
        self.treasury = treasury
        self.apothecary = apothecary
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
            'apothecary': self.apothecary,
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


class Outcome:

    def __init__(self, outcome_type, position=None, player=None, opp_player=None, rolls=None, team=None, n=0, skill=None):
        self.outcome_type = outcome_type
        self.position = position
        self.player = player
        self.opp_player = opp_player
        self.rolls = rolls if rolls is not None else []
        self.team = team
        self.n = n
        self.skill = skill

    def to_json(self):
        rolls = []
        for roll in self.rolls:
            rolls.append(roll.to_json())
        return {
            'outcome_type': self.outcome_type.name,
            'pos': self.position.to_json() if self.position is not None else None,
            'player_id': self.player.player_id if self.player is not None else None,
            'opp_player': self.opp_player.player_id if self.opp_player is not None else None,
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

    def __init__(self, name, races=[], star_players=[], inducements=[], spp_actions={}, spp_levels={}, improvements={}, se_start=0, se_interval=0, se_pace=0):
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


class Formation:

    def __init__(self, name, formation):
        self.name = name
        self.formation = formation

    def _get_player(self, players, t):
        if t == 'S':
            idx = np.argmax([player.get_st() + (0.5 if player.has_skill(Skill.BLOCK) else 0) for player in players])
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
            idx = np.argmax([1 if player.has_skill(Skill.BLOCK) else (0 if player.has_skill(Skill.PASS) or player.has_skill(Skill.CATCH) else 0.5) for player in players])
            return players[idx]
        return players[0]

    def actions(self, game, team):
        home = team == game.state.home_team
        actions = []
        # Move all player on the pitch back to the reserves
        player_on_pitch = []
        for player in team.players:
            if player.position is not None:
                actions.append(Action(ActionType.PLACE_PLAYER, position=None, player=player))
                player_on_pitch.append(player)

        # Go through formation from scrimmage to touchdown zone
        players = [player for player in game.get_reserves(team) + player_on_pitch]

        positions_used = []

        # setup on scrimmage
        for t in ['S', 's', 'p', 'b', 'c', 'm', 'a', 'v', 'd', '0', 'x']:
            for y in range(len(self.formation)):
                if len(players) == 0:
                    return actions
                x = len(self.formation[0])-1
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
                positions_used.append(position)

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
                    positions_used.append(position)
        return actions
