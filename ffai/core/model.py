from copy import copy, deepcopy
import numpy as np
import uuid
from math import sqrt
from bb.core.util import *
from bb.core.table import *


class Configuration:

    def __init__(self):
        self.name = "Default"
        self.arena = None
        self.ruleset = None
        self.roster_size = 16
        self.pitch_max = 11
        self.pitch_min = 3
        self.scrimmage_max = 3
        self.wing_max = 2
        self.rounds = 8
        self.kick_off_table = True
        self.fast_mode = False
        self.debug_mode = False
        self.kick_scatter_distance = "d6"
        self.offensive_formations = []
        self.defensive_formations = []


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

    def to_simple(self):
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
            'casualty_effect': self.casualty_effect.name if self.casualty_effect is not None else None
        }

    def reset(self):
        self.up = True
        self.used = False
        self.stunned = False
        self.bone_headed = False
        self.hypnotized = False
        self.really_stupid = False
        self.heated = False


class Agent:

    def __init__(self, name, human=False):
        self.agent_id = str(uuid.uuid1())
        self.name = name
        self.human = human

    def to_simple(self):
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'human': self.human
        }

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

    def to_simple(self):
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
            'reroll_used': self.reroll_used
        }

    def reset_turn(self):
        self.reroll_used = False

    def use_reroll(self):
        self.rerolls -= 1
        self.reroll_used = True


class GameState:

    def __init__(self, game, home_team, away_team):
        self.stack = Stack()
        self.reports = []
        self.half = 1
        self.round = 0
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

    def clone(self):
        state = GameState(deepcopy(self.home_team), deepcopy(self.away_team))
        state.stack = deepcopy(self.stack)
        state.reports = deepcopy(self.reports)
        state.round = self.round
        state.kicking_first_half = self.kicking_first_half
        state.receiving_first_half = self.receiving_first_half
        state.kicking_this_drive = self.kicking_this_drive
        state.receiving_this_drive = self.receiving_this_drive
        state.current_team = None if state.current_team is None else state.home_team if state.current_team.team_id == state.home_team.team_id else state.away_team
        state.pitch = Pitch(self.pitch.width, self.pitch.height)
        for player in state.home_team.players:
            if player.position is not None:
                state.pitch.board[player.position.y][player.position.x] = player
        for player in state.away_team.players:
            if player.position is not None:
                state.pitch.board[player.position.y][player.position.x] = player
        state.weather = self.weather
        state.gentle_gust = self.gentle_gust
        state.turn_order = [self.team_by_id[team.team_id] for team in self.turn_order]
        state.spectators = self.spectators
        state.active_player = self.player_by_id[self.active_player.player_id] if self.active_player is not None else None
        state.game_over = self.game_over
        state.available_actions = copy(self.available_actions)

    def get_dugout(self, team):
        return self.dugouts[team.team_id]

    def to_simple(self):
        return {
            'half': self.half,
            'kicking_first_half': self.kicking_first_half.team_id if self.kicking_first_half is not None else None,
            'receiving_first_half': self.receiving_first_half.team_id if self.receiving_first_half is not None else None,
            'kicking_this_drive': self.kicking_this_drive.team_id if self.kicking_this_drive is not None else None,
            'receiving_this_drive': self.receiving_this_drive.team_id if self.receiving_this_drive is not None else None,
            'pitch': self.pitch.to_simple(),
            'home_dugout': self.dugouts[self.home_team.team_id].to_simple(),
            'away_dugout': self.dugouts[self.away_team.team_id].to_simple(),
            'home_team': self.home_team.to_simple(),
            'away_team': self.away_team.to_simple(),
            'game_over': self.game_over,
            'weather': self.weather.name,
            'gentle_gust': self.gentle_gust,
            'available_actions': [action.to_simple() for action in self.available_actions],
            'reports': [report.to_simple() for report in self.reports],
            'current_team_id': self.current_team.team_id if self.current_team is not None else None,
            'round': self.round,
            'spectators': self.spectators,
            'active_player_id': self.active_player.player_id if self.active_player is not None else None
        }


class Ball:

    def __init__(self, position, on_ground=True, is_carried=False):
        self.position = position
        self.on_ground = on_ground
        self.is_carried = is_carried

    def move(self, x, y):
        self.position = Square(self.position.x + x, self.position.y + y)

    def move_to(self, pos):
        self.position = Square(pos.x, pos.y)

    def to_simple(self):
        return {
            'position': self.position.to_simple() if self.position is not None else None,
            'on_ground': self.on_ground,
            'is_carried': self.is_carried
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

    def to_simple(self):
        board = []
        for y in range(len(self.board)):
            row = []
            for x in range(len(self.board[0])):
                row.append(self.board[y][x].player_id if self.board[y][x] is not None else None)
            board.append(row)
        return {
            'board': board,
            'balls': [ball.to_simple() for ball in self.balls]
        }

    def put(self, piece, pos):
        piece.position = Square(pos.x, pos.y)
        self.board[pos.y][pos.x] = piece

    def remove(self, piece):
        assert piece.position is not None
        self.board[piece.position.y][piece.position.x] = None
        piece.position = None

    def move(self, piece, pos_to):
        assert piece.position is not None
        assert self.board[pos_to.y][pos_to.x] is None
        for ball in self.balls:
            if ball.position == piece.position and ball.is_carried:
                ball.move_to(pos_to)
        self.remove(piece)
        self.put(piece, pos_to)

    def swap(self, piece_a, piece_b):
        assert piece_a.position is not None
        assert piece_b.position is not None
        pos_a = Square(piece_a.position.x, piece_a.position.y)
        pos_b = Square(piece_b.position.x, piece_b.position.y)
        piece_a.position = pos_b
        piece_b.position = pos_a
        self.board[pos_a.y][pos_a.x] = piece_b
        self.board[pos_b.y][pos_b.x] = piece_a

    def get_balls_at(self, pos, in_air=False):
        balls = []
        for ball in self.balls:
            if ball.position == pos and (ball.on_ground or in_air):
                balls.append(ball)
        return balls

    def get_ball_at(self, pos, in_air=False):
        """
        Assumes there is only one ball on the square
        :param pos:
        :param in_air:
        :return: Ball or None
        """
        for ball in self.balls:
            if ball.position == pos and (ball.on_ground or in_air):
                return ball
        return None

    def get_ball_positions(self):
        return [ball.position for ball in self.balls]

    def get_ball_position(self):
        """
        Assumes there is only one ball on the square
        :return: Ball or None
        """
        for ball in self.balls:
            return ball.position
        return None

    def is_out_of_bounds(self, pos):
        return pos.x < 1 or pos.x >= self.width-1 or pos.y < 1 or pos.y >= self.height-1

    def get_player_at(self, pos):
        return self.board[pos.y][pos.x]

    def get_push_squares(self, pos_from, pos_to):
        squares_to = self.get_adjacent_squares(pos_to, include_out=True)
        squares_empty = []
        squares_out = []
        squares = []
        #print("From:", pos_from)
        for square in squares_to:
            #print("Checking: ", square)
            #print("Distance: ", pos_from.distance(square, manhattan=False))
            include = False
            if pos_from.x == pos_to.x or pos_from.y == pos_to.y:
                if pos_from.distance(square, manhattan=False) >= 2:
                    include = True
            else:
                if pos_from.distance(square, manhattan=True) >= 3:
                    include = True
            #print("Include: ", include)
            if include:
                if self.get_player_at(square) is None:
                    squares_empty.append(square)
                if self.is_out_of_bounds(square):
                    squares_out.append(square)
                squares.append(square)
        if len(squares_empty) > 0:
            return squares_empty
        if len(squares_out) > 0:
            return squares_out
        return squares

    def get_adjacent_squares(self, pos, manhattan=False, include_out=False, exclude_occupied=False):
        squares = []
        for yy in Pitch.range:
            for xx in Pitch.range:
                if yy == 0 and xx == 0:
                    continue
                sq = self.squares[pos.y+yy][pos.x+xx]
                #assert sq.y == pos.y+yy and sq.x == pos.x+xx
                #sq = Square(pos.x+xx, pos.y+yy)
                if not include_out and self.is_out_of_bounds(sq):
                    continue
                if exclude_occupied and self.get_player_at(sq) is not None:
                    continue
                if not manhattan:
                    squares.append(sq)
                elif xx == 0 or yy == 0:
                    squares.append(sq)
        return squares

    def adjacent_player_squares_at(self, player, position, include_own=True, include_opp=True, manhattan=False, only_blockable=False, only_foulable=False):
        squares = []
        for square in self.get_adjacent_squares(position, manhattan=manhattan):
            player_at = self.get_player_at(square)
            if player_at is None:
                continue
            if include_own and player_at.team == player.team or include_opp and not player_at.team == player.team:
                if not only_blockable or player_at.state.up:
                    if not only_foulable or not player_at.state.up:
                        squares.append(square)
        return squares

    def adjacent_player_squares(self, player, include_own=True, include_opp=True, manhattan=False, only_blockable=False, only_foulable=False):
        return self.adjacent_player_squares_at(player, player.position, include_own, include_opp, manhattan, only_blockable, only_foulable)

    def num_tackle_zones_at(self, player, position):
        tackle_zones = 0
        for square in self.adjacent_player_squares_at(player, position, include_own=False, include_opp=True):
            player = self.get_player_at(square)
            if player is not None and player.has_tackle_zone():
                tackle_zones += 1
        return tackle_zones

    def num_tackle_zones_in(self, player):
        tackle_zones = 0
        for square in self.adjacent_player_squares(player, include_own=False, include_opp=True):
            player = self.get_player_at(square)
            if player is not None and player.has_tackle_zone():
                tackle_zones += 1
        return tackle_zones

    def tackle_zones_detailed(self, player):
        tackle_zones = 0
        tacklers = []
        prehensile_tailers = []
        diving_tacklers = []
        shadowers = []
        tentaclers = []
        for square in self.adjacent_player_squares(player.position, include_own=False, include_opp=True):
            player_at = self.get_player_at(square)
            if player_at is not None and player_at.has_tackle_zone():
                tackle_zones += 1
            if player_at is None and player_at.has_skill(Skill.TACKLE):
                tacklers.append(player_at)
            if player_at is None and player_at.has_skill(Skill.PREHENSILE_TAIL):
                prehensile_tailers.append(player_at)
            if player_at is None and player_at.has_skill(Skill.DIVING_TACKLE):
                diving_tacklers.append(player_at)
            if player_at is None and player_at.has_skill(Skill.SHADOWING):
                shadowers.append(player_at)
            if player_at is None and player_at.has_skill(Skill.TENTACLES):
                tentaclers.append(player_at)

        return tackle_zones, tacklers, prehensile_tailers, diving_tacklers, shadowers, tentaclers

    def assists(self, player, opp_player, ignore_guard=False):
        assists = []
        for yy in range(-1, 2, 1):
            for xx in range(-1, 2, 1):
                if yy == 0 and xx == 0:
                    continue
                p = Square(opp_player.position.x+xx, opp_player.position.y+yy)
                if not self.is_out_of_bounds(p) and player.position != p:
                    player_at = self.get_player_at(p)
                    if player_at is not None:
                        if player_at.team == player.team:
                            if not player_at.can_assist():
                                continue
                            if (not ignore_guard and player_at.has_skill(Skill.GUARD)) or \
                                            self.num_tackle_zones_in(player_at) <= 1:
                                # TODO: Check if attacker has a tackle zone
                                assists.append(player_at)
        return assists

    def passes(self, passer, weather):
        squares = []
        distances = []
        distances_allowed = [PassDistance.QUICK_PASS,
                             PassDistance.SHORT_PASS,
                             PassDistance.LONG_PASS,
                             PassDistance.LONG_BOMB,
                             PassDistance.HAIL_MARY] if Skill.HAIL_MARY_PASS in passer.get_skills() \
            else [PassDistance.QUICK_PASS, PassDistance.SHORT_PASS, PassDistance.LONG_PASS, PassDistance.LONG_BOMB]
        if weather == WeatherType.BLIZZARD:
            distances_allowed = [PassDistance.QUICK_PASS, PassDistance.SHORT_PASS]
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                pos = Square(x, y)
                if self.is_out_of_bounds(pos) or passer.position == pos:
                    continue
                distance = self.pass_distance(passer, pos)
                if distance in distances_allowed:
                    squares.append(pos)
                    distances.append(distance)
        return squares, distances

    def pass_distance(self, passer, pos):
        distance_x = abs(passer.position.x - pos.x)
        distance_y = abs(passer.position.y - pos.y)
        if distance_y >= len(Rules.pass_matrix) or distance_x >= len(Rules.pass_matrix[0]):
            return PassDistance.HAIL_MARY
        distance = Rules.pass_matrix[distance_y][distance_x]
        return PassDistance(distance)

    def interceptors(self, passer, pos):
        """
        1) Find line x from a to b
        2) Find squares s where x intersects
        3) Find manhattan neighboring n squares of s
        4) Remove squares where distance to a is larger than dist(a,b)
        5) Remove squares without standing opponents with hands
        6) Determine players on squares
        """

        # 1) Find line x from a to b
        x = get_line((passer.position.x, passer.position.y), (pos.x, pos.y))

        # 2) Find squares s where x intersects
        s = []
        for i in x:
            s.append(Square(i[0], i[1]))

        # 3) Include manhattan neighbors s into n
        # 4) Remove squares where distance to a is larger than dist(a,b)
        max_distance = passer.position.distance(pos)
        n = set()
        for square in s:
            for neighbor in self.get_adjacent_squares(square) + [square]:

                if neighbor in n:
                    continue

                # 4) Remove squares where distance to a is larger than dist(a,b)
                if neighbor.distance(passer.position) > max_distance:
                    continue
                if neighbor.distance(pos) > max_distance:
                    continue
                if neighbor.x > max(passer.position.x, pos.x) or neighbor.x < min(passer.position.x, pos.x):
                    continue
                if neighbor.y > max(passer.position.y, pos.y) or neighbor.y < min(passer.position.y, pos.y):
                    continue

                # 5) Remove squares without standing opponents with hands
                player_at = self.get_player_at(neighbor)
                if player_at is None:
                    continue
                if player_at.team == passer.team:
                    continue
                if player_at.can_catch():
                    continue
                if player_at.has_skill(Skill.NO_HANDS):
                    continue

                n.add(neighbor)

        if passer.position in n:
            n.remove(pos)
        if pos in n:
            n.remove(pos)

        players = []
        for square in n:
            players.append(self.get_player_at(square))

        return players


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

    def to_simple(self):
        return {
            'action_type': self.action_type.name,
            'positions': [position.to_simple() if position is not None else None for position in self.positions],
            'team_id': self.team.team_id if self.team is not None else None,
            "rolls": self.rolls,
            "block_rolls": self.block_rolls,
            "agi_rolls": self.agi_rolls,
            'player_ids': [player.player_id for player in self.players],
            "disabled": self.disabled
        }


class Action:

    def __init__(self, action_type, pos=None, player=None, idx=0, dice_result=None):
        self.action_type = action_type
        self.pos = pos
        self.player = player
        self.idx = idx
        self.dice_result = dice_result

    def to_simple(self):
        return {
            'action_type': self.action_type.name,
            'position': self.pos.to_simple() if self.pos is not None else None,
            'player_id': self.player.player_id if self.player is not None else None,
            'dice_result': self.dice_result
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

    def in_opp_endzone(self, pos, home):
        if home:
            return self.board[pos.y][pos.x] == Tile.AWAY_TOUCHDOWN
        else:
            return self.board[pos.y][pos.x] == Tile.HOME_TOUCHDOWN

    def to_simple(self):
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
        pass


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

    def to_simple(self):
        dice = []
        for die in self.dice:
            dice.append(die.to_simple())
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

    def __init__(self, rnd):
        self.value = rnd.randint(1, 3)

    def get_value(self):
        return self.value

    def to_simple(self):
        return {
            'die_type': 'D3',
            'result': self.value
        }


class D6(Die):

    def __init__(self, rnd):
        self.value = rnd.randint(1, 6)

    def get_value(self):
        return self.value

    def to_simple(self):
        return {
            'die_type': 'D6',
            'result': self.value
        }


class D8(Die):

    def __init__(self, rnd):
        self.value = rnd.randint(1, 8)

    def get_value(self):
        return self.value

    def to_simple(self):
        return {
            'die_type': 'D8',
            'result': self.value
        }


class BBDie(Die):

    def __init__(self, rnd):
        r = rnd.randint(1, 6)
        if r == 6:
            r = 3
        self.value = BBDieResult(r)

    def get_value(self):
        return self.value

    def to_simple(self):
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

    def to_simple(self):
        return {
            'team_id': self.team.team_id,
            'reserves': [player.player_id for player in self.reserves],
            'kod': [player.player_id for player in self.kod],
            'casualties': [player.player_id for player in self.casualties],
            'dungeon': [player.player_id for player in self.dungeon]
        }


class Role:

    def __init__(self, name, races, ma, st, ag, av, skills, cost, feeder, n_skill_sets=[], d_skill_sets=[],
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
        self.n_skill_sets = n_skill_sets
        self.d_skill_sets = d_skill_sets
        self.star_player = star_player


class Piece:

    def __init__(self, position=None):
        self.position = position


class Player(Piece):

    def __init__(self, player_id, role, name, nr, team, extra_skills=[], extra_ma=0, extra_st=0, extra_ag=0, extra_av=0,
                 niggling=0, mng=False, spp=0, position=None):
        super().__init__(position)
        self.player_id = player_id
        self.role = role
        self.name = name
        self.nr = nr
        self.team = team
        self.extra_skills = extra_skills
        self.skills = self.extra_skills + self.role.skills
        self.extra_ma = extra_ma
        self.extra_st = extra_st
        self.extra_ag = extra_ag
        self.extra_av = extra_av
        self.niggling = niggling
        self.mng = mng
        self.spp = spp
        self.state = PlayerState()

    def get_ag(self):
        return self.role.ag + self.extra_ag

    def get_st(self):
        return self.role.st + self.extra_st

    def get_ma(self):
        return self.role.ma + self.extra_ma

    def get_av(self):
        return self.role.av + self.extra_av

    def has_skill(self, skill):
        return skill in self.skills

    def get_skills(self):
        return self.skills

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

    def to_simple(self):
        skills = []
        for skill in self.get_skills():
            skills.append(skill.name)
        return {
            'player_id': self.player_id,
            'name': self.name,
            'role': self.role.name,
            'team_id': self.team.team_id,
            'nr': self.nr,
            'skills': [skill.name for skill in self.skills],
            'ma': self.get_ma(),
            'st': self.get_st(),
            'ag': self.get_ag(),
            'av': self.get_av(),
            'niggling': self.niggling,
            'mng': self.mng,
            'spp': self.spp,
            'state': self.state.to_simple(),
            'position': self.position.to_simple() if self.position is not None else None
        }

    def __eq__(self, other):
        return isinstance(other, Player) and other.player_id == self.player_id


class Square:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_simple(self):
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


class Coach:

    def __init__(self, coach_id, name):
        self.coach_id = coach_id
        self.name = name

    def to_simple(self):
        return {
            'coach_id': self.coach_id,
            'name': self.name
        }


class Race:

    def __init__(self, name, roles, reroll_cost, apothecary, stakes):
        self.name = name
        self.roles = roles
        self.reroll_cost = reroll_cost
        self.apothecary = apothecary
        self.stakes = stakes


class Team:

    def __init__(self, team_id, name, race, coach, players=None, treasury=0, apothecary=False, rerolls=0, ass_coaches=0,
                 cheerleaders=0, fan_factor=0):
        self.team_id = team_id
        self.name = name
        self.coach = coach
        self.race = race
        self.players = players if players is not None else []
        self.treasury = treasury
        self.apothecary = apothecary
        self.rerolls = rerolls
        self.fan_factor = fan_factor
        self.ass_coaches = ass_coaches
        self.cheerleaders = cheerleaders
        self.state = TeamState(self)

    def to_simple(self):
        players = []
        players_by_id = {}
        for player in self.players:
            players.append(player.to_simple())
            players_by_id[player.player_id] = player.to_simple()
        return {
            'team_id': self.team_id,
            'name': self.name,
            'coach': self.coach.to_simple(),
            'race': self.race,
            'treasury': self.treasury,
            'apothecary': self.apothecary,
            'rerolls': self.rerolls,
            'ass_coaches': self.ass_coaches,
            'cheerleaders': self.cheerleaders,
            'fan_factor': self.fan_factor,
            'players_by_id': players_by_id,
            'state': self.state.to_simple()
        }

    def __eq__(self, other):
        return other is not None and other.team_id == self.team_id


class Outcome:

    def __init__(self, outcome_type, pos=None, player=None, opp_player=None, rolls=[], team=None, n=0, skill=None):
        self.outcome_type = outcome_type
        self.pos = pos
        self.player = player
        self.opp_player = opp_player
        self.rolls = rolls
        self.team = team
        self.n = n
        self.skill = skill

    def to_simple(self):
        rolls = []
        for roll in self.rolls:
            rolls.append(roll.to_simple())
        return {
            'outcome_type': self.outcome_type.name,
            'pos': self.pos.to_simple() if self.pos is not None else None,
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

    def _get_player(self, players, type):
        if type == 's':
            idx = np.argmax([player.get_st() + (0.5 if player.has_skill(Skill.BLOCK) else 0) for player in players])
            return players[idx]
        if type == 'm':
            idx = np.argmax([player.get_ma() for player in players])
            return players[idx]
        if type == 'a':
            idx = np.argmax([player.get_ag() for player in players])
            return players[idx]
        if type == 'v':
            idx = np.argmax([player.get_av() for player in players])
            return players[idx]
        if type == 'p':
            idx = np.argmax([1 if player.has_skill(Skill.PASS) else 0 for player in players])
            return players[idx]
        if type == 'c':
            idx = np.argmax([1 if player.has_skill(Skill.CATCH) else 0 for player in players])
            return players[idx]
        if type == 'b':
            idx = np.argmax([1 if player.has_skill(Skill.BLOCK) else 0 for player in players])
            return players[idx]
        if type == 'd':
            idx = np.argmax([1 if player.has_skill(Skill.DODGE) else 0 for player in players])
            return players[idx]
        if type == '0':
            idx = np.argmin([len(player.get_skills()) for player in players])
            return players[idx]
        if type == 'x':
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
                actions.append(Action(ActionType.PLACE_PLAYER, pos=None, player=player))
                player_on_pitch.append(player)
        # Go through formation from scrimmage to touchdown zone
        players = [player for player in game.get_reserves(team) + player_on_pitch]
        for y in range(len(self.formation)):
            for x in reversed(range(len(self.formation[0]))):
                if len(players) == 0:
                    return actions
                type = self.formation[y][x]
                if type == '-':
                    continue
                yy = y + 1
                xx = x + 1 if not home else game.arena.width - x - 2
                player = self._get_player(players, type)
                players.remove(player)
                actions.append(Action(ActionType.PLACE_PLAYER, pos=Square(xx, yy), player=player))
        return actions
