"""
A number of static methods for interpretting the state of the fantasy football pitch that aren't required directly by
the client
"""
import ffai.core.game as g
import ffai.core.table as t
import ffai.core.model as m
import ffai.core.procedure as p
import ffai.util.pathfinding as pf
from typing import Optional, List, Dict
import math


class ActionSequence:

    def __init__(self, action_steps: List[m.Action], score: float = 0, description: str = ''):
        """ Creates a new ActionSequence - an ordered list of sequential m.Actions to attempt to undertake.
        :param action_steps: Sequence of action steps that form this action.
        :param score: A score representing the attractiveness of the move (default: 0)
        :param description: A debug string (default: '')
        """

        # Note the intention of this object is that when the object is acting, as steps are completed,
        # they are removed from the move_sequence so the next move is always the top of the move_sequence
        # list.

        self.action_steps = action_steps
        self.score = score
        self.description = description

    def is_valid(self, game: g.Game) -> bool:
        pass

    def popleft(self):
        return self.action_steps.pop(0)
        #val = self.action_steps[0]
        #del self.action_steps[0]
        #return val

    def is_empty(self):
        return not self.action_steps


class FfHeatMap:
    """ A heat map of a Blood Bowl field.

    A class for analysing zones of control for both teams
    """

    def __init__(self, game: g.Game, team: g.Team):
        self.game=game
        self.team = team
        # Note that the edges are not on the field, but represent crowd squares
        self.units_friendly: List[List[float]] = [[0.0 for y in range(game.state.pitch.height)] for x in range(game.state.pitch.width)]
        self.units_opponent: List[List[float]] = [[0.0 for y in range(game.state.pitch.height)] for x in range(game.state.pitch.width)]


    def add_unit_paths(self, player:m.Player, paths: List[pf.Path]):
        is_friendly: bool = player.team == self.team

        for path in paths:
            if is_friendly:
                self.units_friendly[path.steps[-1].x][path.steps[-1].y] += (1.0 - path.cost)*(1.0 - path.cost)
            else:
                self.units_opponent[path.steps[-1].x][path.steps[-1].y] += (1.0 - path.cost)*(1.0 - path.cost)

    def add_unit_by_paths(self, game: g.Game, paths: Dict[m.Player, List[pf.Path]]):
        for player in paths.keys():
            self.add_unit_paths(player, paths[player])

    def add_players_moved(self, game: g.Game, players: List[m.Player]):
        for player in players:
            adjacents: List[m.Square] = game.get_adjacent_squares(player.position, include_occupied=True)
            self.units_friendly[player.position.x][player.position.y] += 1.0
            for adjacent in adjacents:
                self.units_friendly[player.position.x][player.position.y] += 0.5

    def get_ball_move_square_safety_score(self, square: m.Square) -> float:

        # Basic idea - identify safe regions to move the ball towards
        friendly_heat: float = self.units_friendly[square.x][square.y]
        opponent_heat: float = self.units_opponent[square.x][square.y]

        score: float = 30.0 * max(0.0, (1.0 - opponent_heat/2))
        return score

        #score: float=0.0
        #if opponent_heat < 0.25: score += 15.0
        #if opponent_heat < 0.05: score += 15.0
        #if opponent_heat < 1.5: score += 5
        #if friendly_heat > 3.5: score += 10.0
        #score += max(30.0, 5.0*(friendly_heat-opponent_heat))

        return score

    def get_cage_necessity_score(self, square: m.Square) -> float:
        opponent_friendly: float = self.units_friendly[square.x][square.y]
        opponent_heat: float = self.units_opponent[square.x][square.y]
        score: float = 0.0

        if opponent_heat < 0.4: score -= 80.0
        # if opponent_friendly > opponent_heat: score -= max(30.0, 10.0*(opponent_friendly-opponent_heat))
        # if opponent_heat <1.5: score -=5
        # if opponent_heat > opponent_friendly: score += 10.0*(opponent_friendly-opponent_heat)

        return score


def blitz_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_BLITZ:
            return False
    return True


def handoff_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_HANDOFF:
            return False
    return True


def foul_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_FOUL:
            return False
    return True


def pass_used(game: g.Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == t.ActionType.START_PASS:
            return False
    return True


def get_players(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = True, include_used: bool = True, include_off_pitch: bool = False, only_blockable: bool = False, only_used: bool = False) -> List[m.Player]:
    players: List[m.Player] = []
    selected_players: List[m.Player] = []
    for iteam in game.state.teams:
        if iteam == team and include_own:
            players.extend(iteam.players)
        if iteam != team and include_opp:
            players.extend(iteam.players)
    for player in players:
        if only_blockable and not player.state.up:
            continue
        if only_used and not player.state.used:
            continue

        if include_stunned or not player.state.stunned:
            if include_used or not player.state.used:
                if include_off_pitch or (player.position is not None and not game.state.pitch.is_out_of_bounds(player.position)):
                    selected_players.append(player)

    return selected_players


def caging_squares_north_east(game: g.Game, protect_square: m.Square) -> List[m.Square]:

    # * At it's simplest, a cage requires 4 platers in the North-East, South-East, South-West and North-West
    # * positions, relative to the ball carrier, such that there is no more than 3 squares between the players in
    # * each of those adjacent compass directions.
    # *
    # *   1     3
    # *    xx-xx
    # *    xx-xx
    # *    --o--
    # *    xx-xx
    # *    xx-xx
    # *   3     4
    # *
    # * pitch is 26 long
    # *
    # *
    # * Basically we need one player in each of the corners: 1-4, but spaced such that there is no gap of 3 squares.
    # * If the caging player is in 1-4, but next to ball carrier, he ensures this will automatically be met.
    # *
    # * The only exception to this is when the ball carrier is on, or near, the sideline.  Then return the squares
    # * that can otherwise form the cage.
    # *

    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x <= game.state.pitch.width - 3:
        if y == game.state.pitch.height-2:
            caging_squares.append(game.state.pitch.get_square(x + 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x + 2, y + 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        elif y == game.state.pitch.height-1:
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x + 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y + 2))
            caging_squares.append(game.state.pitch.get_square(x + 2, y + 1))
            # caging_squares.append(game.state.pitch.get_square(x + 3, y + 3))

    return caging_squares


def caging_squares_north_west(game: g.Game, protect_square: m.Square) -> List[m.Square]:

    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x >= 3:
        if y == game.state.pitch.height-2:
            caging_squares.append(game.state.pitch.get_square(x - 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x - 2, y + 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        elif y == game.state.pitch.height-1:
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x - 1, y + 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y + 2))
            caging_squares.append(game.state.pitch.get_square(x - 2, y + 1))
            # caging_squares.append(game.state.pitch.get_square(x - 3, y + 3))

    return caging_squares


def caging_squares_south_west(game: g.Game, protect_square: m.Square) -> List[m.Square]:

    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x >= 3:
        if y == 2:
            caging_squares.append(game.state.pitch.get_square(x - 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x - 2, y - 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        elif y == 1:
            caging_squares.append(game.state.pitch.get_square(x - 1, y))
            caging_squares.append(game.state.pitch.get_square(x - 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x - 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x - 1, y - 2))
            caging_squares.append(game.state.pitch.get_square(x - 2, y - 1))
            # caging_squares.append(game.state.pitch.get_square(x - 3, y - 3))

    return caging_squares


def caging_squares_south_east(game: g.Game, protect_square: m.Square) -> List[m.Square]:

    caging_squares: List[m.Square] = []
    x = protect_square.x
    y = protect_square.y

    if x <= game.state.pitch.width-3:
        if y == 2:
            caging_squares.append(game.state.pitch.get_square(x + 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x + 2, y - 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        elif y == 1:
            caging_squares.append(game.state.pitch.get_square(x + 1, y))
            caging_squares.append(game.state.pitch.get_square(x + 2, y))
        else:
            caging_squares.append(game.state.pitch.get_square(x + 1, y - 1))
            caging_squares.append(game.state.pitch.get_square(x + 1, y - 2))
            caging_squares.append(game.state.pitch.get_square(x + 2, y - 1))
            # caging_squares.append(game.state.pitch.get_square(x + 3, y - 3))

    return caging_squares


def is_caging_position(game: g.Game, player: m.Player, protect_player: m.Player) -> bool:
    return player.position.distance(protect_player.position) <= 2 and not is_castle_position_of(game, player, protect_player)


def has_player_within_n_squares(game: g.Game, units: List[m.Player], square: m.Square, num_squares: int) -> bool:
    for cur in units:
        if cur.position.distance(square) <= num_squares:
            return True
    return False


def has_adjacent_player(game: g.Game, square: m.Square) -> bool:
    return not game.state.pitch.get_adjacent_players(square)


def is_castle_position_of(game: g.Game, player1: m.Player, player2: m.Player) -> bool:
    return player1.position.x == player2.position.x or player1.position.y == player2.position.y


def is_bishop_position_of(game: g.Game, player1: m.Player, player2: m.Player) -> bool:
    return abs(player1.position.x - player2.position.x) == abs(player1.position.y - player2.position.y)


def attacker_would_surf(game: g.Game, attacker: m.Player, defender: m.Player) -> bool:
    if (defender.has_skill(t.Skill.SIDE_STEP) and not attacker.has_skill(t.Skill.GRAB)) or defender.has_skill(t.Skill.STAND_FIRM):
        return False

    if not attacker.position.is_adjacent(defender.position):
        return False

    return direct_surf_squares(game, attacker.position, defender.position)


def direct_surf_squares(game: g.Game, attack_square: m.Square, defend_square: m.Square) -> bool:
    defender_on_sideline: bool = on_sideline(game, defend_square)
    defender_in_endzone: bool = on_endzone(game, defend_square)

    if defender_on_sideline and defend_square.x == attack_square.x:
        return True

    if defender_in_endzone and defend_square.y == attack_square.y:
        return True

    if defender_in_endzone and defender_on_sideline:
        return True

    return False


def reverse_x_for_right(game: g.Game, team: m.Team, x: int) -> int:
    if not game.is_team_side(m.Square(13, 3), team):
        res = game.state.pitch.width - 1 - x
    else:
        res = x
    return res

def reverse_x_for_left(game: g.Game, team: m.Team, x: int) -> int:
    if game.is_team_side(m.Square(13, 3), team):
        res = game.state.pitch.width - 1 - x
    else:
        res = x
    return res

def on_sideline(game: g.Game, square: m.Square) -> bool:
    return square.y == 1 or square.y == game.state.pitch.height - 1


def on_endzone(game: g.Game, square: m.Square) -> bool:
    return square.x == 1 or square.x == game.state.pitch.width - 1


def on_los(game: g.Game, team: g.Team, square: m.Square) -> bool:
    return (reverse_x_for_right(game, team, square.x) == 13) and 4 < square.y < 21


def los_squares(game: g.Game, team: g.Team) -> List[m.Square]:

    squares: List[m.Square] = [
        game.state.pitch.get_square(reverse_x_for_right(game, team, 13), 5),
        game.state.pitch.get_square(reverse_x_for_right(game, team, 13), 6),
        game.state.pitch.get_square(reverse_x_for_right(game, team, 13), 7),
        game.state.pitch.get_square(reverse_x_for_right(game, team, 13), 8),
        game.state.pitch.get_square(reverse_x_for_right(game, team, 13), 9),
        game.state.pitch.get_square(reverse_x_for_right(game, team, 13), 10),
        game.state.pitch.get_square(reverse_x_for_right(game, team, 13), 11)
    ]
    return squares


def distance_to_sideline(game: g.Game, square: m.Square) -> int:
    return min(square.y - 1, game.state.pitch.height - square.y - 2)


def is_endzone(game, square: m.Square) -> bool:
    return square.x == 1 or square.x == game.state.pitch.width - 1


def last_block_proc(game) -> Optional[p.Block]:
    for i in range(len(game.state.stack.items) - 1, -1, -1):
        if isinstance(game.state.stack.items[i], p.Block):
            block_proc = game.state.stack.items[i]
            return block_proc
    return None


def is_adjacent_ball(game: g.Game, square: m.Square) -> bool:
    ball_square = game.state.pitch.get_ball_position()
    return ball_square is not None and ball_square.is_adjacent(square)


def squares_within(game: g.Game, square: m.Square, distance: int) -> List[m.Square]:
    squares: List[m.Square] = []
    for i in range(-distance, distance+1):
        for j in range(-distance, distance+1):
            cur_square = game.state.pitch.get_square(square.x+i, square.y+j)
            if cur_square != square and not game.state.pitch.is_out_of_bounds(cur_square):
                squares.append(cur_square)
    return squares


def distance_to_defending_endzone(game: g.Game, team: m.Team, position: m.Square) -> int:
    res = reverse_x_for_right(game, team, position.x) - 1
    return res


def distance_to_scoring_endzone(game: g.Game, team: m.Team, position: m.Square) -> int:
    res =  reverse_x_for_left(game, team, position.x) - 1
    return res
    #return game.state.pitch.width - 1 - reverse_x_for_right(game, team, position.x)


def players_in_scoring_endzone(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = False) -> List[m.Player]:
    players: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp)
    selected_players: List[m.Player] = []
    for player in players:
        if in_scoring_endzone(game, team, player.position): selected_players.append(player)
    return selected_players


def in_scoring_endzone(game: g.Game, team: m.Team, square: m.Square) -> bool:
    return reverse_x_for_left(game, team, square.x) == 1


def players_in_scoring_distance(game: g.Game, team: m.Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = False) -> List[m.Player]:
    players: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned)
    selected_players: List[m.Player] = []
    for player in players:
        if distance_to_scoring_endzone(game, team, player.position) <= player.num_moves_left(): selected_players.append(player)
    return selected_players


def distance_to_nearest_player(game: g.Game, team: m.Team, square: m.Square, include_own: bool = True, include_opp: bool = True, only_used: bool = False, include_used: bool = True, include_stunned: bool = True, only_blockable: bool = False) -> int:
    opps: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, only_used=only_used, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)
    cur_max = 100
    for opp in opps:
        dist = opp.position.distance(square)
        cur_max = min(cur_max, dist)
    return cur_max


def screening_distance(game: g.Game, from_square: m.Square, to_square: m.Square) -> float:
    # Return the "screening distance" between 3 squares.  (To complete)
    # float dist =math.sqrt(math.pow(square.x - cur.position.x, 3) + math.pow(square.y - cur.position.y, 3))
    return 0.0


def num_opponents_can_reach(game: g.Game, team: m.Team, square: m.Square) -> int:
    opps: List[m.Player] = get_players(game, team, include_own=False, include_opp=True)
    num_opps_reach: int = 0
    for cur in opps:
        dist = max(square.x - cur.position.x, square.y - cur.position.y)
        if cur.state.stunned: continue
        move_allowed = cur.get_ma() + 2
        if not cur.state.up: move_allowed -= 3
        if dist < move_allowed: num_opps_reach += 1
    return num_opps_reach


def num_opponents_on_field(game: g.Game, team: m.Team) -> int:
    opps: List[m.Player] = get_players(game, team, include_own=False, include_opp=True)
    num_opponents = 0
    for cur in opps:
        if cur.position is not None: num_opponents += 1
    return num_opponents


def number_opponents_closer_than_to_endzone(game: g.Game, team: g.Team, square: m.Square) -> int:
    opponents: List[m.Player] = get_players(game, team, include_own=False, include_opp=True)
    num_opps = 0
    distance_square_endzone = distance_to_defending_endzone(game, team, square)

    for opponent in opponents:
        distance_opponent_endzone = distance_to_defending_endzone(game, team, opponent.position)
        if distance_opponent_endzone < distance_square_endzone: num_opps += 1
    return num_opps


def in_scoring_range(game: g.Game, player: m.Player) -> bool:
    return player.num_moves_left() >= distance_to_scoring_endzone(game, player.team, player.position)


def players_in_scoring_range(game: g.Game, team: m.Team, include_own=True, include_opp=True, include_used=True, include_stunned=True) -> List[m.Player]:
    players: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned, include_used=include_used)
    res: List[m.Player] = []
    for player in players:
        if in_scoring_range(game, player): res.append(player)
    return res


def players_in(game: g.Game, team: m.Team, squares: List[m.Square], include_own=True, include_opp=True, include_used=True, include_stunned=True, only_blockable=False) -> List[m.Player]:

    allowed_players: List[m.Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)
    res: List[m.Player] = []

    for square in squares:
        player: Optional[m.Player] = game.state.pitch.get_player_at(square)
        if player is None:
            continue
        if player in allowed_players:
            res.append(player)
    return res
