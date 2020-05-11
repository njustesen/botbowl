#!/usr/bin/env python3
"""
==========================
Author: Peter Moore / Kevin Glass
Year: 2019
==========================
This example contains Grodbot and a generic A* Path Finder, along with a specific test implementation.  Originally written in Java by
Kevin Glass: http://www.cokeandcode.com/main/tutorials/path-finding/ and converted to Python, with modifications, by
Peter Moore.  The main modifications,
    1. Create_paths, which finds solutions to all nodes within search_distance
    3. Support for adding costs as if they are probabilities via p_s = 1-(1-p1)*(1-p2)
    3. Simple class implementations as well as run code that demonstrates the results via main()
"""

from typing import Optional, List, Dict
from ffai.core.model import *
from ffai.core.procedure import *
from ffai.core.table import *
from ffai.core.game import Game
from ffai.ai.registry import register_bot
import time
import copy
from functools import lru_cache


class Path:

    def __init__(self, steps: List['Square'], prob: float):
        self.steps = steps
        self.prob = prob

    def __len__(self) -> int:
        return len(self.steps)

    def get_last_step(self) -> 'Square':
        return self.steps[-1]

    def is_empty(self) -> bool:
        return len(self) == 0


class Node:

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.prob: float = 1
        self.moves: float = 0
        self.__parent: Optional[Node] = None
        self.depth: int = 0

    @property
    def parent(self: 'Node'):
        return self.__parent

    @parent.setter
    def parent(self: 'Node', parent: 'Node'):
        if parent is not None:
            self.depth = parent.depth + 1
        self.__parent = parent


class SortedList:

    def __init__(self, sort_lambda):
        self.list = []
        self.sort_lambda = sort_lambda

    def first(self):
        return self.list[0]

    def clear(self):
        self.list.clear()

    def append(self, o):
        self.list.append(o)
        self.list.sort(key=self.sort_lambda)

    def remove(self, o):
        self.list.remove(o)

    def __len__(self):
        return len(self.list)

    def contains(self, o):
        return o in self.list


class FFMover:
    def __init__(self, player: Player, allow_skill_reroll=True):
        self.allow_skill_reroll = allow_skill_reroll
        self.player: Player = player
        self.move_allowed: int = player.num_moves_left()
        self.cur_depth = 0


class FFTileMap:
    def __init__(self, game: Game):
        self.game: Game = game
        self.width = game.state.pitch.width
        self.height = game.state.pitch.height
        self.visited: List[List[bool]] = [[False for y in range(self.height)] for x in range(self.width)]

    def get_width_in_tiles(self) -> int:
        return self.width

    def get_height_in_tiles(self) -> int:
        return self.height

    def clear_visited(self):
        for x in range(self.get_width_in_tiles()):
            for y in range(self.get_height_in_tiles()):
                self.visited[x][y] = False

    def path_finder_visited(self, x: int, y: int):
        self.visited[x][y] = True

    def has_visited(self, x: int, y: int) -> bool:
        return self.visited[x][y]

    @lru_cache(maxsize=400)
    def blocked(self, mover: FFMover, x: int, y: int) -> bool:
        square = self.game.get_square(x, y)

        # Need to ignore the "crowd" squares on the boundary by blocking them.
        return (x <= 0) or (y <= 0) or (x >= self.width - 1) or (y >= self.height - 1) or self.game.get_player_at(
            square) is not None

    def get_prob(self, mover: FFMover, sx: int, sy: int, tx: int, ty: int) -> float:
        square_from = self.game.get_square(sx, sy)
        square_to = self.game.get_square(tx, ty)
        moving_unit = mover.player
        # A huge speed up if we can cache this function call
        dodge_prob = self.dodge_prob(moving_unit, square_from, square_to, allow_dodge_reroll=mover.allow_skill_reroll)
        move_prob = 1.0
        cur_depth: int = mover.cur_depth  # essentially number of moves already done.
        if cur_depth != -1 and (cur_depth + 1 > self.num_moves_left(moving_unit, include_gfi=False)):
            move_prob = 5.0 / 6.0
            if self.game.state.weather == WeatherType.BLIZZARD:
                move_prob = 4.0 / 6.0
            if mover.allow_skill_reroll and moving_unit.has_skill(Skill.SURE_FEET):
                move_prob += (1 - move_prob) * move_prob
        return move_prob * dodge_prob

    @lru_cache(maxsize=400)
    def num_moves_left(self, player, include_gfi=False):
        return player.num_moves_left(include_gfi=include_gfi)

    @lru_cache(maxsize=10000)
    def dodge_prob(self, moving_unit, square_from, square_to, allow_dodge_reroll):
        return self.game.get_dodge_prob_from(moving_unit, square_from, square_to, allow_dodge_reroll=allow_dodge_reroll)


class FFPathFinder:

    def __init__(self, tile_map: FFTileMap, max_search_distance: int):
        self.open: SortedList = SortedList(lambda x: x.prob)
        self.closed: List[Node] = []
        self.tile_map: FFTileMap = tile_map
        self.max_search_distance: int = max_search_distance
        self.nodes: List[List[Node]] = []
        for x in range(tile_map.get_width_in_tiles()):
            nodes_cur: List[Node] = []
            for y in range(tile_map.get_height_in_tiles()):
                nodes_cur.append(Node(x, y))
            self.nodes.append(nodes_cur)

    def find_path(self, mover: FFMover, sx: int, sy: int, tx: int = None, ty: int = None, tile: Tile = None,
                  player: Player = None) -> Optional[Path]:
        # easy first check, if the destination is blocked, we can't get there
        FFTileMap.dodge_prob.cache_clear()
        FFTileMap.num_moves_left.cache_clear()
        if tx is not None and ty is not None and self.tile_map.blocked(mover, tx, ty):
            return None

        if tx == sx and ty == sy:
            return Path([], 1.0)

        # initial state for A*. The closed group is empty. Only the starting

        # tile is in the open list and it'e're already there
        self.nodes[sx][sy].prob = 1
        self.nodes[sx][sy].moves = 0 if mover.player.state.up or mover.player.has_skill(Skill.JUMP_UP) else 3
        self.nodes[sx][sy].depth = 0
        self.closed.clear()
        self.open.clear()
        self.open.append(self.nodes[sx][sy])

        if tx is not None and ty is not None:
            self.nodes[tx][ty].parent = None

        # Make a set of goals found
        goals = set()

        # while we haven't exceeded our max search depth
        while len(self.open) != 0:
            # pull out the first node in our open list, this is determined to
            # be the most likely to be the next step based on our heuristic
            current = self.get_first_in_open()

            if tx is not None and ty is not None and current == self.nodes[tx][ty]:
                goals.add(self.tile_map.game.get_square(tx, ty))

            if tile is not None and self.tile_map.game.arena.board[current.y][current.x] == tile:
                goals.add(self.tile_map.game.get_square(current.x, current.y))

            if player is not None and player.position.distance(current) == 1:
                goals.add(self.tile_map.game.get_square(current.x, current.y))

            self.remove_from_open(current)
            self.add_to_closed(current)

            if current.moves == self.max_search_distance:
                continue
            if current.moves >= self.max_search_distance:
                current.parent = None
                continue

            # search through all the neighbours of the current node evaluating
            # them as next steps
            for x in range(-1, 2):
                for y in range(-1, 2):
                    # not a neighbour, its the current tile
                    if (x == 0) and (y == 0):
                        continue

                    # determine the location of the neighbour and evaluate it
                    xp = x + current.x
                    yp = y + current.y

                    if self.is_valid_location(mover, sx, sy, xp, yp):
                        mover.cur_depth = current.depth
                        next_step_prob = current.prob * self.tile_map.get_prob(mover, current.x, current.y, xp, yp)
                        next_step_moves = current.moves + 1
                        neighbour = self.nodes[xp][yp]
                        self.tile_map.path_finder_visited(xp, yp)

                        # if the new cost we've determined for this node is lower than
                        # it has been previously makes sure the node hasn't
                        # determined that there might have been a better path to get to
                        # this node so it needs to be re-evaluated

                        if next_step_prob > neighbour.prob:
                            if self.in_open_list(neighbour):
                                self.remove_from_open(neighbour)

                            if self.in_closed_list(neighbour):
                                self.remove_from_closed(neighbour)

                        # if the node hasn't already been processed and discarded then
                        # reset its prob to our current prob and add it as a next possible
                        # step (i.e. to the open list)
                        if not self.in_open_list(neighbour) and not (self.in_closed_list(neighbour)):
                            neighbour.prob = next_step_prob
                            neighbour.moves = next_step_moves
                            neighbour.parent = current
                            self.add_to_open(neighbour)

        # Search is over - backtrack for goals to find safest path
        best_path = None
        for goal in goals:
            path = self.create_path(sx, sy, goal.x, goal.y)
            if best_path is None or path.prob > best_path.prob:
                best_path = path
            if best_path is not None and path.prob == best_path.prob and len(path.steps) < len(best_path.steps):
                best_path = path
        return best_path

    def create_path(self, sx: int, sy: int, tx: int, ty: int) -> Optional[Path]:
        if tx == sx and ty == sy:
            return Path([], 1.0)
        if tx is None or ty is None:
            return None
        target = self.nodes[tx][ty]
        target_prob: float = target.prob
        if target.parent is None:
            return None
        path_steps: List[Square] = []
        while target != self.nodes[sx][sy]:
            path_steps.insert(0, self.tile_map.game.get_square(target.x, target.y))
            target = target.parent
        return Path(path_steps, target_prob)

    def find_paths(self, mover, sx: int, sy: int) -> List[Path]:
        """
        Find all paths up to self.max_search_distance starting from (sx, sy).
        :return: 3-D List of either Paths (where a path to the node exists) or None, where no Path exists
        """
        t0 = time.time()
        self.nodes[sx][sy].prob = 1
        self.nodes[sx][sy].depth = 0
        self.nodes[sx][sy].moves = 0
        self.closed.clear()
        self.open.clear()
        self.open.append(self.nodes[sx][sy])  # Start with starting node.

        while len(self.open) > 0:
            current = self.get_first_in_open()
            self.remove_from_open(current)
            self.add_to_closed(current)
            if current.moves == self.max_search_distance:
                continue
            if current.moves >= self.max_search_distance:
                current.parent = None
                continue
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x == 0 and y == 0:
                        continue

                    xp = x + current.x
                    yp = y + current.y

                    if self.is_valid_location(mover, sx, sy, xp, yp):
                        mover.cur_depth = current.depth
                        next_step_prob = current.prob * self.tile_map.get_prob(mover, current.x, current.y, xp, yp)
                        next_step_moves = current.moves + 1
                        neighbour = self.nodes[xp][yp]
                        self.tile_map.path_finder_visited(xp, yp)

                        if next_step_prob > neighbour.prob:
                            if self.in_open_list(neighbour):
                                self.remove_from_open(neighbour)

                            if self.in_closed_list(neighbour):
                                self.remove_from_closed(neighbour)

                        if (not self.in_open_list(neighbour)) and (not self.in_closed_list(neighbour)):
                            neighbour.prob = next_step_prob
                            neighbour.moves = next_step_moves
                            neighbour.parent = current
                            self.add_to_open(neighbour)

        print(time.time() - t0)
        paths = self.create_paths(sx, sy)
        return paths

    def create_paths(self, sx: int, sy: int) -> List[Path]:
        paths = []
        for x in range(self.tile_map.get_width_in_tiles()):
            for y in range(self.tile_map.get_height_in_tiles()):
                if self.tile_map.has_visited(x, y):
                    node = self.nodes[x][y]
                    path_cur = self.create_path(sx, sy, x, y)
                    if path_cur is not None:
                        paths.append(path_cur)
        # l = [len(path) for path in paths]
        return paths

    def get_computed_prob(self, ix: int, iy: int) -> float:
        return self.nodes[ix][iy].prob

    def get_first_in_open(self) -> Node:
        return self.open.first()

    def add_to_open(self, node: Node):
        self.open.append(node)

    def in_open_list(self, node: Node) -> bool:
        return self.open.contains(node)

    def remove_from_open(self, node: Node):
        self.open.remove(node)

    def add_to_closed(self, node: Node):
        self.closed.append(node)

    def in_closed_list(self, node: Node) -> bool:
        return node in self.closed

    def remove_from_closed(self, node: Node):
        self.closed.remove(node)

    def is_valid_location(self, mover: FFMover, sx: int, sy: int, x: int, y: int) -> bool:
        valid = not self.tile_map.blocked(mover, x, y)
        # valid = not (sx == x and sy == y) and not self.tile_map.blocked(mover, x, y)
        # valid = 0 <= x < self.tile_map.get_width_in_tiles() and 0 <= y < self.tile_map.get_height_in_tiles()
        # valid = valid and not (sx == x and sy == y)
        # valid = valid and not self.tile_map.blocked(mover, x, y)
        return valid


def _alter_state(game, player, from_position, moves_used):
    orig_player, orig_ball = None, None
    if from_position is not None or moves_used is not None:
        orig_player = copy.deepcopy(player)
        orig_ball = copy.deepcopy(game.get_ball())
    # Move player if another starting position is used
    if from_position is not None:
        assert game.get_player_at(from_position) is None or game.get_player_at(from_position) == player
        game.move(player, from_position)
        if from_position == game.get_ball_position() and game.get_ball().on_ground:
            game.get_ball().carried = True
    if moves_used != None:
        assert moves_used >= 0
        player.state.moves = moves_used
        if moves_used > 0:
            player.state.up = True
    return orig_player, orig_ball


def _reset_state(game, player, orig_player, orig_ball):
    if orig_player is not None:
        game.move(player, orig_player.position)
        player.state = orig_player.state
    if orig_ball is not None:
        game.ball = orig_ball


def get_safest_path(game, player, position, from_position=None, num_moves_used=None, allow_skill_reroll=True,
                    max_search_distance=False):
    """
    :param game:
    :param player: the player to move
    :param position: the location to move to
    :param allow_skill_reroll: whether to allow skill re-rolls in the probability computations. If enabled, skill re-rolls can
    currently be used at every step regardless of whether it was used before.
    :param max_search_distance: the maximum search distance. If None, it will use the player's number of moves left.
    :return a path containing the list of squares that forms the safest (and thereafter shortest) path for the given player to the
    given position and the probability of success.
    """
    orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)
    FFTileMap.dodge_prob.cache_clear()
    FFTileMap.num_moves_left.cache_clear()
    FFTileMap.blocked.cache_clear()

    if game.ff_map is None:
        game.ff_map = FFTileMap(game)
    player_mover = FFMover(player, allow_skill_reroll=allow_skill_reroll)
    max_steps = player.num_moves_left() - 1 if not max_search_distance else max_search_distance
    finder = FFPathFinder(game.ff_map, max_steps)
    path = finder.find_path(player_mover, player.position.x, player.position.y, position.x, position.y)

    _reset_state(game, player, orig_player, orig_ball)

    return path


def get_safest_path_to_player(game, player, target_player, from_position=None, num_moves_used=None,
                              allow_skill_reroll=True, max_search_distance=False):
    """
    :param game:
    :param player: the player to move
    :param target_player: the player to move adjacent to
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param allow_skill_reroll: whether to allow skill re-rolls in the probability computations. If enabled, skill re-rolls can
    currently be used at every step regardless of whether it was used before.
    :param max_search_distance: the maximum search distance. If None, it will use the player's number of moves left.
    :return a path containing the list of squares that forms the safest (and thereafter shortest) path for the given player to the
    a position that is adjacent to the other player and the probability of success.
    """
    orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)
    FFTileMap.dodge_prob.cache_clear()
    FFTileMap.num_moves_left.cache_clear()
    FFTileMap.blocked.cache_clear()

    if game.ff_map is None:
        game.ff_map = FFTileMap(game)
    player_mover = FFMover(player, allow_skill_reroll=allow_skill_reroll)
    max_steps = player.num_moves_left() - 1 if not max_search_distance else max_search_distance
    finder = FFPathFinder(game.ff_map, max_steps)
    path = finder.find_path(player_mover, player.position.x, player.position.y, player=target_player)

    _reset_state(game, player, orig_player, orig_ball)

    return path


def get_all_paths(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True,
                  max_search_distance=False):
    """
    :param game:
    :param player: the player to move
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param allow_skill_reroll: whether to allow skill re-rolls in the probability computations. If enabled, skill re-rolls can
    currently be used at every step regardless of whether it was used before.
    :param max_search_distance: the maximum search distance. If None, it will use the player's number of moves left.
    :return a list of paths, each containing the list of squares that forms the safest (and thereafter shortest) path for the given player to the
    given position and the probability of success, for each reachable square.
    """
    orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)
    FFTileMap.dodge_prob.cache_clear()
    FFTileMap.num_moves_left.cache_clear()
    FFTileMap.blocked.cache_clear()

    if game.ff_map is None:
        game.ff_map = FFTileMap(game)
    player_mover = FFMover(player, allow_skill_reroll=allow_skill_reroll)
    max_steps = player.num_moves_left() if not max_search_distance else max_search_distance
    finder = FFPathFinder(game.ff_map, max_steps)
    paths = finder.find_paths(player_mover, player.position.x, player.position.y)

    _reset_state(game, player, orig_player, orig_ball)

    return paths


def get_safest_scoring_path(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True,
                            max_search_distance=None):
    """
    :param game:
    :param player:
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param max_search_distance: the maximum search distance. If None, it will use the player's number of moves left.
    :param allow_skill_reroll: whether to allow skill re-rolls in the probability computations. If enabled, skill re-rolls can
    currently be used at every step regardless of whether it was used before.
    :return: the safest path to a square in the opponent endzone.
    """
    orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)

    if game.ff_map is None:
        game.ff_map = FFTileMap(game)
    player_mover = FFMover(player, allow_skill_reroll=allow_skill_reroll)
    max_steps = player.num_moves_left() if not max_search_distance else max_search_distance
    finder = FFPathFinder(game.ff_map, max_steps)
    tile = Tile.HOME_TOUCHDOWN if player.team == game.state.away_team else Tile.AWAY_TOUCHDOWN
    path = finder.find_path(player_mover, player.position.x, player.position.y, tile=tile)

    _reset_state(game, player, orig_player, orig_ball)

    return path


class ActionSequence:

    def __init__(self, action_steps: List[Action], score: float = 0, description: str = ''):
        """ Creates a new ActionSequence - an ordered list of sequential Actions to attempt to undertake.
        :param action_steps: Sequence of action steps that form this action.
        :param score: A score representing the attractiveness of the move (default: 0)
        :param description: A debug string (default: '')
        """

        # Note the intention of this object is that when the object is acting, as steps are completed,
        # they are removed from the move_sequence so the next move is always the top of the move_sequence
        # lis

        self.action_steps = action_steps
        self.score = score
        self.description = description

    def is_valid(self, game: Game) -> bool:
        pass

    def popleft(self):
        return self.action_steps.pop(0)
        # val = self.action_steps[0]
        # del self.action_steps[0]
        # return val

    def is_empty(self):
        return not self.action_steps


class FfHeatMap:
    """ A heat map of a Blood Bowl field.

    A class for analysing zones of control for both teams
    """

    def __init__(self, game: Game, team: Team):
        self.game = game
        self.team = team
        # Note that the edges are not on the field, but represent crowd squares
        self.units_friendly: List[List[float]] = [[0.0 for y in range(game.state.pitch.height)] for x in range(game.state.pitch.width)]
        self.units_opponent: List[List[float]] = [[0.0 for y in range(game.state.pitch.height)] for x in range(game.state.pitch.width)]

    def add_unit_paths(self, player: Player, paths: List[Path]):
        is_friendly: bool = player.team == self.team

        for path in paths:
            if is_friendly:
                self.units_friendly[path.steps[-1].x][path.steps[-1].y] += path.prob * path.prob
            else:
                self.units_opponent[path.steps[-1].x][path.steps[-1].y] += path.prob * path.prob

    def add_unit_by_paths(self, game: Game, paths: Dict[Player, List[Path]]):
        for player in paths.keys():
            self.add_unit_paths(player, paths[player])

    def add_players_moved(self, game: Game, players: List[Player]):
        for player in players:
            adjacents: List[Square] = game.get_adjacent_squares(player.position, occupied=True)
            self.units_friendly[player.position.x][player.position.y] += 1.0
            for adjacent in adjacents:
                self.units_friendly[player.position.x][player.position.y] += 0.5

    def get_ball_move_square_safety_score(self, square: Square) -> float:

        # Basic idea - identify safe regions to move the ball towards
        # friendly_heat: float = self.units_friendly[square.x][square.y]
        opponent_heat: float = self.units_opponent[square.x][square.y]

        score: float = 30.0 * max(0.0, (1.0 - opponent_heat / 2))

        # score: float=0.0
        # if opponent_heat < 0.25: score += 15.0
        # if opponent_heat < 0.05: score += 15.0
        # if opponent_heat < 1.5: score += 5
        # if friendly_heat > 3.5: score += 10.0
        # score += max(30.0, 5.0*(friendly_heat-opponent_heat))

        return score

    def get_cage_necessity_score(self, square: Square) -> float:
        # opponent_friendly: float = self.units_friendly[square.x][square.y]
        opponent_heat: float = self.units_opponent[square.x][square.y]
        score: float = 0.0

        if opponent_heat < 0.4:
            score -= 80.0
        # if opponent_friendly > opponent_heat: score -= max(30.0, 10.0*(opponent_friendly-opponent_heat))
        # if opponent_heat <1.5: score -=5
        # if opponent_heat > opponent_friendly: score += 10.0*(opponent_friendly-opponent_heat)

        return score


def blitz_used(game: Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == ActionType.START_BLITZ:
            return False
    return True


def handoff_used(game: Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == ActionType.START_HANDOFF:
            return False
    return True


def foul_used(game: Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == ActionType.START_FOUL:
            return False
    return True


def pass_used(game: Game) -> bool:
    for action in game.state.available_actions:
        if action.action_type == ActionType.START_PASS:
            return False
    return True


def get_players(game: Game, team: Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = True, include_used: bool = True, include_off_pitch: bool = False, only_blockable: bool = False, only_used: bool = False) -> List[Player]:
    players: List[Player] = []
    selected_players: List[Player] = []
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
                if include_off_pitch or (player.position is not None and not game.is_out_of_bounds(player.position)):
                    selected_players.append(player)

    return selected_players


def caging_squares_north_east(game: Game, protect_square: Square) -> List[Square]:

    # * At it's simplest, a cage requires 4 players in the North-East, South-East, South-West and North-West
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
    # * If the caging player is in 1-4, but next to ball carrier, he ensures this will automatically be me
    # *
    # * The only exception to this is when the ball carrier is on, or near, the sideline.  Then return the squares
    # * that can otherwise form the cage.
    # *

    caging_squares: List[Square] = []
    x = protect_square.x
    y = protect_square.y

    if x <= game.state.pitch.width - 3:
        if y == game.state.pitch.height - 2:
            caging_squares.append(game.get_square(x + 1, y + 1))
            caging_squares.append(game.get_square(x + 2, y + 1))
            caging_squares.append(game.get_square(x + 1, y))
            caging_squares.append(game.get_square(x + 2, y))
        elif y == game.state.pitch.height - 1:
            caging_squares.append(game.get_square(x + 1, y))
            caging_squares.append(game.get_square(x + 2, y))
        else:
            caging_squares.append(game.get_square(x + 1, y + 1))
            caging_squares.append(game.get_square(x + 1, y + 2))
            caging_squares.append(game.get_square(x + 2, y + 1))
            # caging_squares.append(game.state.pitch.get_square(x + 3, y + 3))

    return caging_squares


def caging_squares_north_west(game: Game, protect_square: Square) -> List[Square]:

    caging_squares: List[Square] = []
    x = protect_square.x
    y = protect_square.y

    if x >= 3:
        if y == game.state.pitch.height-2:
            caging_squares.append(game.get_square(x - 1, y + 1))
            caging_squares.append(game.get_square(x - 2, y + 1))
            caging_squares.append(game.get_square(x - 1, y))
            caging_squares.append(game.get_square(x - 2, y))
        elif y == game.state.pitch.height-1:
            caging_squares.append(game.get_square(x - 1, y))
            caging_squares.append(game.get_square(x - 2, y))
        else:
            caging_squares.append(game.get_square(x - 1, y + 1))
            caging_squares.append(game.get_square(x - 1, y + 2))
            caging_squares.append(game.get_square(x - 2, y + 1))
            # caging_squares.append(game.state.pitch.get_square(x - 3, y + 3))

    return caging_squares


def caging_squares_south_west(game: Game, protect_square: Square) -> List[Square]:

    caging_squares: List[Square] = []
    x = protect_square.x
    y = protect_square.y

    if x >= 3:
        if y == 2:
            caging_squares.append(game.get_square(x - 1, y - 1))
            caging_squares.append(game.get_square(x - 2, y - 1))
            caging_squares.append(game.get_square(x - 1, y))
            caging_squares.append(game.get_square(x - 2, y))
        elif y == 1:
            caging_squares.append(game.get_square(x - 1, y))
            caging_squares.append(game.get_square(x - 2, y))
        else:
            caging_squares.append(game.get_square(x - 1, y - 1))
            caging_squares.append(game.get_square(x - 1, y - 2))
            caging_squares.append(game.get_square(x - 2, y - 1))
            # caging_squares.append(game.state.pitch.get_square(x - 3, y - 3))

    return caging_squares


def caging_squares_south_east(game: Game, protect_square: Square) -> List[Square]:

    caging_squares: List[Square] = []
    x = protect_square.x
    y = protect_square.y

    if x <= game.state.pitch.width - 3:
        if y == 2:
            caging_squares.append(game.get_square(x + 1, y - 1))
            caging_squares.append(game.get_square(x + 2, y - 1))
            caging_squares.append(game.get_square(x + 1, y))
            caging_squares.append(game.get_square(x + 2, y))
        elif y == 1:
            caging_squares.append(game.get_square(x + 1, y))
            caging_squares.append(game.get_square(x + 2, y))
        else:
            caging_squares.append(game.get_square(x + 1, y - 1))
            caging_squares.append(game.get_square(x + 1, y - 2))
            caging_squares.append(game.get_square(x + 2, y - 1))
            # caging_squares.append(game.get_square(x + 3, y - 3))

    return caging_squares


def is_caging_position(game: Game, player: Player, protect_player: Player) -> bool:
    return player.position.distance(protect_player.position) <= 2 and not is_castle_position_of(game, player, protect_player)


def has_player_within_n_squares(game: Game, units: List[Player], square: Square, num_squares: int) -> bool:
    for cur in units:
        if cur.position.distance(square) <= num_squares:
            return True
    return False


def has_adjacent_player(game: Game, square: Square) -> bool:
    return not game.get_adjacent_players(square)


def is_castle_position_of(game: Game, player1: Player, player2: Player) -> bool:
    return player1.position.x == player2.position.x or player1.position.y == player2.position.y


def is_bishop_position_of(game: Game, player1: Player, player2: Player) -> bool:
    return abs(player1.position.x - player2.position.x) == abs(player1.position.y - player2.position.y)


def attacker_would_surf(game: Game, attacker: Player, defender: Player) -> bool:
    if (defender.has_skill(Skill.SIDE_STEP) and not attacker.has_skill(Skill.GRAB)) or defender.has_skill(Skill.STAND_FIRM):
        return False

    if not attacker.position.is_adjacent(defender.position):
        return False

    return direct_surf_squares(game, attacker.position, defender.position)


def direct_surf_squares(game: Game, attack_square: Square, defend_square: Square) -> bool:
    defender_on_sideline: bool = on_sideline(game, defend_square)
    defender_in_endzone: bool = on_endzone(game, defend_square)

    if defender_on_sideline and defend_square.x == attack_square.x:
        return True

    if defender_in_endzone and defend_square.y == attack_square.y:
        return True

    if defender_in_endzone and defender_on_sideline:
        return True

    return False


def reverse_x_for_right(game: Game, team: Team, x: int) -> int:
    if not game.is_team_side(Square(13, 3), team):
        res = game.state.pitch.width - 1 - x
    else:
        res = x
    return res


def reverse_x_for_left(game: Game, team: Team, x: int) -> int:
    if game.is_team_side(Square(13, 3), team):
        res = game.state.pitch.width - 1 - x
    else:
        res = x
    return res


def on_sideline(game: Game, square: Square) -> bool:
    return square.y == 1 or square.y == game.state.pitch.height - 1


def on_endzone(game: Game, square: Square) -> bool:
    return square.x == 1 or square.x == game.state.pitch.width - 1


def on_los(game: Game, team: Team, square: Square) -> bool:
    return (reverse_x_for_right(game, team, square.x) == 13) and 4 < square.y < 21


def los_squares(game: Game, team: Team) -> List[Square]:

    squares: List[Square] = [
        game.get_square(reverse_x_for_right(game, team, 13), 5),
        game.get_square(reverse_x_for_right(game, team, 13), 6),
        game.get_square(reverse_x_for_right(game, team, 13), 7),
        game.get_square(reverse_x_for_right(game, team, 13), 8),
        game.get_square(reverse_x_for_right(game, team, 13), 9),
        game.get_square(reverse_x_for_right(game, team, 13), 10),
        game.get_square(reverse_x_for_right(game, team, 13), 11)
    ]
    return squares


def distance_to_sideline(game: Game, square: Square) -> int:
    return min(square.y - 1, game.state.pitch.height - square.y - 2)


def is_endzone(game, square: Square) -> bool:
    return square.x == 1 or square.x == game.state.pitch.width - 1


def last_block_proc(game) -> Optional[Block]:
    for i in range(len(game.state.stack.items) - 1, -1, -1):
        if isinstance(game.state.stack.items[i], Block):
            block_proc = game.state.stack.items[i]
            return block_proc
    return None


def is_adjacent_ball(game: Game, square: Square) -> bool:
    ball_square = game.get_ball_position()
    return ball_square is not None and ball_square.is_adjacent(square)


def squares_within(game: Game, square: Square, distance: int) -> List[Square]:
    squares: List[Square] = []
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            cur_square = game.get_square(square.x + i, square.y + j)
            if cur_square != square and not game.is_out_of_bounds(cur_square):
                squares.append(cur_square)
    return squares


def distance_to_defending_endzone(game: Game, team: Team, position: Square) -> int:
    res = reverse_x_for_right(game, team, position.x) - 1
    return res


def distance_to_scoring_endzone(game: Game, team: Team, position: Square) -> int:
    res = reverse_x_for_left(game, team, position.x) - 1
    return res
    # return game.state.pitch.width - 1 - reverse_x_for_right(game, team, position.x)


def players_in_scoring_endzone(game: Game, team: Team, include_own: bool = True, include_opp: bool = False) -> List[Player]:
    players: List[Player] = get_players(game, team, include_own=include_own, include_opp=include_opp)
    selected_players: List[Player] = []
    for player in players:
        if in_scoring_endzone(game, team, player.position):
            selected_players.append(player)
    return selected_players


def in_scoring_endzone(game: Game, team: Team, square: Square) -> bool:
    return reverse_x_for_left(game, team, square.x) == 1


def players_in_scoring_distance(game: Game, team: Team, include_own: bool = True, include_opp: bool = True, include_stunned: bool = False) -> List[Player]:
    players: List[Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned)
    selected_players: List[Player] = []
    for player in players:
        if distance_to_scoring_endzone(game, team, player.position) <= player.num_moves_left():
            selected_players.append(player)
    return selected_players


def distance_to_nearest_player(game: Game, team: Team, square: Square, include_own: bool = True, include_opp: bool = True, only_used: bool = False, include_used: bool = True, include_stunned: bool = True, only_blockable: bool = False) -> int:
    opps: List[Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, only_used=only_used, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)
    cur_max = 100
    for opp in opps:
        dist = opp.position.distance(square)
        cur_max = min(cur_max, dist)
    return cur_max


def screening_distance(game: Game, from_square: Square, to_square: Square) -> float:
    # Return the "screening distance" between 3 squares.  (To complete)
    # float dist =math.sqrt(math.pow(Square.x - cur.position.x, 3) + math.pow(Square.y - cur.position.y, 3))
    return 0.0


def num_opponents_can_reach(game: Game, team: Team, square: Square) -> int:
    opps: List[Player] = get_players(game, team, include_own=False, include_opp=True)
    num_opps_reach: int = 0
    for cur in opps:
        dist = max(square.x - cur.position.x, square.y - cur.position.y)
        if cur.state.stunned:
            continue
        move_allowed = cur.get_ma() + 2
        if not cur.state.up:
            move_allowed -= 3
        if dist < move_allowed:
            num_opps_reach += 1
    return num_opps_reach


def num_opponents_on_field(game: Game, team: Team) -> int:
    opps: List[Player] = get_players(game, team, include_own=False, include_opp=True)
    num_opponents = 0
    for cur in opps:
        if cur.position is not None:
            num_opponents += 1
    return num_opponents


def number_opponents_closer_than_to_endzone(game: Game, team: Team, square: Square) -> int:
    opponents: List[Player] = get_players(game, team, include_own=False, include_opp=True)
    num_opps = 0
    distance_square_endzone = distance_to_defending_endzone(game, team, square)

    for opponent in opponents:
        distance_opponent_endzone = distance_to_defending_endzone(game, team, opponent.position)
        if distance_opponent_endzone < distance_square_endzone:
            num_opps += 1
    return num_opps


def in_scoring_range(game: Game, player: Player) -> bool:
    return player.num_moves_left() >= distance_to_scoring_endzone(game, player.team, player.position)


def players_in_scoring_range(game: Game, team: Team, include_own=True, include_opp=True, include_used=True, include_stunned=True) -> List[Player]:
    players: List[Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_stunned=include_stunned, include_used=include_used)
    res: List[Player] = []
    for player in players:
        if in_scoring_range(game, player):
            res.append(player)
    return res


def players_in(game: Game, team: Team, squares: List[Square], include_own=True, include_opp=True, include_used=True, include_stunned=True, only_blockable=False) -> List[Player]:

    allowed_players: List[Player] = get_players(game, team, include_own=include_own, include_opp=include_opp, include_used=include_used, include_stunned=include_stunned, only_blockable=only_blockable)
    res: List[Player] = []

    for square in squares:
        player: Optional[Player] = game.get_player_at(square)
        if player is None:
            continue
        if player in allowed_players:
            res.append(player)
    return res


class GrodBot(Agent):
    """
    A Bot that uses path finding to evaluate all possibilities.

    WIP!!! Hand-offs and Pass actions going a bit funny.

    """

    mean_actions_available = []
    steps = []

    BASE_SCORE_BLITZ = 60.0
    BASE_SCORE_FOUL = -50.0
    BASE_SCORE_BLOCK = 65   # For a two dice block
    BASE_SCORE_HANDOFF = 40.0
    BASE_SCORE_PASS = 40.0
    BASE_SCORE_MOVE_TO_OPPONENT = 45.0
    BASE_SCORE_MOVE_BALL = 45.0
    BASE_SCORE_MOVE_TOWARD_BALL = 45.0
    BASE_SCORE_MOVE_TO_SWEEP = 0.0
    BASE_SCORE_CAGE_BALL = 70.0
    BASE_SCORE_MOVE_TO_BALL = 60.0
    BASE_SCORE_BALL_AND_CHAIN = 75.0
    BASE_SCORE_DEFENSIVE_SCREEN = 0.0
    ADDITIONAL_SCORE_DODGE = 0.0  # Lower this value to dodge more.
    ADDITIONAL_SCORE_NEAR_SIDELINE = -20.0
    ADDITIONAL_SCORE_SIDELINE = -40.0

    def __init__(self, name):
        super().__init__(name)
        self.my_team = None
        self.opp_team = None
        self.current_move: Optional[ActionSequence] = None
        self.verbose = True
        self.debug = False
        self.heat_map: Optional[FfHeatMap] = None
        self.actions_available = []

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_debug(self, debug):
        self.debug = debug

    def act(self, game):

        # Refresh my_team and opp_team (they seem to be copies)
        proc = game.state.stack.peek()
        available_actions = game.state.available_actions
        available_action_types = [available_action.action_type for available_action in available_actions]

        # Update local my_team and opp_team variables to latest copy (to ensure fresh data)
        if hasattr(proc, 'team'):
            assert proc.team == self.my_team
            self.my_team = proc.team
            self.opp_team = game.get_opp_team(self.my_team)

        # For statistical purposes, keeps a record of # action choices.
        available = 0
        for action_choice in available_actions:
            if len(action_choice.positions) == 0 and len(action_choice.players) == 0:
                available += 1
            elif len(action_choice.positions) > 0:
                available += len(action_choice.positions)
            else:
                available += len(action_choice.players)
        self.actions_available.append(available)

        # Evaluate appropriate action for each possible procedure
        if isinstance(proc, CoinTossFlip):
            action = self.coin_toss_flip(game)
        elif isinstance(proc, CoinTossKickReceive):
            action = self.coin_toss_kick_receive(game)
        elif isinstance(proc, Setup):
            action = self.setup(game)
        elif isinstance(proc, PlaceBall):
            action = self.place_ball(game)
        elif isinstance(proc, HighKick):
            action = self.high_kick(game)
        elif isinstance(proc, Touchback):
            action = self.touchback(game)
        elif isinstance(proc, Turn) and proc.quick_snap:
            action = self.quick_snap(game)
        elif isinstance(proc, Turn) and proc.blitz:
            action = self.blitz(game)
        elif isinstance(proc, Turn):
            action = self.turn(game)
        elif isinstance(proc, PlayerAction):
            action = self.player_action(game)
        elif isinstance(proc, Block):
            action = self.block(game)
        elif isinstance(proc, Push):
            action = self.push(game)
        elif isinstance(proc, FollowUp):
            action = self.follow_up(game)
        elif isinstance(proc, Apothecary):
            action = self.apothecary(game)
        elif isinstance(proc, PassAction):
            action = self.pass_action(game)
        elif isinstance(proc, Catch):
            action = self.catch(game)
        elif isinstance(proc, Interception):
            action = self.interception(game)
        elif isinstance(proc, Reroll):
            action = self.reroll(game)
        elif isinstance(proc, Shadowing):
            action = self.shadowing(game)
        else:
            if self.debug:
                raise Exception("Unknown procedure: ", proc)
            elif ActionType.USE_SKILL in available_action_types:
                # Catch-all for things like Break Tackle, Diving Tackle etc
                return Action(ActionType.USE_SKILL)
            else:
                # Ugly catch-all -> simply pick an action
                action_choice = available_actions[0]
                player = action_choice.players[0] if action_choice.players else None
                position = action_choice.positions[0] if action_choice.positions else None
                action = Action(action_choice.action_type, position=position, player=player)
                # raise Exception("Unknown procedure: ", proc)

        # Check returned Action is valid
        action_found = False
        for available_action in available_actions:
            if isinstance(action.action_type, type(available_action.action_type)):
                if available_action.players and available_action.positions:
                    action_found = (action.player in available_action.players) and (action.player in available_action.players)
                elif available_action.players:
                    action_found = action.player in available_action.players
                elif available_action.positions:
                    action_found = action.position in available_action.positions
                else:
                    action_found = True
        if not action_found:
            if self.debug:
                raise Exception('Invalid action')
            else:
                # Ugly catch-all -> simply pick an action
                action_choice = available_actions[0]
                player = action_choice.players[0] if action_choice.players else None
                position = action_choice.positions[0] if action_choice.positions else None
                action = Action(action_choice.action_type, position=position, player=player)

        if self.verbose:
            current_team = game.state.current_team.name if game.state.current_team is not None else available_actions[0].team.name
            print('      Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + current_team + ', Action=' + action.action_type.name)

        return action

    def reroll(self, game):
        proc = game.state.stack.peek()
        # target_roll = proc.context.roll.target
        # target_higher = proc.context.roll.target_higher
        # dice = proc.context.roll.dice
        # num_dice = len(dice)
        if proc.can_use_pro:
            return Action(ActionType.USE_SKILL)
        if isinstance(proc.context, GFI):
            return Action(ActionType.USE_REROLL)
        if isinstance(proc.context, Dodge):
            return Action(ActionType.USE_REROLL)
        if isinstance(proc.context, Catch):
            return Action(ActionType.USE_REROLL)
        if isinstance(proc.context, Pickup):
            return Action(ActionType.USE_REROLL)
        else:
            return Action(ActionType.USE_REROLL)

    def new_game(self, game: Game, team):
        """
        Called when a new game starts.
        """
        self.my_team = team
        self.opp_team = game.get_opp_team(team)
        self.actions_available = []

    def coin_toss_flip(self, game: Game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.TAILS)
        # return Action(ActionType.HEADS)

    def coin_toss_kick_receive(self, game: Game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.RECEIVE)
        # return Action(ActionType.KICK)

    def setup(self, game: Game) -> Action:
        """
        Move players from the reserves to the pitch
        """

        if isinstance(game.state.stack.peek(), Setup):
            proc: Setup = game.state.stack.peek()
        else:
            raise ValueError('Setup procedure expected')

        if proc.reorganize:
            # We are dealing with perfect defence.  For now do nothing, but we could send all players back to reserve box
            action_steps: List[Action] = [Action(ActionType.END_SETUP)]
            self.current_move = ActionSequence(action_steps, description='Perfect Defence do nothing')

        else:

            if not get_players(game, self.my_team, include_own=True, include_opp=False, include_off_pitch=False):
                # If no players are on the pitch yet, create a new ActionSequence for the setup.
                action_steps: List[Action] = []

                turn = game.state.round
                half = game.state.half
                opp_score = 0
                for team in game.state.teams:
                    if team != self.my_team:
                        opp_score = max(opp_score, team.state.score)
                score_diff = self.my_team.state.score - opp_score

                # Choose 11 best players to field
                players_available: List[Player] = []
                for available_action in game.state.available_actions:
                    if available_action.action_type == ActionType.PLACE_PLAYER:
                        players_available = available_action.players

                players_sorted_value = sorted(players_available, key=lambda x: player_value(game, x), reverse=True)
                n_keep: int = min(11, len(players_sorted_value))
                players_available = players_sorted_value[:n_keep]

                # Are we kicking or receiving?
                if game.state.receiving_this_drive:
                    place_squares: List[Square] = [
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 7),
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 8),
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 9),
                        # Receiver next
                        game.get_square(reverse_x_for_right(game, self.my_team, 8), 8),
                        # Support line players
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 10),
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 11),
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 5),
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 13),
                        # A bit wide semi-defensive
                        game.get_square(reverse_x_for_right(game, self.my_team, 11), 4),
                        game.get_square(reverse_x_for_right(game, self.my_team, 11), 12),
                        # Extra help at the back
                        game.get_square(reverse_x_for_right(game, self.my_team, 10), 8)
                    ]
                    players_sorted_bash = sorted(players_available, key=lambda x: player_bash_ability(game, x), reverse=True)
                    players_sorted_blitz = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)

                else:
                    place_squares: List[Square] = [

                        # LOS squares first
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 7),
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 8),
                        game.get_square(reverse_x_for_right(game, self.my_team, 13), 9),

                        game.get_square(reverse_x_for_right(game, self.my_team, 12), 3),
                        game.get_square(reverse_x_for_right(game, self.my_team, 12), 13),
                        game.get_square(reverse_x_for_right(game, self.my_team, 11), 2),
                        game.get_square(reverse_x_for_right(game, self.my_team, 11), 14),

                        game.get_square(reverse_x_for_right(game, self.my_team, 12), 5),
                        game.get_square(reverse_x_for_right(game, self.my_team, 12), 10),
                        game.get_square(reverse_x_for_right(game, self.my_team, 11), 11),
                        game.get_square(reverse_x_for_right(game, self.my_team, 11), 5)
                        ]

                    players_sorted_bash = sorted(players_available, key=lambda x: player_bash_ability(game, x), reverse=True)
                    players_sorted_blitz = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)

                for i in range(len(players_available)):
                    action_steps.append(Action(ActionType.PLACE_PLAYER, player=players_sorted_bash[i], position=place_squares[i]))

                action_steps.append(Action(ActionType.END_SETUP))

                self.current_move = ActionSequence(action_steps, description='Setup')

        # We must have initialised the action sequence, lets execute it
        if self.current_move.is_empty():
            raise Exception('what')
        else:
            next_action: Action = self.current_move.popleft()
        return next_action

    def place_ball(self, game: Game):
        """
        Place the ball when kicking.
        """

        # Note left_center square is 7,8
        center_opposite: Square = Square(reverse_x_for_left(game, self.my_team, 7), 8)
        return Action(ActionType.PLACE_BALL, position=center_opposite)

    def high_kick(self, game: Game):
        """
        Select player to move under the ball.
        """
        ball_pos = game.get_ball_position()
        if game.is_team_side(game.get_ball_position(), self.my_team) and game.get_player_at(game.get_ball_position()) is None:
            players_available = [player for player in game.get_players_on_pitch(self.my_team, up=True) if game.num_tackle_zones_in(player) == 0]
            if players_available:
                players_sorted = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)
                player = players_sorted[0]
                return Action(ActionType.SELECT_PLAYER, player=player, position=ball_pos)
        return Action(ActionType.SELECT_NONE)

    def touchback(self, game: Game):
        """
        Select player to give the ball to.
        """
        players_available = game.get_players_on_pitch(self.my_team, up=True)
        if players_available:
            players_sorted = sorted(players_available, key=lambda x: player_blitz_ability(game, x), reverse=True)
            player = players_sorted[0]
            return Action(ActionType.SELECT_PLAYER, player=player)
        return Action(ActionType.SELECT_NONE)

    def set_next_move(self, game: Game):
        """ Set self.current_move

        :param game:
        """
        self.current_move = None

        players_moved: List[Player] = get_players(game, self.my_team, include_own=True, include_opp=False, include_used=True, only_used=False)
        players_to_move: List[Player] = get_players(game, self.my_team, include_own=True, include_opp=False, include_used=False)
        paths_own: Dict[Player, List[Path]] = dict()
        for player in players_to_move:
            paths = get_all_paths(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=player.num_moves_left())
            paths_own[player] = paths

        players_opponent: List[Player] = get_players(game, self.my_team, include_own=False, include_opp=True, include_stunned=False)
        paths_opposition: Dict[Player, List[Path]] = dict()
        for player in players_opponent:
            paths = get_all_paths(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=player.num_moves_left())
            paths_opposition[player] = paths

        # Create a heat-map of control zones
        heat_map: FfHeatMap = FfHeatMap(game, self.my_team)
        heat_map.add_unit_by_paths(game, paths_opposition)
        heat_map.add_unit_by_paths(game, paths_own)
        heat_map.add_players_moved(game, get_players(game, self.my_team, include_own=True, include_opp=False, only_used=True))
        self.heat_map = heat_map

        all_actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            if action_choice.action_type == ActionType.START_MOVE:
                players_available: List[Player] = action_choice.players
                for player in players_available:
                    paths = paths_own[player]
                    all_actions.extend(potential_move_actions(game, heat_map, player, paths))
            elif action_choice.action_type == ActionType.START_BLITZ:
                players_available: List[Player] = action_choice.players
                for player in players_available:
                    paths = get_all_paths(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=player.num_moves_left()-1)
                    all_actions.extend(potential_blitz_actions(game, heat_map, player, paths))
            elif action_choice.action_type == ActionType.START_FOUL:
                players_available: List[Player] = action_choice.players
                for player in players_available:
                    paths = paths_own[player]
                    all_actions.extend(potential_foul_actions(game, heat_map, player, paths))
            elif action_choice.action_type == ActionType.START_BLOCK:
                players_available: List[Player] = action_choice.players
                for player in players_available:
                    all_actions.extend(potential_block_actions(game, heat_map, player))
            elif action_choice.action_type == ActionType.START_PASS:
                players_available: List[Player] = action_choice.players
                for player in players_available:
                    player_square: Square = player.position
                    if game.get_ball_position() == player_square:
                        paths = paths_own[player]
                        all_actions.extend(potential_pass_actions(game, heat_map, player, paths))
            elif action_choice.action_type == ActionType.START_HANDOFF:
                players_available: List[Player] = action_choice.players
                for player in players_available:
                    player_square: Square = player.position
                    if game.get_ball_position() == player_square:
                        paths = paths_own[player]
                        all_actions.extend(potential_handoff_actions(game, heat_map, player, paths))
            elif action_choice.action_type == ActionType.END_TURN:
                all_actions.extend(potential_end_turn_action(game))

        if all_actions:
            all_actions.sort(key=lambda x: x.score, reverse=True)
            self.current_move = all_actions[0]

            if self.verbose:
                print('   Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + game.state.current_team.name + ', Action=' + self.current_move.description + ', Score=' + str(self.current_move.score))

    def set_continuation_move(self, game: Game):
        """ Set self.current_move

        :param game:
        """
        self.current_move = None

        player: Player = game.state.active_player
        paths = get_all_paths(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=player.num_moves_left() - 1)

        all_actions: List[ActionSequence] = []
        for action_choice in game.state.available_actions:
            if action_choice.action_type == ActionType.MOVE:
                players_available: List[Player] = action_choice.players
                all_actions.extend(potential_move_actions(game, self.heat_map, player, paths, is_continuation=True))
            elif action_choice.action_type == ActionType.END_PLAYER_TURN:
                all_actions.extend(potential_end_player_turn_action(game, self.heat_map, player))

        if all_actions:
            all_actions.sort(key=lambda x: x.score, reverse=True)
            self.current_move = all_actions[0]

            if self.verbose:
                print('   Turn=H' + str(game.state.half) + 'R' + str(game.state.round) + ', Team=' + game.state.current_team.name + ', Action=Continue Move + ' + self.current_move.description + ', Score=' + str(self.current_move.score))

    def turn(self, game: Game) -> Action:
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
        next_action: Action = self.current_move.popleft()
        return next_action

    def quick_snap(self, game: Game):

        self.current_move = None
        return Action(ActionType.END_TURN)

    def blitz(self, game: Game):

        self.current_move = None
        return Action(ActionType.END_TURN)

    def player_action(self, game: Game):
        """
        Take the next action from the current stack and execute
        """
        if self.current_move.is_empty():
            self.set_continuation_move(game)

        action_step = self.current_move.popleft()
        return action_step

    def shadowing(self, game: Game):
        """
        Select block die or reroll.
        """
        # Loop through available dice results
        proc = game.state.stack.peek()
        return Action(ActionType.USE_SKILL)

    def block(self, game: Game):
        """
        Select block die or reroll.
        """
        # Loop through available dice results
        proc = game.state.stack.peek()
        if proc.waiting_juggernaut:
            return Action(ActionType.USE_SKILL)
        if proc.waiting_wrestle_attacker or proc.waiting_wrestle_defender:
            return Action(ActionType.USE_SKILL)

        active_player: Player = game.state.active_player
        attacker: Player = game.state.stack.items[-1].attacker
        defender: Player = game.state.stack.items[-1].defender
        favor: Team = game.state.stack.items[-1].favor

        actions: List[ActionSequence] = []
        check_reroll = False
        for action_choice in game.state.available_actions:
            if action_choice.action_type == ActionType.USE_REROLL:
                check_reroll = True
                continue
            action_steps: List[Action] = [
                Action(action_choice.action_type)
                ]
            score = block_favourability(action_choice.action_type, self.my_team, active_player, attacker, defender, favor)
            actions.append(ActionSequence(action_steps, score=score, description='Block die choice'))

        if check_reroll and check_reroll_block(game, self.my_team, actions, favor):
            return Action(ActionType.USE_REROLL)
        else:
            actions.sort(key=lambda x: x.score, reverse=True)
            current_move = actions[0]
            return current_move.action_steps[0]

    def push(self, game: Game):
        """
        Select square to push to.
        """
        # Loop through available squares
        block_proc: Optional[Block] = last_block_proc(game)
        attacker: Player = block_proc.attacker
        defender: Player = block_proc.defender
        is_blitz_action = block_proc.blitz
        score: float = -100.0
        for to_square in game.state.available_actions[0].positions:
            cur_score = score_push(game, defender.position, to_square)
            if cur_score > score:
                score = cur_score
                push_square = to_square
        return Action(ActionType.PUSH, position=push_square)

    def follow_up(self, game: Game):
        """
        Follow up or not. ActionType.FOLLOW_UP must be used together with a position.
        """
        player = game.state.active_player
        do_follow = check_follow_up(game)
        for position in game.state.available_actions[0].positions:
            if do_follow and player.position != position:
                return Action(ActionType.FOLLOW_UP, position=position)
            elif not do_follow and player.position == position:
                return Action(ActionType.FOLLOW_UP, position=position)

    def apothecary(self, game: Game):
        """
        Use apothecary?
        """
        # Update here -> apothecary BH in first half, KO or BH in second half
        return Action(ActionType.USE_APOTHECARY)
        # return Action(ActionType.DONT_USE_APOTHECARY)

    def interception(self, game: Game):
        """
        Select interceptor.
        """
        for action in game.state.available_actions:
            if action.action_type == ActionType.SELECT_PLAYER:
                for player, agi_rolls in zip(action.players, action.agi_rolls):
                    return Action(ActionType.SELECT_PLAYER, player=player)
        return Action(ActionType.SELECT_NONE)

    def pass_action(self, game: Game):
        """
        Reroll or not.
        """
        return Action(ActionType.USE_REROLL)
        # return Action(ActionType.DONT_USE_REROLL)

    def end_game(self, game: Game):
        """
        Called when a game end.
        """
        print(f'''Result for {self.name}''')
        print('------------------')
        print(f'''Num steps: {len(self.actions_available)}''')
        print(f'''Avg. branching factor: {np.mean(self.actions_available)}''')
        GrodBot.steps.append(len(self.actions_available))
        GrodBot.mean_actions_available.append(np.mean(self.actions_available))
        print(f'''Avg. Num steps: {np.mean(GrodBot.steps)}''')
        print(f'''Avg. overall branching factor: {np.mean(GrodBot.mean_actions_available)}''')
        winner = game.get_winner()
        print(f'''Casualties: {game.state.home_team.name} ({game.home_agent.name}): {game.num_casualties(game.state.home_team)} ... {game.state.away_team.name}  ({game.away_agent.name}): {game.num_casualties(game.state.away_team)}''')
        print(f'''Score: {game.state.home_team.name} ({game.home_agent.name}): {game.state.home_team.state.score} ... {game.state.away_team.name}  ({game.away_agent.name}): {game.state.away_team.state.score}''')
        if winner is None:
            print(f'''It's a draw''')
        elif winner == self:
            print(f'''I won''')
        else:
            print(f'''I lost''')
        print('------------------')


def block_favourability(block_result: ActionType, team: Team, active_player: Player, attacker: Player, defender: Player, favor: Team) -> float:

    if attacker.team == active_player.team:
        if block_result == ActionType.SELECT_DEFENDER_DOWN:
            return 6.0
        elif block_result == ActionType.SELECT_DEFENDER_STUMBLES:
            if defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE):
                return 4.0       # push back
            else:
                return 6.0
        elif block_result == ActionType.SELECT_PUSH:
            return 4.0
        elif block_result == ActionType.SELECT_BOTH_DOWN:
            if defender.has_skill(Skill.BLOCK) and not attacker.has_skill(Skill.BLOCK):
                return 1.0        # skull
            elif not attacker.has_skill(Skill.BLOCK):
                return 2                                            # both down
            elif attacker.has_skill(Skill.BLOCK) and defender.has_skill(Skill.BLOCK):
                return 3.0          # nothing happens
            else:
                return 5.0                                                                                  # only defender is down
        elif block_result == ActionType.SELECT_ATTACKER_DOWN:
            return 1.0                                                                                        # skull
    else:
        if block_result == ActionType.SELECT_DEFENDER_DOWN:
            return 1.0                                                                                        # least favourable
        elif block_result == ActionType.SELECT_DEFENDER_STUMBLES:
            if defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE):
                return 3       # not going down, so I like this.
            else:
                return 1.0                                                                                  # splat.  No good.
        elif block_result == ActionType.SELECT_PUSH:
            return 3.0
        elif block_result == ActionType.SELECT_BOTH_DOWN:
            if not attacker.has_skill(Skill.BLOCK) and defender.has_skill(Skill.BLOCK):
                return 6.0        # Attacker down, I am not.
            if not attacker.has_skill(Skill.BLOCK) and not defender.has_skill(Skill.BLOCK):
                return 5.0    # Both down is pretty good.
            if attacker.has_skill(Skill.BLOCK) and not defender.has_skill(Skill.BLOCK):
                return 2.0        # Just I splat
            else:
                return 4.0                                                                                  # Nothing happens (both have block).
        elif block_result == ActionType.SELECT_ATTACKER_DOWN:
            return 6.0                                                                                        # most favourable!

    return 0.0


def potential_end_player_turn_action(game: Game, heat_map, player: Player) -> List[ActionSequence]:
    actions: List[ActionSequence] = []
    action_steps: List[Action] = [
        Action(ActionType.END_PLAYER_TURN, player=player)
        ]
    # End turn happens on a score of 1.0.  Any actions with a lower score are never selected.
    actions.append(ActionSequence(action_steps, score=1.0, description='End Turn'))
    return actions


def potential_end_turn_action(game: Game) -> List[ActionSequence]:
    actions: List[ActionSequence] = []
    action_steps: List[Action] = [
        Action(ActionType.END_TURN)
        ]
    # End turn happens on a score of 1.0.  Any actions with a lower score are never selected.
    actions.append(ActionSequence(action_steps, score=1.0, description='End Turn'))
    return actions


def potential_block_actions(game: Game, heat_map: FfHeatMap, player: Player) -> List[ActionSequence]:

    # Note to self: need a "stand up and end move option.
    move_actions: List[ActionSequence] = []
    if not player.state.up:
        # There is currently a bug in the controlling logic.  Prone players shouldn't be able to block
        return move_actions
    blockable_players: List[Player] = game.get_adjacent_opponents(player, standing=True, stunned=False, down=False)
    for blockable_player in blockable_players:
        action_steps: List[Action] = [
            Action(ActionType.START_BLOCK, player=player),
            Action(ActionType.BLOCK, position=blockable_player.position, player=player),
            Action(ActionType.END_PLAYER_TURN, player=player)
        ]

        action_score = score_block(game, heat_map, player, blockable_player)
        score = action_score

        move_actions.append(ActionSequence(action_steps, score=score, description='Block ' + player.name + ' to (' + str(blockable_player.position.x) + ',' + str(blockable_player.position.y) + ')'))
        # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_blitz_actions(game: Game, heat_map: FfHeatMap, player: Player, paths: List[Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: Square = game.get_square(path.steps[-1].x, path.steps[-1].y)
        blockable_players = game.get_adjacent_players(end_square, team=game.get_opp_team(player.team), down=False, stunned=False)
        blockable_squares = [player.position for player in blockable_players]
        for blockable_square in blockable_squares:
            action_steps: List[Action] = []
            action_steps.append(Action(ActionType.START_BLITZ, player=player))
            if not player.state.up:
                action_steps.append(Action(ActionType.STAND_UP, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(Action(ActionType.MOVE, position=game.get_square(step.x, step.y), player=player))
            action_steps.append(Action(ActionType.BLOCK, position=blockable_square, player=player))
            # action_steps.append(Action(ActionType.END_PLAYER_TURN, player=player))

            action_score = score_blitz(game, heat_map, player, end_square, game.get_player_at(blockable_square))
            path_score = path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Blitz ' + player.name + ' to ' + str(blockable_square.x) + ',' + str(blockable_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_pass_actions(game: Game, heat_map: FfHeatMap, player: Player, paths: List[Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: Square = game.get_square(path.steps[-1].x, path.steps[-1].y)
        # Need possible receving players
        to_squares, distances = game.get_pass_distances_at(player, end_square)
        for to_square in to_squares:
            action_steps: List[Action] = []
            action_steps.append(Action(ActionType.START_PASS, player=player))

            receiver: Optional[Player] = game.get_player_at(to_square)

            if not player.state.up:
                action_steps.append(Action(ActionType.STAND_UP, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(Action(ActionType.MOVE, position=game.get_square(step.x, step.y), player=player))
            action_steps.append(Action(ActionType.PASS, position=to_square, player=player))
            action_steps.append(Action(ActionType.END_PLAYER_TURN, player=player))

            action_score = score_pass(game, heat_map, player, end_square, to_square)
            path_score = path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Pass ' + player.name + ' to ' + str(to_square.x) + ',' + str(to_square.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_handoff_actions(game: Game, heat_map: FfHeatMap, player: Player, paths: List[Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: Square = game.get_square(path.steps[-1].x, path.steps[-1].y)
        handoffable_players = game.get_adjacent_players(end_square, team=player.team, standing=True, down=False, stunned=False)
        for handoffable_player in handoffable_players:
            action_steps: List[Action] = []
            action_steps.append(Action(ActionType.START_HANDOFF, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(Action(ActionType.MOVE, position=game.get_square(step.x, step.y), player=player))
            action_steps.append(Action(ActionType.HANDOFF, position=handoffable_player.position, player=player))
            action_steps.append(Action(ActionType.END_PLAYER_TURN, player=player))

            action_score = score_handoff(game, heat_map, player, handoffable_player, end_square)
            path_score = path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Handoff ' + player.name + ' to ' + str(handoffable_player.position.x) + ',' + str(handoffable_player.position.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_foul_actions(game: Game, heat_map: FfHeatMap, player: Player, paths: List[Path]) -> List[ActionSequence]:
    move_actions: List[ActionSequence] = []
    for path in paths:
        path_steps = path.steps
        end_square: Square = game.get_square(path.steps[-1].x, path.steps[-1].y)
        foulable_players = game.get_adjacent_players(end_square, team=game.get_opp_team(player.team),  standing=False, stunned=True, down=True)
        for foulable_player in foulable_players:
            action_steps: List[Action] = []
            action_steps.append(Action(ActionType.START_FOUL, player=player))
            if not player.state.up:
                action_steps.append(Action(ActionType.STAND_UP, player=player))
            for step in path_steps:
                # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
                action_steps.append(Action(ActionType.MOVE, position=game.get_square(step.x, step.y)))
            action_steps.append(Action(ActionType.FOUL, foulable_player.position, player=player))
            action_steps.append(Action(ActionType.END_PLAYER_TURN, player=player))

            action_score = score_foul(game, heat_map, player, foulable_player, end_square)
            path_score = path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
            score = action_score + path_score

            move_actions.append(ActionSequence(action_steps, score=score, description='Foul ' + player.name + ' to ' + str(foulable_player.position.x) + ',' + str(foulable_player.position.y)))
            # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def potential_move_actions(game: Game, heat_map: FfHeatMap, player: Player, paths: List[Path], is_continuation: bool = False) -> List[ActionSequence]:

    move_actions: List[ActionSequence] = []
    ball_square: Square = game.get_ball_position()
    for path in paths:
        path_steps = path.steps
        action_steps: List[Action] = []
        if not is_continuation:
            action_steps.append(Action(ActionType.START_MOVE, player=player))
        if not player.state.up:
            action_steps.append(Action(ActionType.STAND_UP, player=player))
        for step in path_steps:
            # Note we need to add 1 to x and y because the outermost layer of squares is not actually reachable
            action_steps.append(Action(ActionType.MOVE, position=game.get_square(step.x, step.y), player=player))

        to_square: Square = game.get_square(path_steps[-1].x, path_steps[-1].y)
        action_score, is_complete, description = score_move(game, heat_map, player, to_square)
        if is_complete:
            action_steps.append(Action(ActionType.END_PLAYER_TURN, player=player))

        path_score = path_cost_to_score(path)  # If an extra GFI required for block, should increase here.  To do.
        if is_continuation and path_score > 0:
            # Continuing actions (after a Blitz block for example) may choose risky options, so penalise
            path_score = -10 + path_score * 2
        score = action_score + path_score

        move_actions.append(ActionSequence(action_steps, score=score, description='Move: ' + description + ' ' + player.name + ' to ' + str(path_steps[-1].x) + ',' + str(path_steps[-1].y)))
        # potential action -> sequence of steps such as "START_MOVE, MOVE (to square) etc
    return move_actions


def score_blitz(game: Game, heat_map: FfHeatMap, attacker: Player, block_from_square: Square, defender: Player) -> float:
    score: float = GrodBot.BASE_SCORE_BLITZ

    ball_carrier: Optional[Player] = game.get_ball_carrier()
    is_ball_carrier = attacker == ball_carrier

    num_block_dice: int = game.num_block_dice_at(attacker, defender, block_from_square, blitz=True, dauntless_success=False)
    ball_position: Player = game.get_ball_position()
    if num_block_dice == 3:
        score += 30.0
    if num_block_dice == 2:
        score += 10.0
    if num_block_dice == 1:
        score += -30.0
    if num_block_dice == -2:
        score += -75.0
    if num_block_dice == -3:
        score += -100.0
    if attacker.has_skill(Skill.BLOCK):
        score += 20.0
    if defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE):
        score -= 10.0
    if defender.has_skill(Skill.BLOCK):
        score += -10.0
    if ball_position == attacker.position:
        if attacker.position.is_adjacent(defender.position) and block_from_square == attacker.position:
            score += 20.0  # Favour blitzing with ball carrier at start of move
        else:
            score += -40.0  # But don't blitz with ball carrier after that
    if defender.position == ball_position:
        score += 50.0              # Blitzing ball carrier
    if defender.position.is_adjacent(ball_position):
        score += 20.0   # Blitzing someone adjacent to ball carrier
    if direct_surf_squares(game, block_from_square, defender.position):
        score += 25.0  # A surf
    if game.get_adjacent_opponents(attacker, stunned=False, down=False) and not is_ball_carrier:
        score -= 10.0
    if attacker.position == block_from_square:
        score -= 20.0   # A Blitz where the block is the starting square is unattractive
    if in_scoring_range(game, defender):
        score += 10.0  # Blitzing players closer to the endzone is attractive
    return score


def score_foul(game: Game, heat_map: FfHeatMap, attacker: Player, defender: Player, to_square: Square) -> float:
    score = GrodBot.BASE_SCORE_FOUL
    ball_carrier: Optional[Player] = game.get_ball_carrier()

    if ball_carrier == attacker:
        score = score - 30.0
    if attacker.has_skill(Skill.DIRTY_PLAYER):
        score = score + 10.0
    if attacker.has_skill(Skill.SNEAKY_GIT):
        score = score + 10.0
    if defender.state.stunned:
        score = score - 15.0

    assists_for, assists_against = game.num_assists_at(attacker, defender, to_square, foul=True)
    score = score + (assists_for-assists_against) * 15.0

    if attacker.team.state.bribes > 0:
        score += 40.0
    if attacker.has_skill(Skill.CHAINSAW):
        score += 30.0
    # TVdiff = defender.GetBaseTV() - attacker.GetBaseTV()
    tv_diff = 10.0
    score = score + tv_diff

    return score


def score_move(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool, str):

    scores: List[(float, bool, str)] = [
        [*score_receiving_position(game, heat_map, player, to_square), 'move to receiver'],
        [*score_move_towards_ball(game, heat_map, player, to_square), 'move toward ball'],
        [*score_move_to_ball(game, heat_map, player, to_square), 'move to ball'],
        [*score_move_ball(game, heat_map, player, to_square), 'move ball'],
        [*score_sweep(game, heat_map, player, to_square), 'move to sweep'],
        [*score_defensive_screen(game, heat_map, player, to_square), 'move to defensive screen'],
        [*score_offensive_screen(game, heat_map, player, to_square), 'move to offsensive screen'],
        [*score_caging(game, heat_map, player, to_square), 'move to cage'],
        [*score_mark_opponent(game, heat_map, player, to_square), 'move to mark opponent']
        ]

    scores.sort(key=lambda tup: tup[0], reverse=True)
    score, is_complete, description = scores[0]

    # All moves should avoid the sideline
    if distance_to_sideline(game, to_square) == 0:
        score += GrodBot.ADDITIONAL_SCORE_SIDELINE
    if distance_to_sideline(game, to_square) == 1:
        score += GrodBot.ADDITIONAL_SCORE_NEAR_SIDELINE

    return score, is_complete, description


def score_receiving_position(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):
    ball_carrier = game.get_ball_carrier()
    if ball_carrier is not None and (player.team != ball_carrier.team or player == game.get_ball_carrier()):
        return 0.0, True

    receivingness = player_receiver_ability(game, player)
    score = receivingness - 30.0
    if in_scoring_endzone(game, player.team, to_square):
        num_in_range = len(players_in_scoring_endzone(game, player.team, include_own=True, include_opp=False))
        if player.team.state.turn == 8:
            score += 40   # Pretty damned urgent to get to end zone!
        score -= num_in_range * num_in_range * 40  # Don't want too many catchers in the endzone ...

    score += 5.0 * (max(distance_to_scoring_endzone(game, player.team, player.position), player.get_ma()) - max(distance_to_scoring_endzone(game, player.team, to_square), player.get_ma()))
    # Above score doesn't push players to go closer than their MA from the endzone.

    if distance_to_scoring_endzone(game, player.team, to_square) > player.get_ma() + 2:
        score -= 30.0
    opp_team = game.get_opp_team(player.team)
    opps: List[Player] = game.get_adjacent_players(player.position, opp_team, stunned=False, down=False)
    if opps:
        score -= 40.0 + 20.0 * len(opps)
    score -= 10.0 * len(game.get_adjacent_players(to_square, opp_team, stunned=False, down=False))
    num_in_range = len(players_in_scoring_distance(game, player.team, include_own=True, include_opp=False))
    score -= num_in_range * num_in_range * 20.0     # Lower the score if we already have some receivers.
    if players_in(game, player.team, squares_within(game, to_square, 2), include_opp=False, include_own=True):
        score -= 20.0

    return score, True


def score_move_towards_ball(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):
    ball_square: Square = game.get_ball_position()
    ball_carrier = game.get_ball_carrier()
    if ball_carrier is not None:
        ball_team = ball_carrier.team
    else:
        ball_team = None

    if (to_square == ball_square) or ((ball_team is not None) and (ball_team == player.team)):
        return 0.0, True

    score = GrodBot.BASE_SCORE_MOVE_TOWARD_BALL
    if ball_carrier is None:
        score += 20.0

    player_distance_to_ball = ball_square.distance(player.position)
    destination_distance_to_ball = ball_square.distance(to_square)

    score += (player_distance_to_ball - destination_distance_to_ball)

    if destination_distance_to_ball > 3:
        pass
        # score -= 50

    # ma_allowed = player.move_allowed()

    # current_distance_to_ball = ball_square.distance(player.position)

    # Cancel the penalty for being near the sideline if the ball is on the sideline
    # if distance_to_sideline(game, ball_square) <= 1:
    #     if distance_to_sideline(game, to_square): score += 10.0

    # Increase score if moving closer to the ball
    # score += (current_distance_to_ball - distance_to_ball)*3

    return score, True


def score_move_to_ball(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):
    ball_square: Square = game.get_ball_position()
    ball_carrier = game.get_ball_carrier()
    if (ball_square != to_square) or (ball_carrier is not None):
        return 0.0, True

    score = GrodBot.BASE_SCORE_MOVE_TO_BALL
    if player.has_skill(Skill.SURE_HANDS) or not player.team.state.reroll_used:
        score += 15.0
    if player.get_ag() < 2:
        score += -10.0
    if player.get_ag() == 3:
        score += 5.0
    if player.get_ag() > 3:
        score += 10.0
    num_tz = game.num_tackle_zones_at(player, ball_square)
    score += - 10 * num_tz    # Lower score if lots of tackle zones on ball.

    # If there is only 1 or 3 players left to move, lets improve score of trying to pick the ball up
    players_to_move: List[Player] = get_players(game, player.team, include_own=True, include_opp=False, include_used=False, include_stunned=False)
    if len(players_to_move) == 1:
        score += 25
    if len(players_to_move) == 2:
        score += 15

    # If the current player is the best player to pick up the ball, increase the score
    players_sorted_blitz = sorted(players_to_move, key=lambda x: player_blitz_ability(game, x), reverse=True)
    if players_sorted_blitz[0] == player:
        score += 9

    # Cancel the penalty for being near the sideline if the ball is on/near the sideline (it's applied later)
    if distance_to_sideline(game, ball_square) == 1:
        score -= GrodBot.ADDITIONAL_SCORE_NEAR_SIDELINE
    if distance_to_sideline(game, ball_square) == 0:
        score -= GrodBot.ADDITIONAL_SCORE_SIDELINE

    # Need to increase score if no other player is around to get the ball (to do)

    return score, False


def score_move_ball(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):
    # ball_square: Square = game.get_ball_position()
    ball_carrier = game.get_ball_carrier()
    if (ball_carrier is None) or player != ball_carrier:
        return 0.0, True

    score = GrodBot.BASE_SCORE_MOVE_BALL
    if in_scoring_endzone(game, player.team, to_square):
        if player.team.state.turn == 8:
            score += 115.0  # Make overwhelmingly attractive
        else:
            score += 60.0  # Make scoring attractive
    elif player.team.state.turn == 8:
        score -= 100.0  # If it's the last turn, heavily penalyse a non-scoring action
    else:
        score += heat_map.get_ball_move_square_safety_score(to_square)
        opps: List[Player] = game.get_adjacent_players(to_square, team=game.get_opp_team(player.team), stunned=False)
        if opps:
            score -= (40.0 + 20.0 * len(opps))
        opps_close_to_destination = players_in(game, player.team, squares_within(game, to_square, 2), include_own=False, include_opp=True, include_stunned=False)
        if opps_close_to_destination:
            score -= (20.0 + 5.0 * len(opps_close_to_destination))
        if not blitz_used(game):
            score -= 30.0  # Lets avoid moving the ball until the Blitz has been used (often helps to free the move).

        dist_player = distance_to_scoring_endzone(game, player.team, player.position)
        dist_destination = distance_to_scoring_endzone(game, player.team, to_square)
        score += 5.0 * (dist_player - dist_destination)  # Increase score the closer we get to the scoring end zone

        # Try to keep the ball central
        if distance_to_sideline(game, to_square) < 3:
            score -= 30

    return score, True


def score_sweep(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):
    ball_carrier = game.get_ball_carrier()
    if ball_carrier is not None:
        ball_team = ball_carrier.team
    else:
        ball_team = None
    if ball_team == player.team:
        return 0.0, True  # Don't sweep unless the other team has the ball
    if distance_to_defending_endzone(game, player.team, game.get_ball_position()) < 9:
        return 0.0, True  # Don't sweep when the ball is close to the endzone
    if players_in_scoring_distance(game, player.team, include_own=False, include_opp=True):
        return 0.0, True  # Don't sweep when there are opponent units in scoring range

    score = GrodBot.BASE_SCORE_MOVE_TO_SWEEP
    blitziness = player_blitz_ability(game, player)
    score += blitziness - 60.0
    score -= 30.0 * len(game.get_adjacent_opponents(player, standing=True, down=False, stunned=False))

    # Now to evaluate ideal square for Sweeping:

    x_preferred = int(reverse_x_for_left(game, player.team, (game.state.pitch.width-2) / 4))
    y_preferred = int((game.state.pitch.height-2) / 2)
    score -= abs(y_preferred - to_square .y) * 10.0

    # subtract 5 points for every square away from the preferred sweep location.
    score -= abs(x_preferred - to_square .x) * 5.0

    # Check if a player is already sweeping:
    for i in range(-2, 3):
        for j in range(-2, 3):
            cur: Square = game.get_square(x_preferred + i, y_preferred + j)
            player: Optional[Player] = game.get_player_at(cur)
            if player is not None and player.team == player.team:
                score -= 90.0

    return score, True


def score_defensive_screen(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):
    ball_square = game.get_ball_position()
    ball_carrier = game.get_ball_carrier()
    if ball_carrier is not None:
        ball_team = ball_carrier.team
    else:
        ball_team = None

    if ball_team is None or ball_team == player.team:
        return 0.0, True  # Don't screen if we have the ball or ball is on the ground

    # This one is a bit trickier by nature, because it involves combinations of two or more players...
    #    Increase score if square is close to ball carrier.
    #    Decrease if far away.
    #    Decrease if square is behind ball carrier.
    #    Increase slightly if square is 1 away from sideline.
    #    Decrease if close to a player on the same team WHO IS ALREADY screening.
    #    Increase slightly if most of the players movement must be used to arrive at the screening square.

    score = GrodBot.BASE_SCORE_DEFENSIVE_SCREEN

    distance_ball_carrier_to_end = distance_to_defending_endzone(game, player.team, ball_square)
    distance_square_to_end = distance_to_defending_endzone(game, player.team, to_square)

    if distance_square_to_end + 1.0 < distance_ball_carrier_to_end:
        score += 30.0  # Increase score defending on correct side of field.

    distance_to_ball = ball_square.distance(to_square)
    score += 4.0*max(5.0 - distance_to_ball, 0.0)  # Increase score defending in front of ball carrier
    score += distance_square_to_end/10.0  # Increase score a small amount to screen closer to opponents.
    distance_to_closest_opponent = distance_to_nearest_player(game, player.team, to_square, include_own=False, include_opp=True, include_stunned=False)
    if distance_to_closest_opponent <= 1.5:
        score -= 30.0
    elif distance_to_closest_opponent <= 2.95:
        score += 10.0
    elif distance_to_closest_opponent > 2.95:
        score += 5.0
    if distance_to_sideline(game, to_square) == 1:
        score -= GrodBot.ADDITIONAL_SCORE_NEAR_SIDELINE  # Cancel the negative score of being 1 from sideline.

    distance_to_closest_friendly_used = distance_to_nearest_player(game, player.team, to_square, include_own=True, include_opp=False, only_used=True)
    if distance_to_closest_friendly_used >= 4:
        score += 2.0
    elif distance_to_closest_friendly_used >= 3:
        score += 40.0  # Increase score if the square links with another friendly (hopefully also screening)
    elif distance_to_closest_friendly_used > 2:
        score += 10.0   # Descrease score if very close to another defender
    else:
        score -= 10.0  # Decrease score if too close to another defender.

    distance_to_closest_friendly_unused = distance_to_nearest_player(game, player.team, to_square, include_own=True, include_opp=False, include_used=True)
    if distance_to_closest_friendly_unused >= 4:
        score += 3.0
    elif distance_to_closest_friendly_unused >= 3:
        score += 8.0  # Increase score if the square links with another friendly (hopefully also screening)
    elif distance_to_closest_friendly_unused > 2:
        score += 3.0  # Descrease score if very close to another defender
    else:
        score -= 10.0  # Decrease score if too close to another defender.

    return score, True


def score_offensive_screen(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):

    # Another subtle one.  Basically if the ball carrier "breaks out", I want to screen him from
    # behind, rather than cage him.  I may even want to do this with an important receiver.
    #     Want my players to be 3 squares from each other, not counting direct diagonals.
    #     Want my players to be hampering the movement of opponent ball or players.
    #     Want my players in a line between goal line and opponent.
    #

    ball_carrier: Player = game.get_ball_carrier()
    ball_square: Player = game.get_ball_position()
    if ball_carrier is None or ball_carrier.team != player.team:
        return 0.0, True

    score = 0.0     # Placeholder - not implemented yet.

    return score, True


def score_caging(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):
    ball_carrier: Player = game.get_ball_carrier()
    if ball_carrier is None or ball_carrier.team != player.team or ball_carrier == player:
        return 0.0, True          # Noone has the ball.  Don't try to cage.
    ball_square: Square = game.get_ball_position()

    cage_square_groups: List[List[Square]] = [
        caging_squares_north_east(game, ball_square),
        caging_squares_north_west(game, ball_square),
        caging_squares_south_east(game, ball_square),
        caging_squares_south_west(game, ball_square)
        ]

    dist_opp_to_ball = distance_to_nearest_player(game, player.team, ball_square, include_own=False, include_opp=True, include_stunned=False)
    avg_opp_ma = average_ma(game, get_players(game, player.team, include_own=False, include_opp=True, include_stunned=False))

    for curGroup in cage_square_groups:
        if to_square in curGroup and not players_in(game, player.team, curGroup, include_opp=False, include_own=True, only_blockable=True):
            # Test square is inside the cage corner and no player occupies the corner
            if to_square in curGroup:
                score = GrodBot.BASE_SCORE_CAGE_BALL
            dist = distance_to_nearest_player(game, player.team, to_square, include_own=False, include_stunned=False, include_opp=True)
            score += dist_opp_to_ball - dist
            if dist_opp_to_ball > avg_opp_ma:
                score -= 30.0
            if not ball_carrier.state.used:
                score -= 30.0
            if to_square.is_adjacent(game.get_ball_position()):
                score += 5
            if is_bishop_position_of(game, player, ball_carrier):
                score -= 2
            score += heat_map.get_cage_necessity_score(to_square)
            if not ball_carrier.state.used:
                score = max(0.0, score - GrodBot.BASE_SCORE_CAGE_BALL)  # Penalise forming a cage if ball carrier has yet to move
            if not player.state.up:
                score += 5.0
            return score, True

    return 0, True


def score_mark_opponent(game: Game, heat_map: FfHeatMap, player: Player, to_square: Square) -> (float, bool):

    # Modification - no need to mark prone opponents already marked
    ball_carrier = game.get_ball_carrier()
    opp_team = game.get_opp_team(player.team)
    if ball_carrier is not None:
        ball_team = ball_carrier.team
    else:
        ball_team = None
    ball_square = game.get_ball_position()

    if ball_square == player.position:
        return 0.0, True  # Don't mark opponents deliberately with the ball
    all_opponents: List[Player] = game.get_adjacent_players(to_square, team=opp_team)
    if not all_opponents:
        return 0.0, True

    if (ball_carrier is not None) and (ball_carrier == player):
        return 0.0, True

    score = GrodBot.BASE_SCORE_MOVE_TO_OPPONENT
    if to_square.is_adjacent(game.get_ball_position()):
        if ball_team == player.team:
            score += 20.0
        else:
            score += 30.0

    for opp in all_opponents:
        if distance_to_scoring_endzone(game, opp.team, to_square) < opp.get_ma() + 2:
            score += 10.0  # Mark opponents in scoring range first.
            break         # Only add score once.

    if len(all_opponents) == 1:
        score += 20.0
        num_friendly_next_to = game.num_tackle_zones_in(all_opponents[0])
        if all_opponents[0].state.up:
            if num_friendly_next_to == 1:
                score += 5.0
            else:
                score -= 10.0 * num_friendly_next_to

        if not all_opponents[0].state.up:
            if num_friendly_next_to == 0:
                score += 5.0
            else:
                score -= 10.0 * num_friendly_next_to  # Unless we want to start fouling ...

    if not player.state.up:
        score += 25.0
    if not player.has_skill(Skill.GUARD):
        score -= len(all_opponents) * 10.0
    else:
        score += len(all_opponents) * 10.0

    ball_is_near = False
    for current_opponent in all_opponents:
        if current_opponent.position.is_adjacent(game.get_ball_position()):
            ball_is_near = True

    if ball_is_near:
        score += 8.0
    if player.position != to_square and game.num_tackle_zones_in(player) > 0:
        score -= 40.0

    if ball_square is not None:
        distance_to_ball = ball_square.distance(to_square)
        score -= distance_to_ball / 5.0   # Mark opponents closer to ball when possible

    if ball_team is not None and ball_team != player.team:
        distance_to_other_endzone = distance_to_scoring_endzone(game, player.team, to_square)
        # This way there is a preference for most advanced (distance wise) units.
    return score, True


def score_handoff(game: Game, heat_map: FfHeatMap, ball_carrier: Player, receiver: Player, from_square: Square) -> float:
    if receiver == ball_carrier:
        return 0.0

    score = GrodBot.BASE_SCORE_HANDOFF
    score += probability_fail_to_score(probability_catch_fail(game, receiver))
    if not ball_carrier.team.state.reroll_used:
        score += +10.0
    score -= 5.0 * (distance_to_scoring_endzone(game, ball_carrier.team, receiver.position) - distance_to_scoring_endzone(game, ball_carrier.team, ball_carrier.position))
    if receiver.state.used:
        score -= 30.0
    if (game.num_tackle_zones_in(ball_carrier) > 0 or game.num_tackle_zones_in(receiver) > 0) and not blitz_used(game):
        score -= 50.0  # Don't try a risky hand-off if we haven't blitzed yet
    if in_scoring_range(game, receiver) and not in_scoring_range(game, ball_carrier):
        score += 40.0
    # score += heat_map.get_ball_move_square_safety_score(receiver.position)
    return score


def score_pass(game: Game, heat_map: FfHeatMap, passer: Player, from_square: Square, to_square: Square) -> float:

    receiver = game.get_player_at(to_square)

    if receiver is None:
        return 0.0
    if receiver.team != passer.team:
        return 0.0
    if receiver == passer:
        return 0.0

    score = GrodBot.BASE_SCORE_PASS
    score += probability_fail_to_score(probability_catch_fail(game, receiver))
    dist: PassDistance = game.get_pass_distance(from_square, receiver.position)
    score += probability_fail_to_score(probability_pass_fail(game, passer, from_square, dist))
    if not passer.team.state.reroll_used:
        score += +10.0
    score = score - 5.0 * (distance_to_scoring_endzone(game, receiver.team, receiver.position) - distance_to_scoring_endzone(game, passer.team, passer.position))
    if receiver.state.used:
        score -= 30.0
    if game.num_tackle_zones_in(passer) > 0 or game.num_tackle_zones_in(receiver) > 0 and not blitz_used(game):
        score -= 50.0
    if in_scoring_range(game, receiver) and not in_scoring_range(game, passer):
        score += 40.0
    return score


def score_block(game: Game, heat_map: FfHeatMap, attacker: Player, defender: Player) -> float:
    score = GrodBot.BASE_SCORE_BLOCK
    ball_carrier = game.get_ball_carrier()
    ball_square = game.get_ball_position()
    if attacker.has_skill(Skill.CHAINSAW):
        score += 15.0
        score += 20.0 - 2 * defender.get_av()
        # Add something in case the defender is really valuable?
    else:
        num_block_dice = game.num_block_dice(attacker, defender)
        if num_block_dice == 3:
            score += 15.0
        if num_block_dice == 2:
            score += 0.0
        if num_block_dice == 1:
            score += -66.0  # score is close to zero.
        if num_block_dice == -2:
            score += -95.0
        if num_block_dice == -3:
            score += -150.0

        if not attacker.team.state.reroll_used and not attacker.has_skill(Skill.LONER):
            score += 10.0
        if attacker.has_skill(Skill.BLOCK) or attacker.has_skill(Skill.WRESTLE):
            score += 20.0
        if defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE):
            score += -10.0
        if defender.has_skill(Skill.BLOCK):
            score += -10.0
        if attacker_would_surf(game, attacker, defender):
            score += 32.0
        if attacker.has_skill(Skill.LONER):
            score -= 10.0

    if attacker == ball_carrier:
        score += -45.0
    if defender == ball_carrier:
        score += 35.0
    if defender.position.is_adjacent(ball_square):
        score += 15.0

    return score


def score_push(game: Game, from_square: Square, to_square: Square) -> float:
    score = 0.0
    ball_square = game.get_ball_position()
    if distance_to_sideline(game, to_square) == 0:
        score = score + 10.0    # Push towards sideline
    if ball_square is not None and to_square .is_adjacent(ball_square):
        score = score - 15.0    # Push away from ball
    if direct_surf_squares(game, from_square, to_square):
        score = score + 10.0
    return score


def check_follow_up(game: Game) -> bool:
    # To do: the  logic here is faulty for the current game state,  in terms of how and when actions are evaluated.  I.e.
    # the check appears to happen before the defending player is placed prone (but after the player is pushed?)
    # What I want is to follow up, generally, if the defender is prone and not otherwise.
    active_player: Player = game.state.active_player

    block_proc = last_block_proc(game)

    attacker: Player = block_proc.attacker
    defender: Player = block_proc.defender
    is_blitz_action = block_proc.blitz
    for position in game.state.available_actions[0].positions:
        if active_player.position != position:
            follow_up_square: Square = position

    defender_prone = (block_proc.selected_die == BBDieResult.DEFENDER_DOWN) or ((block_proc.selected_die == BBDieResult.DEFENDER_STUMBLES) and (attacker.has_skill(Skill.TACKLE) or not defender.has_skill(Skill.DODGE)))

    num_tz_cur = game.num_tackle_zones_in(active_player)
    num_tz_new = game.num_tackle_zones_at(active_player, follow_up_square)
    opp_adj_cur = game.get_adjacent_opponents(active_player, stunned=False, down=False)
    opp_adj_new = game.get_adjacent_players(follow_up_square, team=game.get_opp_team(active_player.team), stunned=False, down=False)

    num_tz_new -= defender_prone

    # If blitzing (with squares of movement left) always follow up if the new square is not in any tackle zone.
    if is_blitz_action and attacker.num_moves_left() > 0 and num_tz_new == 0:
        return True

    # If Attacker has the ball, strictly follow up only if there are less opponents next to new square.
    if game.get_ball_carrier == attacker:
        if len(opp_adj_new) - defender_prone < len(opp_adj_cur):
            return True
        return False

    if game.get_ball_carrier == defender:
        return True   # Always follow up if defender has ball
    if distance_to_sideline(game, follow_up_square) == 0:
        return False    # No if moving to sideline
    if distance_to_sideline(game, defender.position) == 0:
        return True  # Follow up if opponent is on sideline
    if follow_up_square.is_adjacent(game.get_ball_position()):
        return True  # Follow if moving next to ball
    if attacker.position.is_adjacent(game.get_ball_position()):
        return False  # Don't follow if already next to ball

    # Follow up if less standing opponents in the next square or equivalent, but defender is now prone
    if (num_tz_new == 0) or (num_tz_new < num_tz_cur) or (num_tz_new == num_tz_cur and not defender_prone):
        return True
    if attacker.has_skill(Skill.GUARD) and num_tz_new > num_tz_cur:
        return True      # Yes if attacker has guard
    if attacker.get_st() > defender.get_st() + num_tz_new - num_tz_cur:
        return True  # Follow if stronger
    if is_blitz_action and attacker.num_moves_left() == 0:
        return True  # If blitzing but out of moves, follow up to prevent GFIing...

    return False


def check_reroll_block(game: Game, team: Team, block_results: List[ActionSequence], favor: Team) -> bool:
    block_proc: Optional[Block] = last_block_proc(game)
    attacker: Player = block_proc.attacker
    defender: Player = block_proc.defender
    is_blitz_action = block_proc.blitz
    ball_carrier: Optional[Player] = game.get_ball_carrier()

    best_block_score: float = 0
    cur_block_score: float = -1

    if len(block_results) > 0:
        best_block_score = block_results[0].score

    if len(block_results) > 1:
        cur_block_score = block_results[1].score
        if favor == team and cur_block_score > best_block_score:
            best_block_score = cur_block_score
        if favor != team and cur_block_score < best_block_score:
            best_block_score = cur_block_score

    if len(block_results) > 2:
        cur_block_score = block_results[2].score
        if favor == team and cur_block_score > best_block_score:
            best_block_score = cur_block_score
        if favor != team and cur_block_score < best_block_score:
            best_block_score = cur_block_score

    if best_block_score < 4:
        return True
    elif ball_carrier == defender and best_block_score < 5:
        return True  # Reroll if target has ball and not knocked over.
    else:
        return False


def scoring_urgency_score(game: Game, heat_map: FfHeatMap, player: Player) -> float:
    if player.team.state.turn == 8:
        return 40
    return 0


def path_cost_to_score(path: Path) -> float:
    cost: float = 1-path.prob

    # assert 0 <= cost <= 1

    score = -(cost * cost * (250.0 + GrodBot.ADDITIONAL_SCORE_DODGE))
    return score


def probability_fail_to_score(probability: float) -> float:
    score = -(probability * probability * (250.0 + GrodBot.ADDITIONAL_SCORE_DODGE))
    return score


def probability_catch_fail(game: Game, receiver: Player) -> float:
    num_tz = 0.0
    if not receiver.has_skill(Skill.NERVES_OF_STEEL):
        num_tz = game.num_tackle_zones_in(receiver)
    probability_success = min(5.0, receiver.get_ag()+1.0-num_tz)/6.0
    if receiver.has_skill(Skill.CATCH):
        probability_success += (1.0-probability_success)*probability_success
    probability = 1.0 - probability_success
    return probability


def probability_pass_fail(game: Game, passer: Player, from_square: Square, dist: PassDistance) -> float:
    num_tz = 0.0
    if not passer.has_skill(Skill.NERVES_OF_STEEL):
        num_tz = game.num_tackle_zones_at(passer, from_square)
    if passer.has_skill(Skill.ACCURATE):
        num_tz -= 1
    if passer.has_skill(Skill.STRONG_ARM and dist != PassDistance.QUICK_PASS):
        num_tz -= 1
    if dist == PassDistance.HAIL_MARY:
        return -100.0
    if dist == PassDistance.QUICK_PASS:
        num_tz -= 1
    if dist == PassDistance.SHORT_PASS:
        num_tz -= 0
    if dist == PassDistance.LONG_PASS:
        num_tz += 1
    if dist == PassDistance.LONG_BOMB:
        num_tz += 2
    probability_success = min(5.0, passer.get_ag()-num_tz)/6.0
    if passer.has_skill(Skill.PASS):
        probability_success += (1.0-probability_success)*probability_success
    probability = 1.0 - probability_success
    return probability


def choose_gaze_victim(game: Game, player: Player) -> Player:
    best_victim: Optional[Player] = None
    best_score = 0.0
    ball_square: Square = game.get_ball_position()
    potentials: List[Player] = game.get_adjacent_players(player, team=game.get_opp_team(player.team), down=False, standing=True, stunned=False)
    for unit in potentials:
        current_score = 5.0
        current_score += 6.0 - unit.get_ag()
        if unit.position.is_adjacent(ball_square):
            current_score += 5.0
        if current_score > best_score:
            best_score = current_score
            best_victim = unit
    return best_victim


def average_st(game: Game, players: List[Player]) -> float:
    values = [player.get_st() for player in players]
    return sum(values)*1.0 / len(values)


def average_av(game: Game, players: List[Player]) -> float:
    values = [player.get_av() for player in players]
    return sum(values)*1.0 / len(values)


def average_ma(game: Game, players: List[Player]) -> float:
    values = [player.get_ma() for player in players]
    return sum(values)*1.0 / len(values)


def player_bash_ability(game: Game, player: Player) -> float:
    bashiness: float = 0.0
    bashiness += 10.0 * player.get_st()
    bashiness += 5.0 * player.get_av()
    if player.has_skill(Skill.BLOCK):
        bashiness += 10.0
    if player.has_skill(Skill.WRESTLE):
        bashiness += 10.0
    if player.has_skill(Skill.MIGHTY_BLOW):
        bashiness += 5.0
    if player.has_skill(Skill.CLAWS):
        bashiness += 5.0
    if player.has_skill(Skill.PILING_ON):
        bashiness += 5.0
    if player.has_skill(Skill.GUARD):
        bashiness += 15.0
    if player.has_skill(Skill.DAUNTLESS):
        bashiness += 10.0
    if player.has_skill(Skill.FOUL_APPEARANCE):
        bashiness += 5.0
    if player.has_skill(Skill.TENTACLES):
        bashiness += 5.0
    if player.has_skill(Skill.STUNTY):
        bashiness -= 10.0
    if player.has_skill(Skill.REGENERATION):
        bashiness += 10.0
    if player.has_skill(Skill.THICK_SKULL):
        bashiness += 3.0
    return bashiness


def team_bash_ability(game: Game, players: List[Player]) -> float:
    total = 0.0
    for player in players:
        total += player_bash_ability(game, player)
    return total


def player_pass_ability(game: Game, player: Player) -> float:
    passing_ability = 0.0
    passing_ability += player.get_ag() * 15.0    # Agility most important.
    passing_ability += player.get_ma() * 2.0     # Fast movements make better ball throwers.
    if player.has_skill(Skill.PASS):
        passing_ability += 10.0
    if player.has_skill(Skill.SURE_HANDS):
        passing_ability += 5.0
    if player.has_skill(Skill.EXTRA_ARMS):
        passing_ability += 3.0
    if player.has_skill(Skill.NERVES_OF_STEEL):
        passing_ability += 3.0
    if player.has_skill(Skill.ACCURATE):
        passing_ability += 5.0
    if player.has_skill(Skill.STRONG_ARM):
        passing_ability += 5.0
    if player.has_skill(Skill.BONE_HEAD):
        passing_ability -= 15.0
    if player.has_skill(Skill.REALLY_STUPID):
        passing_ability -= 15.0
    if player.has_skill(Skill.WILD_ANIMAL):
        passing_ability -= 15.0
    if player.has_skill(Skill.ANIMOSITY):
        passing_ability -= 10.0
    if player.has_skill(Skill.LONER):
        passing_ability -= 15.0
    if player.has_skill(Skill.DUMP_OFF):
        passing_ability += 5.0
    if player.has_skill(Skill.SAFE_THROW):
        passing_ability += 5.0
    if player.has_skill(Skill.NO_HANDS):
        passing_ability -= 100.0
    return passing_ability


def player_blitz_ability(game: Game, player: Player) -> float:
    blitzing_ability = player_bash_ability(game, player)
    blitzing_ability += player.get_ma() * 10.0
    if player.has_skill(Skill.TACKLE):
        blitzing_ability += 5.0
    if player.has_skill(Skill.SPRINT):
        blitzing_ability += 5.0
    if player.has_skill(Skill.SURE_FEET):
        blitzing_ability += 5.0
    if player.has_skill(Skill.STRIP_BALL):
        blitzing_ability += 5.0
    if player.has_skill(Skill.DIVING_TACKLE):
        blitzing_ability += 5.0
    if player.has_skill(Skill.MIGHTY_BLOW):
        blitzing_ability += 5.0
    if player.has_skill(Skill.CLAWS):
        blitzing_ability += 5.0
    if player.has_skill(Skill.PILING_ON):
        blitzing_ability += 5.0
    if player.has_skill(Skill.BONE_HEAD):
        blitzing_ability -= 15.0
    if player.has_skill(Skill.REALLY_STUPID):
        blitzing_ability -= 15.0
    if player.has_skill(Skill.WILD_ANIMAL):
        blitzing_ability -= 10.0
    if player.has_skill(Skill.LONER):
        blitzing_ability -= 15.0
    if player.has_skill(Skill.SIDE_STEP):
        blitzing_ability += 5.0
    if player.has_skill(Skill.JUMP_UP):
        blitzing_ability += 5.0
    if player.has_skill(Skill.HORNS):
        blitzing_ability += 10.0
    if player.has_skill(Skill.JUGGERNAUT):
        blitzing_ability += 10.0
    if player.has_skill(Skill.LEAP):
        blitzing_ability += 5.0
    return blitzing_ability


def player_receiver_ability(game: Game, player: Player) -> float:
    receiving_ability = 0.0
    receiving_ability += player.get_ma() * 5.0
    receiving_ability += player.get_ag() * 10.0
    if player.has_skill(Skill.CATCH):
        receiving_ability += 15.0
    if player.has_skill(Skill.EXTRA_ARMS):
        receiving_ability += 10.0
    if player.has_skill(Skill.NERVES_OF_STEEL):
        receiving_ability += 5.0
    if player.has_skill(Skill.DIVING_CATCH):
        receiving_ability += 5.0
    if player.has_skill(Skill.DODGE):
        receiving_ability += 10.0
    if player.has_skill(Skill.SIDE_STEP):
        receiving_ability += 5.0
    if player.has_skill(Skill.BONE_HEAD):
        receiving_ability -= 15.0
    if player.has_skill(Skill.REALLY_STUPID):
        receiving_ability -= 15.0
    if player.has_skill(Skill.WILD_ANIMAL):
        receiving_ability -= 15.0
    if player.has_skill(Skill.LONER):
        receiving_ability -= 15.0
    if player.has_skill(Skill.NO_HANDS):
        receiving_ability -= 100.0
    return receiving_ability


def player_run_ability(game: Game, player: Player) -> float:
    running_ability = 0.0
    running_ability += player.get_ma() * 10.0    # Really favour fast units
    running_ability += player.get_ag() * 10.0    # Agility to be prized
    running_ability += player.get_st() * 5.0     # Doesn't hurt to be strong!
    if player.has_skill(Skill.SURE_HANDS):
        running_ability += 10.0
    if player.has_skill(Skill.BLOCK):
        running_ability += 10.0
    if player.has_skill(Skill.EXTRA_ARMS):
        running_ability += 5.0
    if player.has_skill(Skill.DODGE):
        running_ability += 10.0
    if player.has_skill(Skill.SIDE_STEP):
        running_ability += 5.0
    if player.has_skill(Skill.STAND_FIRM):
        running_ability += 3.0
    if player.has_skill(Skill.BONE_HEAD):
        running_ability -= 15.0
    if player.has_skill(Skill.REALLY_STUPID):
        running_ability -= 15.0
    if player.has_skill(Skill.WILD_ANIMAL):
        running_ability -= 15.0
    if player.has_skill(Skill.LONER):
        running_ability -= 15.0
    if player.has_skill(Skill.ANIMOSITY):
        running_ability -= 5.0
    if player.has_skill(Skill.DUMP_OFF):
        running_ability += 5.0
    if player.has_skill(Skill.NO_HANDS):
        running_ability -= 100.0
    return running_ability


def player_value(game: Game, player: Player) -> float:
    value = player.get_ag()*40 + player.get_av()*30 + player.get_ma()*30 + player.get_st()*50 + len(player.get_skills())*20
    return value


# Register bot
register_bot('GrodBot', GrodBot)

