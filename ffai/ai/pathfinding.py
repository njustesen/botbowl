"""
==========================
Author: Kevin Glass / Peter Moore
Year: 2019
==========================
This module contains a generic A* Path Finder, along with a specific test implementation.  Originally written in Java by
Kevin Glass: http://www.cokeandcode.com/main/tutorials/path-finding/ and converted to Python, with modifications, by
Peter Moore.  The main modifications,
    1. Create_paths, which finds solutions to all nodes within search_distance
    3. Support for adding costs as if they are probabilities via p_s = 1-(1-p1)*(1-p2)
    3. Simple class implementations as well as run code that demonstrates the results via main()
"""
from typing import Optional, List
from ffai.core.model import Player, Square
from ffai.core.table import Skill, WeatherType, Tile
from ffai.core.game import Game
import time
import copy


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
        self.cost: float = 0
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

    def value(self: 'Node'):
        return 1 - self.cost


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

    def blocked(self, mover: FFMover, x: int, y: int) -> bool:
        square = self.game.get_square(x, y)

        # Need to ignore the "crowd" squares on the boundary by blocking them.
        return (x <= 0) or (y <= 0) or (x >= self.width - 1) or (y >= self.height - 1) or self.game.get_player_at(
            square) is not None

    def get_cost(self, mover: FFMover, sx: int, sy: int, tx: int, ty: int) -> float:
        square_from = self.game.get_square(sx, sy)
        square_to = self.game.get_square(tx, ty)
        moving_unit = mover.player
        dodge_prob = self.game.get_dodge_prob_from(moving_unit, square_from, square_to,
                                                   allow_dodge_reroll=mover.allow_skill_reroll)
        move_prob = 1.0
        cur_depth: int = mover.cur_depth  # essentially number of moves already done.
        if cur_depth != -1 and (cur_depth + 1 > moving_unit.num_moves_left(include_gfi=False)):
            move_prob = 1.0 / 6.0
            if self.game.state.weather == WeatherType.BLIZZARD:
                move_prob = 2.0 / 6.0
            if moving_unit.has_skill(Skill.SURE_FEET):
                move_prob += (1 - move_prob) * move_prob
        cost = 1 - move_prob * dodge_prob
        return cost

    def get_movement(self, mover: FFMover, sx: int, sy: int, tx: int, ty: int) -> float:
        return self.game.get_square(sx, sy).distance(self.game.get_square(tx, ty))


class FFPathFinder:

    def __init__(self, tile_map: FFTileMap, max_search_distance: int):
        self.open: SortedList = SortedList(lambda x: x.value())
        self.closed: List[Node] = []
        self.tile_map: FFTileMap = tile_map
        self.max_search_distance: int = max_search_distance
        self.nodes: List[List[Node]] = []
        for x in range(tile_map.get_width_in_tiles()):
            nodes_cur: List[Node] = []
            for y in range(tile_map.get_height_in_tiles()):
                nodes_cur.append(Node(x, y))
            self.nodes.append(nodes_cur)

    def find_path(self, mover: FFMover, sx: int, sy: int, tx: int = None, ty: int = None, tile: Tile = None, player: Player = None) -> Optional[Path]:
        # easy first check, if the destination is blocked, we can't get there
        if tx is not None and ty is not None and self.tile_map.blocked(mover, tx, ty):
            return None

        if tx == sx and ty == sy:
            return Path([], 1.0)

        # initial state for A*. The closed group is empty. Only the starting

        # tile is in the open list and it'e're already there
        self.nodes[sx][sy].cost = 0
        self.nodes[sx][sy].moves = 0 if mover.player.state.up or mover.player.has_skill(Skill.JUMP_UP) else 3
        self.nodes[sx][sy].depth = 0
        self.closed.clear()
        self.open.clear()
        self.open.append(self.nodes[sx][sy])

        if tx is not None and ty is not None:
            self.nodes[tx][ty].parent = None

        # Make a set of goals found
        goals = set()

        # while we haven'n't exceeded our max search depth
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
                        next_step_cost = 1.0 - ((1.0 - current.cost) * (1.0 - self.get_cost(mover, current.x, current.y, xp, yp)))
                        next_step_moves = current.moves + self.get_moves(mover, current.x, current.y, xp, yp)
                        neighbour = self.nodes[xp][yp]
                        self.tile_map.path_finder_visited(xp, yp)

                        # if the new cost we've determined for this node is lower than
                        # it has been previously makes sure the node hasn'e've
                        # determined that there might have been a better path to get to
                        # this node so it needs to be re-evaluated

                        if next_step_cost < neighbour.cost:
                            if self.in_open_list(neighbour):
                                self.remove_from_open(neighbour)

                            if self.in_closed_list(neighbour):
                                self.remove_from_closed(neighbour)

                        # if the node hasn't already been processed and discarded then
                        # reset it's cost to our current cost and add it as a next possible
                        # step (i.e. to the open list)
                        if not self.in_open_list(neighbour) and not (self.in_closed_list(neighbour)):
                            neighbour.cost = next_step_cost
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
        target_cost: float = target.cost
        if target.parent is None:
            return None
        path_steps: List[Square] = []
        while target != self.nodes[sx][sy]:
            path_steps.insert(0, self.tile_map.game.get_square(target.x, target.y))
            target = target.parent
        return Path(path_steps, 1-target_cost)

    def find_paths(self, mover, sx: int, sy: int) -> List[Path]:
        """
        Find all paths up to self.max_search_distance starting from (sx, sy).
        :return: 3-D List of either Paths (where a path to the node exists) or None, where no Path exists
        """
        t0 = time.time()
        self.nodes[sx][sy].cost = 0
        self.nodes[sx][sy].depth = 0
        self.nodes[sx][sy].moves = 0
        self.closed.clear()
        self.open.clear()
        self.open.append(self.nodes[sx][sy])      # Start with starting node.

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
                        next_step_cost = 1.0 - (1.0 - current.cost) * (1.0 - self.get_cost(mover, current.x, current.y, xp, yp))
                        next_step_moves = current.moves + self.get_moves(mover, current.x, current.y, xp, yp)
                        neighbour = self.nodes[xp][yp]
                        self.tile_map.path_finder_visited(xp, yp)

                        if next_step_cost < neighbour.cost:
                            if self.in_open_list(neighbour):
                                self.remove_from_open(neighbour)

                            if self.in_closed_list(neighbour):
                                self.remove_from_closed(neighbour)
                        if (not self.in_open_list(neighbour)) and (not self.in_closed_list(neighbour)):
                            neighbour.cost = next_step_cost
                            neighbour.moves = next_step_moves
                            neighbour.parent = current
                            self.add_to_open(neighbour)

        paths = self.create_paths(sx, sy)
        return paths

    def create_paths(self, sx: int, sy: int) -> List[Path]:
        paths = []
        for x in range(self.tile_map.get_width_in_tiles()):
            for y in range(self.tile_map.get_height_in_tiles()):
                if self.tile_map.has_visited(x,y):
                    node = self.nodes[x][y]
                    path_cur = self.create_path(sx, sy, x, y)
                    if path_cur is not None:
                        paths.append(path_cur)
        #l = [len(path) for path in paths]
        return paths

    def get_computed_cost(self, ix: int, iy: int) -> float:
        return self.nodes[ix][iy].cost

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
        valid = 0 <= x < self.tile_map.get_width_in_tiles() and 0 <= y < self.tile_map.get_height_in_tiles()
        valid = valid and not (sx == x and sy == y)
        valid = valid and not self.tile_map.blocked(mover, x, y)
        return valid

    def get_cost(self, mover: FFMover, sx: int, sy: int, tx: int, ty: int) -> float:
        return self.tile_map.get_cost(mover, sx, sy, tx, ty)

    def get_moves(self, mover: FFMover, sx: int, sy: int, tx: int, ty: int) -> float:
        return self.tile_map.get_movement(mover, sx, sy, tx, ty)


def _alter_state(game, player, from_position, moves_used):
    orig_player, orig_ball = None, None
    if from_position is not None or moves_used is not None:
        orig_player = copy.deepcopy(player)
        orig_ball = copy.deepcopy(game.get_ball())
    # Moove player if another starting position is used
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


def get_safest_path(game, player, position, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=False):
    """
    :param game:
    :param player: the player to move
    :param position: the location to move to
    :param allow_skill_reroll: whether to allow skill re-rolls in the probability computations. If enabled, skill re-rolls can
    currently be used at every step regardless of whether it was used before.
    :param max_search_distance: the maximum search distance. If None, it will use the player's number of moves left.
    :return a path containing the list of squares that forms the safest (and thereafter shortest) path for the given player to the
    given position and the cost/probability of failure.
    """
    orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)

    if game.ff_map is None:
        game.ff_map = FFTileMap(game)
    player_mover = FFMover(player, allow_skill_reroll=allow_skill_reroll)
    max_steps = player.num_moves_left() - 1 if not max_search_distance else max_search_distance
    finder = FFPathFinder(game.ff_map, max_steps)
    path = finder.find_path(player_mover, player.position.x, player.position.y, position.x, position.y)

    _reset_state(game, player, orig_player, orig_ball)

    return path


def get_safest_path_to_player(game, player, target_player, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=False):
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
    a position that is adjacent to the other player and the cost/probability of failure.
    """
    orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)

    if game.ff_map is None:
        game.ff_map = FFTileMap(game)
    player_mover = FFMover(player, allow_skill_reroll=allow_skill_reroll)
    max_steps = player.num_moves_left() - 1 if not max_search_distance else max_search_distance
    finder = FFPathFinder(game.ff_map, max_steps)
    path = finder.find_path(player_mover, player.position.x, player.position.y, player=target_player)

    _reset_state(game, player, orig_player, orig_ball)

    return path


def get_all_paths(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=False):
    """
    :param game:
    :param player: the player to move
    :param from_position: position to start movement from. If None, it will start from the player's current position.
    :param num_moves_used: the number of moves already used by the player. If None, it will use the player's current number of used moves.
    :param allow_skill_reroll: whether to allow skill re-rolls in the probability computations. If enabled, skill re-rolls can
    currently be used at every step regardless of whether it was used before.
    :param max_search_distance: the maximum search distance. If None, it will use the player's number of moves left.
    :return a list of paths, each containing the list of squares that forms the safest (and thereafter shortest) path for the given player to the
    given position and the cost/probability of failure, for each reachable square.
    """
    orig_player, orig_ball = _alter_state(game, player, from_position, num_moves_used)

    if game.ff_map is None:
        game.ff_map = FFTileMap(game)
    player_mover = FFMover(player, allow_skill_reroll=allow_skill_reroll)
    max_steps = player.num_moves_left() if not max_search_distance else max_search_distance
    finder = FFPathFinder(game.ff_map, max_steps)
    paths = finder.find_paths(player_mover, player.position.x, player.position.y)

    _reset_state(game, player, orig_player, orig_ball)

    return paths


def get_safest_scoring_path(game, player, from_position=None, num_moves_used=None, allow_skill_reroll=True, max_search_distance=None):
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
