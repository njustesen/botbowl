"""
==========================
Author: Peter Moore
Year: 2019
==========================
FFAI specific implementation of pathfinding.
"""
from __future__ import annotations
from ffai.util.pathfinding import *
import ffai.core.model as ff
from typing import List
import copy


class FfMover(implements(Mover)):

    def __init__(self, unit: ff.Player, from_square: ff.Square):

        self.square: ff.Square = from_square
        self.unit: ff.Player = unit
        self.move_allowed: int = unit.move_allowed()


class FfTileMap(implements(TileMap)):

    def __init__(self, game_state: ff.GameState, mover: FfMover):
        self.game_state: ff.GameState = game_state
        self.mover: FfMover = mover
        self.visited_list: List[ff.Square] = []

    def get_width_in_tiles(self) -> int:
        return self.game_state.pitch.width

    def get_height_in_tiles(self) -> int:
        return self.game_state.pitch.height

    def get_path(self) -> List[ff.Square]:
        return self.visited_list

    def path_finder_visited(self, x: int, y: int):
        self.visited_list.append(self.game_state.pitch.squares[x][y])

    def blocked(self, mover: FfMover, x: int, y: int) -> bool:
        square = self.game_state.pitch.squares[x][y]
        return self.game_state.pitch.get_player_at(square) is not None

    def get_cost(self, mover: FfMover, sx: int, sy: int, tx: int, ty: int) -> float:
        square_from = self.game_state.pitch.squares[sx][sy]
        square_to = self.game_state.pitch.squares[tx][ty]
        moving_unit = mover.unit
        agility = moving_unit.get_ag()

        cost = 0.0
        num_zones_from, tacklers, prehensile_tailers, diving_tacklers, shadowers, tentaclers = self.game_state.pitch.tackle_zones_detailed(mover.unit, square_from)

        if num_zones_from > 0:
            num_zones_to = self.game_state.pitch.num_tackle_zones_at(mover.unit, square_to)
            if moving_unit.has_skill(ff.Skill.STUNTY):
                num_zones_to = 0
            num_zones_to = num_zones_to - len(prehensile_tailers) - 2 * len(diving_tacklers)
            cost = (min(5.0, max(1.0, 5.0 - agility + num_zones_to)) / 6.0)
            if moving_unit.has_skill(ff.Skill.DODGE) and not tacklers:  # Should also check if already dodged
                cost = cost * cost
        cur_depth: int = mover.cur_depth  # essentially number of moves already done.
        if cur_depth != -1 and (cur_depth + 1 > moving_unit.move_allowed(include_gfi=False)):
            incr_cost = 1.0/6.0
            if self.game_state.weather == ff.WeatherType.BLIZZARD:
                incr_cost = incr_cost*2
            if moving_unit.has_skill(ff.Skill.SURE_FEET):
                incr_cost = incr_cost * incr_cost
            cost = 1 - (1 - cost)*(1 - incr_cost)
        return cost


class AllPathsFinder(AStarPathFinder):

    def __init__(self, game_state: ff.GameState, cur_unit: ff.Player, square: ff.Square, move_allowed: int = None):

        heuristic = BruteForceHeuristic()
        if move_allowed is None:
            move_allowed = cur_unit.move_allowed()
        mover = FfMover(cur_unit, square)
        tile_map = FfTileMap(game_state, mover)
        max_search_distance = move_allowed
        allow_diag_movement = True
        super().__init__(tile_map, max_search_distance, allow_diag_movement, heuristic)

        self.find_all_paths()
        self.set_all_paths()

    def get_valid_move_nodes(self) -> List[Node]:
        return_nodes: List[Node] = []
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[0])):
                cur: Node = self.nodes[i][j]
                if cur.path is not None:
                    return_nodes.append(cur)
                elif cur.parent is not None or self.game_state.pitch.squares[cur.x][cur.y] == self.tile_map.mover.square:
                    return_nodes.append(cur)
        return return_nodes

    def find_all_leaping_paths(self):

        replacement_nodes: List[List[Node]] = []
        for x in range(self.tile_map.get_width_in_tiles()):
            replacement_nodes_cur: List[Optional[Node]] = []
            for y in range(self.tile_map.get_height_in_tiles()):
                replacement_nodes_cur.append(copy.deepcopy(self.nodes[x][y]))
            replacement_nodes.append(replacement_nodes_cur)
        end_first_move_nodes = self.get_valid_move_nodes()
        for end_first_move_node in end_first_move_nodes:
            first_move_path = end_first_move_node.path
            if len(first_move_path)-1 <= self.mover.unit.move_allowed()-2:
                leap_to_squares = self.game_state.pitch.get_adjacent_squares(self.game_state.pitch.squares[end_first_move_node.x, end_first_move_node.y], exclude_occupied=True, include_leap=True)
                basic_cost = first_move_path.get_cost()
                for leap_to_square in leap_to_squares:
                    path_finder_second_move = AllPathsFinder(self.game_state, self.mover.unit, leap_to_square, move_allowed=len(first_move_path)-1+2)
                    end_second_move_nodes = path_finder_second_move.get_valid_move_nodes()
                    for end_second_move_node in end_second_move_nodes:
                        extra_cost = end_second_move_node.cost
                        leap_ag = self.mover.unit.get_ag()
                        if self.mover.unit.has_skill(ff.Skill.VERY_LONG_LEGS):
                            leap_ag = leap_ag + 1
                        leap_cost = 1 - leap_ag / 6
                        extra_cost = 1 - (1-leap_cost) * (1-extra_cost)
                        total_cost = 1 - (1-extra_cost) * (1-basic_cost)
                        old_cost = replacement_nodes[end_second_move_node.x][end_second_move_node.y].cost
                        if total_cost <= old_cost or replacement_nodes[end_second_move_node.x][end_second_move_node.y].path is None:
                            replacement_nodes[end_second_move_node.x][end_second_move_node.y].cost = total_cost
                            new_path = copy.deepcopy(first_move_path)
                            new_path.add_steps(end_second_move_node.path.steps)
                            replacement_nodes[end_second_move_node.x][end_second_move_node.y].path = new_path
        self.nodes = replacement_nodes

    def find_all_paths(self):

        sx = self.tile_map.mover.square.x
        sy = self.tile_map.mover.square.y

        self.nodes[sx][sy].cost = 0
        self.nodes[sx][sy].depth = 0
        self.closed.clear()
        self.open.clear()
        self.open.append(self.nodes[sx][sy])      # Start with starting node.

        while len(self.open) != 0:
            current = self.get_first_in_open()
            self.remove_from_open(current)
            self.add_to_closed(current)
            if current.depth == self.max_search_distance:
                continue
            if current.depth >= self.max_search_distance:
                current.parent = None
                continue
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if (x == 0) and (y == 0):
                        continue

                    xp = x + current.x
                    yp = y + current.y

                    if self.is_valid_location(self.mover, sx, sy, xp, yp):
                        self.mover.cur_depth = current.depth
                        next_step_cost = 1.0 - ((1.0 - current.cost) * (1.0 - self.get_movement_cost(self.mover, current.x, current.y, xp, yp)))
                        neighbour = self.nodes[xp][yp]
                        self.tile_map.path_finder_visited(xp, yp)

                        if next_step_cost < neighbour.cost:
                            if self.in_open_list(neighbour):
                                self.remove_from_open(neighbour)

                        if self.in_closed_list(neighbour):
                            self.remove_from_closed(neighbour)

                        if not self.in_open_list(neighbour) and not self.in_closed_list(neighbour):
                            neighbour.cost = next_step_cost
                            neighbour.parent = current
                            self.add_to_open(neighbour)

