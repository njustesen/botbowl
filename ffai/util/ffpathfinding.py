"""
==========================
Author: Peter Moore
Year: 2019
==========================
FFAI mover and tilemap -> specific implementations for the FFAI project.  This version does NOT handle Leap actions.

WARNING: COMPLETELY UNTESTED - WORK IN PROGRESS
"""
from __future__ import annotations
from ffai.util.pathfinding import *
import ffai.core.model as ff
from typing import List


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
