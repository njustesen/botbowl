"""
==========================
Author: Kevin Glass / Peter Moore
Year: 2019
==========================
This module contains a generic A* Path Finder, along with a specific test implementation.  Originally written in Java by
Kevin Glass: http://www.cokeandcode.com/main/tutorials/path-finding/ and converted to Python, with modifications, by
Peter Moore.
"""
from __future__ import annotations
import math
from interface import implements, Interface
from typing import Optional, List


class PathFinder(Interface):

    def find_path(self, mover: Mover, sx: int, sy: int, tx: int, ty: int) -> Path:
        pass


class Mover(Interface):
    pass


class AStarHeuristic(Interface):

    @staticmethod
    def get_cost(tile_map: TileBasedMap, mover: Mover, x: int, y: int, tx: int, ty: int) -> float:
        pass


class TileBasedMap(Interface):

    def get_width_in_tiles(self) -> int:
        pass

    def get_height_in_tiles(self) -> int:
        pass

    def path_finder_visited(self, x: int, y: int):
        pass

    def blocked(self, mover: Mover, x: int, y: int) -> bool:
        pass

    def get_cost(self, mover: Mover, sx: int, sy: int, tx: int, ty: int) -> float:
        pass


class Node:

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.cost: float = 0
        self._parent: Optional[Node] = None
        self.heuristic: float = 0
        self.depth: int = 0
        self.path: Optional[Path] = None  # Optionally, we can store a list of paths (solutions) at each node.

    @property
    def parent(self: Node):
        return self._parent

    @parent.setter
    def parent(self: Node, parent: Node):
        if parent is not None:
            self.depth = parent.depth + 1
        self._parent = parent

    def value(self: Node):
        return self.heuristic + self.cost


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


class Step:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self: Step, other: Step) -> bool:
        return (other.x == self.x) and (other.y == self.y)


class Path:

    def __init__(self):
        self.steps: List[Step] = []

    def __len__(self: Path) -> int:
        return len(self.steps)

    def get_step(self: Path, index: int) -> Step:
        return self.steps[index]

    def get_last_step(self: Path) -> Step:
        return self.steps[-1]

    def add_steps(self: Path, steps):
        self.steps.extend(steps)

    def get_x(self: Path, index: int) -> int:
        step = self.get_step(index)
        return step.x

    def get_y(self: Path, index: int) -> int:
        step = self.get_step(index)
        return step.y

    def append_step(self: Path, x: int, y: int):
        self.steps.append(Step(x, y))

    def remove_last(self: Path):
        del self.steps[-1]

    def prepend_step(self: Path, x: int, y: int):
        self.steps.insert(0, Step(x, y))

    def is_empty(self: Path) -> bool:
        return len(self) == 0

    def contains(self: Path, x: int, y: int) -> bool:
        return Step(x, y) in self.steps


class AStarPathFinder(implements(PathFinder)):

    def __init__(self, tile_map: TileBasedMap, max_search_distance: int, allow_diag_movement: bool, heuristic: AStarHeuristic):
        self.heuristic: AStarHeuristic = heuristic
        self.open: SortedList = SortedList(lambda x: x.cost + x.heuristic)
        self.closed: List[Node] = []
        self.tile_map: TileBasedMap = tile_map
        self.probability_costs: bool = False
        self.max_search_distance: int = max_search_distance
        self.allow_diag_movement: bool = allow_diag_movement
        self.nodes: List[List[Node]] = []
        for x in range(tile_map.get_width_in_tiles()):
            nodes_cur: List[Node] = []
            for y in range(tile_map.get_height_in_tiles()):
                nodes_cur.append(Node(x, y))
            self.nodes.append(nodes_cur)

    def set_costs_to_be_additive(self):
        self.probability_costs = False

    def find_path(self, mover: Mover, sx: int, sy: int, tx: int, ty: int) -> Optional[Path]:
        # easy first check, if the destination is blocked, we can't get there
        if self.tile_map.blocked(mover, tx, ty):
            return None

        # initial state for A*. The closed group is empty. Only the starting

        # tile is in the open list and it'e're already there
        self.nodes[sx][sy].cost = 0
        self.nodes[sx][sy].depth = 0
        self.closed.clear()
        self.open.clear()
        self.open.append(self.nodes[sx][sy])

        self.nodes[tx][ty].parent = None

        # while we haven'n't exceeded our max search depth
        max_depth = 0
        while (max_depth < self.max_search_distance) and (len(self.open) != 0):
            # pull out the first node in our open list, this is determined to
            # be the most likely to be the next step based on our heuristic
            current = self.get_first_in_open()
            if current == self.nodes[tx][ty]:
                break

            self.remove_from_open(current)
            self.add_to_closed(current)

            # search through all the neighbours of the current node evaluating
            # them as next steps
            for x in range(-1, 2):
                for y in range(-1, 2):
                    # not a neighbour, its the current tile
                    if (x == 0) and (y == 0):
                        continue

                    if not self.allow_diag_movement:
                        if (x != 0) and (y != 0):
                            continue

                    # determine the location of the neighbour and evaluate it
                    xp = x + current.x
                    yp = y + current.y

                    if self.is_valid_location(mover, sx, sy, xp, yp):
                        if self.probability_costs:
                            next_step_cost = 1.0 - ((1.0 - current.cost) * (1.0 - self.get_movement_cost(mover, current.x, current.y, xp, yp)))
                        else:
                            next_step_cost = current.cost + self.get_movement_cost(mover, current.x, current.y, xp, yp)

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
                            neighbour.heuristic = self.get_heuristic_cost(mover, xp, yp, tx, ty)
                            neighbour.parent = current
                            max_depth = max(max_depth, neighbour.cost)
                            self.add_to_open(neighbour)

        # since we'e've run out of search
        # there was no path. Just return null

        if self.nodes[tx][ty].parent is None:
            return None

        # At this point we've definitely found a path so we can uses the parent
        # references of the nodes to find out way from the target location back
        # to the start recording the nodes on the way.

        path = Path()
        target = self.nodes[tx][ty]
        while target != self.nodes[sx][sy]:
            path.prepend_step(target.x, target.y)
            target = target.parent

        path.prepend_step(sx, sy)
        return path

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

    def is_valid_location(self, mover: Mover, sx: int, sy: int, x: int, y: int) -> bool:
        valid = 0 <= x < self.tile_map.get_width_in_tiles() and 0 <= y < self.tile_map.get_height_in_tiles()
        valid = valid and not (sx == x and sy == y)
        valid = valid and not self.tile_map.blocked(mover, x, y)
        return valid

    def get_movement_cost(self, mover: Mover, sx: int, sy: int, tx: int, ty: int) -> float:
        return self.tile_map.get_cost(mover, sx, sy, tx, ty)

    def get_heuristic_cost(self, mover: Mover, x: int, y: int, tx: int, ty: int) -> float:
        return self.heuristic.get_cost(self.tile_map, mover, x, y, tx, ty)


class ClosestHeuristic(implements(AStarHeuristic)):

    @staticmethod
    def get_cost(tile_map: TileBasedMap, mover: Mover, x: int, y: int, tx: int, ty: int) -> float:
        dx = tx - x
        dy = ty - y
        cost = math.sqrt(float(dx*dx+dy*dy))
        return cost


class BruteForceHeuristic(implements(AStarHeuristic)):

    @staticmethod
    def get_cost(tile_map: TileBasedMap, mover: Mover, x: int, y: int, tx: int, ty: int) -> float:
        return 0.0


class TestMover(implements(Mover)):

    TANK = 0
    PLANE = 1
    BOAT = 2

    def __init__(self, type: int):
        self.type = type


class TestMap(implements(TileBasedMap)):

    WIDTH = 30
    HEIGHT = 30

    OPEN = 0
    BLOCKED = 1

    def __init__(self):
        self.terrain: List[List[int]] = []
        for x in range(self.WIDTH):
            terrain_cur: List[int] = []
            for y in range(self.HEIGHT):
                terrain_cur.append(0)
            self.terrain.append(terrain_cur)

        self.visited: List[List[bool]] = []
        for x in range(self.WIDTH):
            visited_cur: List[bool] = []
            for y in range(self.HEIGHT):
                visited_cur.append(False)
            self.visited.append(visited_cur)

        self.fill_area(0,0,5,5,self.BLOCKED)
        self.fill_area(0,5,3,10,self.BLOCKED)
        self.fill_area(0,5,3,10,self.BLOCKED)
        self.fill_area(0,15,7,15,self.BLOCKED)
        self.fill_area(7,26,22,4,self.BLOCKED)

        self.fill_area(16,13,5,5,self.BLOCKED)
        self.fill_area(17,5,10,3,self.BLOCKED)
        self.fill_area(20,8,5,3,self.BLOCKED)

        self.fill_area(8,2,7,3,self.BLOCKED)
        self.fill_area(10,5,3,3,self.BLOCKED)

    def fill_area(self, x: int, y: int, width: int, height: int, type: int):
        for xp in range(x, x+width):
            for yp in range(y, y+height):
                self.terrain[xp][yp] = type

    def clear_visited(self):
        for x in range(self.get_width_in_tiles()):
            for y in range(self.get_height_in_tiles()):
                self.visited[x][y] = False

    def visited(self, x: int, y: int) -> bool:
        return self.visited[x][y]

    def get_terrain(self, x: int, y: int) -> int:
        return self.terrain[x][y]

    def blocked(self, mover: Mover, x: int, y: int) -> bool:

        unit: int = mover.type  # For this simple demonstation, ignored.  We could have different terrain types
                                # that affect TANK, PLANE or BOAT differently

        return self.terrain[x][y] == self.BLOCKED

    def get_cost(self, mover: Mover, sx: int, sy: int, tx: int, ty: int) -> float:
        return 1

    def get_height_in_tiles(self) -> int:
        return self.WIDTH

    def get_width_in_tiles(self) -> int:
        return self.HEIGHT

    def path_finder_visited(self, x: int, y: int):
        self.visited[x][y] = True

    def display_map(self, start_pos, end_pos, steps=None):

        t = {
            self.OPEN: ' ',
            self.BLOCKED: 'X',
        }

        path: List[List[bool]] = []
        for i in range(self.get_width_in_tiles()):
            path_cur: List[bool] = []
            for j in range(self.get_height_in_tiles()):
                path_cur.append(False)
            path.append(path_cur)

        if steps is not None:
            for step in steps:
                path[step.x][step.y] = True

        for i in range(self.get_width_in_tiles()):
            print_ln_cur = ''
            for j in range(self.get_height_in_tiles()):
                c = t.get(self.terrain[i][j])
                if i == start_pos.x and j == start_pos.x:
                    print_ln_cur = print_ln_cur + 'S'
                elif i == end_pos.x and j == end_pos.x:
                    print_ln_cur = print_ln_cur + 'E'
                elif path[i][j]:
                    print_ln_cur = print_ln_cur + 'o'
                else:
                    print_ln_cur = print_ln_cur + c
            print(print_ln_cur)

def main():
    print('Running test path finder!!!')
    print(' ')
    test_map = TestMap()
    print(' ')
    heuristic = ClosestHeuristic()
    finder = AStarPathFinder(test_map, 500, True, heuristic)
    mover = TestMover(TestMover.TANK)
    path = finder.find_path(mover, 15, 15, 29, 29)
    if path is not None:
        test_map.display_map(Node(15, 15), Node(29, 29), path.steps)
    else:
        test_map.display_map(Node(15, 15), Node(29, 29))
if __name__ == "__main__":
    main()
