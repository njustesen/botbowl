"""
==========================
Author: Niels Justesen
Year: 2018
==========================
A few utilities used across the core modules.
"""

import os
from collections.abc import Iterable
from copy import copy
from typing import Sized

import botbowl
from botbowl.core.forward_model import Reversible
from botbowl.core.model import *


def parse_enum(enum_class, name):
    enum_name = name.upper().replace(" ", "_").replace("'", "").replace("é", "e").replace("-", "_")
    if enum_name not in enum_class.__members__:
        raise Exception("Unknown enum name " + enum_name + " (orig: " + name + ")")
    return enum_class[enum_name]


def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 3), (3, 3), (3, 4)]
    >>> print points2
    [(3, 4), (3, 3), (1, 3), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


class Stack(Reversible):
    def __init__(self):
        super().__init__()
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def remove(self, item):
        self.items.remove(item)

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


def get_data_path(rel_path):
    root_dir = botbowl.__file__.replace("__init__.py", "")
    filename = os.path.join(root_dir, "data/" + rel_path)
    return os.path.abspath(os.path.realpath(filename))


def compare_iterable(s1, s2, path=""):
    diff = []

    if type(s1) != type(s2):
        diff.append(f"{path}: __class__: '{type(s1)}' _notEqual_ '{type(s2)}'")

    elif hasattr(s1, "to_json"):
        diff.extend(compare_iterable(s1.to_json(), s2.to_json()))

    elif hasattr(s1, "compare"):
        diff.extend(s1.compare(s2, f"{path}"))

    elif isinstance(s1, Sized) and len(s1) != len(s2):
        diff.append(f"{path}: __len__: '{len(s1)}' _notEqual_ '{len(s2)}'")

    elif isinstance(s1, dict):
        for key in s1.keys():
            diff.extend(compare_iterable(s1[key], s2[key], f"{path}.{key}"))

    elif isinstance(s1, list):
        for i, (item1, item2) in enumerate(zip(s1, s2)):
            diff.extend(compare_iterable(item1, item2, f"{path}[{i}]"))

    else:
        if s1 != s2:
            d = f"{path}: '{s1}' _notEqual_ '{s2}'"
            diff.append(d)
    return diff


def compare_object(self, other, path="", ignored_keys=None, ignored_types=None):
    diff = []

    if type(self) != type(other):
        diff.append(f"{path}: __class__: '{type(self)}' _notEqual_ '{type(other)}'")
        return diff

    for attr_name in dir(self):
        self_attr = getattr(self, attr_name)
        if attr_name[0] == "_" or callable(self_attr) or \
                ignored_keys is not None and attr_name in ignored_keys or \
                ignored_types is not None and any([isinstance(self_attr, T) for T in ignored_types]):
            continue

        other_attr = getattr(other, attr_name)
        diff.extend(compare_iterable(self_attr, other_attr, path=f"{path}.{attr_name}"))
    return diff
