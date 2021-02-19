"""
==========================
Author: Niels Justesen
Year: 2018
==========================
A few utilities used across the core modules.
"""

import os

from pytest import set_trace

import ffai


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


class Stack:
    def __init__(self):
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
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)


class LogEntry:
    def __init__(self, owner, key, from_val, to_val):
        self.owner = owner
        self.key = key
        self.from_val = from_val
        self.to_val = to_val

    def step_backward(self):
        self.owner.reset_to(self.key, self.from_val)

    def step_forward(self):
        self.owner.reset_to(self.key, self.to_val)


class LoggedState:
    def __init__(self):
        super().__setattr__("_logger", None)

    def __setattr__(self, key, value):
        if getattr(self, "_logger") is not None and hasattr(self, key):
            from_val = getattr(self, key)
            self._logger.log_state_change(LogEntry(self, key, from_val, value))
        super().__setattr__(key, value)

    def reset_to(self, key, value):
        super().__setattr__(key, value)

    def set_logger(self, logger):
        super().__setattr__("_logger", logger)


class Logger:
    def __init__(self):
        self.action_log = [[]]
        self.current_step = 0
        self.enabled = False

    def log_state_change(self, log_entry):
        if self.enabled:
            self.action_log[self.current_step].append(log_entry)

    def step_backward(self, to_step=0, clear_log=True):
        revert_actions = reversed([log_entry for step in self.action_log[to_step:] for log_entry in step])
        for log_entry in revert_actions:
            log_entry.step_backward()

        # Reset log to current step
        if clear_log:
            self.current_step = to_step
            self.action_log = self.action_log[:to_step]
            self.action_log.append([])

    def step_forward(self, to_step):
        raise NotImplementedError("Not yet done")

    def next_step(self):
        if self.enabled:
            self.action_log.append([])
            self.current_step += 1





def get_data_path(rel_path):
    root_dir = ffai.__file__.replace("__init__.py", "")
    filename = os.path.join(root_dir, "data/" + rel_path)
    return os.path.abspath(os.path.realpath(filename))


def compare_json(s1, s2, path=""):
    """ Assume they have the same keys"""
    keys  = s1.keys()
    diff = []
    for key in keys:
        if isinstance(s1[key], dict):
            diff.extend(compare_json(s1[key], s2[key], path+"/"+key))
        elif s1[key] != s2[key]:
            d = f"{path}/{key}: \n{s1[key]} \n{s2[key]}"
            diff.append(d)
    return diff