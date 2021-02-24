"""
==========================
Author: Niels Justesen
Year: 2018
==========================
A few utilities used across the core modules.
"""

import os
from copy import copy

from pytest import set_trace

import ffai

from enum import Enum

from ffai.core.model import *


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


class GenericLogEntry:
    def __init__(self, owner, forward_func, forward_args, backward_func, backward_args):
        self.owner = owner
        self.forward_func = forward_func
        self.forward_args = forward_args
        self.backward_func = backward_func
        self.backward_args = backward_args

    def step_forward(self):
        self.forward_func(self.owner, *self.forward_args)

    def step_backward(self):
        try:
            self.backward_func(self.owner, *self.backward_args)
        except TypeError as error:
            set_trace()


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


class LogEntryBoard:
    def __init__(self, board, piece, square, put=True):
        self.board = board
        self.piece = piece
        self.square = copy(square)
        self.put = put

    def _put(self):
        self.board[self.square.y][self.square.x] = self.piece
        self.piece.position = copy(self.square)

    def _remove(self):
        self.board[self.square.y][self.square.x] = None
        self.piece.position = None

    def step_backward(self):
        if self.put:
            self._remove()
        else:
            self._put()

    def step_forward(self):
        if self.put:
            self._put()
        else:
            self._remove()


immutable_types = {int, float, str, tuple, bool, range, type(None)}


class LoggedState:

    def __init__(self, ignored_keys=[]):
        super().__setattr__("_logger", None)
        super().__setattr__("_ignored_keys", ignored_keys)

    def __setattr__(self, key, value):
        if self.logger_initialized() and hasattr(self, key):
            from_val = getattr(self, key)
            if not (key in self._ignored_keys or self._allowed_change(from_val, value)):

                new_value, successful = add_logging(value, self._logger)
                if not successful:

                    error_text = f"In owner {self}. Member attribute '{key}'\n" + \
                                 f"may not be reassigned to an object of type '{type(value)}' because the owner \n" + \
                                 f"a LoggedState and the object can't be converted to a LoggedState.\n" + \
                                 f"To solve this error, make sure this object can be converted."
                    raise AttributeError(error_text)
                else:
                    value = new_value
            self.log_this(LogEntry(self, key, from_val, value))
        super().__setattr__(key, value)

    def _allowed_change(self, from_val, to_val):

        from_val_ok = type(from_val) in immutable_types \
                      or isinstance(from_val, Enum) \
                      or isinstance(from_val, LoggedState)
        to_val_ok = type(to_val) in immutable_types \
                    or isinstance(to_val, Enum) or isinstance(to_val, LoggedState)

        return from_val_ok and to_val_ok

    def log_this(self, entry):  # To be used in derived classes
        self._logger.log_state_change(entry)

    def reset_to(self, key, value):
        super().__setattr__(key, value)

    def set_logger(self, logger):
        if self.logger_initialized():
            return

        super().__setattr__("_logger", logger)

        for attr_name in dir(self):
            if attr_name[0] == "_" or attr_name in self._ignored_keys:
                continue

            attr = getattr(self, attr_name)
            if callable(attr):
                continue
            new_value, successful = add_logging(attr, logger)  # TODO: raise error if not successful
            super().__setattr__(attr_name, new_value)

    def logger_initialized(self):
        return self._logger is not None


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

        self.current_step = to_step

        # Reset log to current step
        if clear_log:
            self.action_log = self.action_log[:to_step]
            self.action_log.append([])
        else:
            pass
            # raise NotImplementedError("Not clearing the log is yet to be implemented.")

    def step_forward(self, to_step):
        revert_actions = [log_entry for step in self.action_log[self.current_step:to_step + 1] for log_entry in step]

        for log_entry in revert_actions:
            log_entry.step_forward()
        self.current_step = to_step

    def next_step(self):
        if self.enabled:
            self.action_log.append([])
            self.current_step += 1


class LoggedList(list, LoggedState):

    def __init__(self, value):
        super().__init__(value)
        LoggedState.__init__(self)

    def set_logger(self, logger):
        if self.logger_initialized():
            return

        LoggedState.set_logger(self, logger)
        for i in range(len(self)):
            new_value, successful = add_logging(self[i], logger)  # TODO: raise error if not successful
            list.__setitem__(self, i, new_value)

    def append(self, value):
        if self.logger_initialized():
            log_entry = GenericLogEntry(self, forward_func=list.append, forward_args=(value,),
                                        backward_func=list.pop, backward_args=())
            self.log_this(log_entry)

            # TODO: consider calling add_logging() here too?
            if isinstance(value, LoggedState) and not value.logger_initialized():
                value.set_logger(self._logger)

        list.append(self, value)

    def pop(self, i=None):
        assert i is None  # Not implemented yet
        if self.logger_initialized():
            log_entry = GenericLogEntry(self, forward_func=list.pop, forward_args=(),
                                        backward_func=list.append, backward_args=(self[-1],))
            self.log_this(log_entry)
        return list.pop(self)

    def __setitem__(self, key, value):
        if self.logger_initialized():
            log_entry = GenericLogEntry(self, forward_func=list.__setitem__, forward_args=(key, value),
                                        backward_func=list.__setitem__, backward_args=(key, self[key],))
            self.log_this(log_entry)
        return list.__setitem__(self, key, value)

    def clear(self):
        if self.logger_initialized():
            log_entry = GenericLogEntry(self, forward_func=list.clear, forward_args=(),
                                        backward_func=list.extend, backward_args=(self[:],))
            self.log_this(log_entry)

        list.clear(self)

    def extend(self, value):
        raise NotImplementedError()

    def remove(self, value):
        if self.logger_initialized():
            log_entry = GenericLogEntry(self, forward_func=list.remove, forward_args=(value,),
                                        backward_func=list.insert, backward_args=(self.index(value), value,))
            self.log_this(log_entry)

        list.remove(self, value)


class LoggedSet(set, LoggedState):
    def __init__(self, value):
        super().__init__(value)
        LoggedState.__init__(self)

    def add(self, value):
        if self.logger_initialized():
            log_entry = GenericLogEntry(self, forward_func=set.add, forward_args=(value,),
                                        backward_func=set.remove, backward_args=(value,))
            self.log_this(log_entry)

            # TODO: consider calling add_logging() here too?
            if isinstance(value, LoggedState) and not value.logger_initialized():
                value.set_logger(self._logger)

        set.add(self, value)

    def pop(self):
        raise NotImplementedError()

    def remove(self, value):
        raise NotImplementedError()


class LoggedDict(dict, LoggedState):
    def __init__(self, value):
        super().__init__()
        raise NotImplementedError()


# replacement_type = [(list, LoggedList), (dict, LoggedDict), (set, LoggedSet)]
replacement_type = [(list, LoggedList), (set, LoggedSet)]


def add_logging(value, logger=None):
    if isinstance(value, LoggedState):
        if not value.logger_initialized():
            value.set_logger(logger)
        return value, True

    val_type = type(value)
    if isinstance(value, Enum) or val_type in immutable_types:
        return value, True

    new_types = [t[1] for t in replacement_type if val_type == t[0]]
    if len(new_types) == 1:
        new_type = new_types.pop()

        # try:
        new_value = new_type(value)
        # except TypeError:
        #    set_trace()
        if logger is not None:
            new_value.set_logger(logger)
        return new_value, True
    else:
        print(f"Not possible to add logging to type='{val_type}'.Obj={value}")
        return value, False


class Stack(LoggedState):
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
    root_dir = ffai.__file__.replace("__init__.py", "")
    filename = os.path.join(root_dir, "data/" + rel_path)
    return os.path.abspath(os.path.realpath(filename))


def compare_iterable(s1, s2, path=""):
    diff = []

    if type(s1) != type(s2):
        diff.append(f"{path}: __class__: '{type(s1)}' _notEqual_ '{type(s2)}'")

    elif isinstance(s1, dict):

        if len(s1) != len(s2):
            diff.append(f"{path}: __len__: '{len(s1)}' _notEqual_ '{len(s2)}'")
        else:
            for key in s1.keys():
                if isinstance(s1[key], dict) or isinstance(s1[key], list):
                    diff.extend(compare_iterable(s1[key], s2[key], path + "." + key))
                elif s1[key] != s2[key]:
                    d = f"{path}.{key}: '{s1[key]}' _notEqual_ '{s2[key]}'"
                    diff.append(d)
    elif isinstance(s1, list):

        if len(s1) != len(s2):
            diff.append(f"{path}: __len__: '{len(s1)}' _notEqual_ '{len(s2)}'")
        else:
            for i, (item1, item2) in enumerate(zip(s1, s2)):
                diff.extend(compare_iterable(item1, item2, f"{path}[{i}]"))
    else:
        if s1 != s2:
            d = f"{path}: '{s1}' _notEqual_ '{s2}'"
            diff.append(d)
    return diff
