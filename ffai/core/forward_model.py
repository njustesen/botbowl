"""
==========================
Author: Mattias Bermell
Year: 2021
==========================
Classes and method for recursively tracing changes to objects.
"""
from copy import deepcopy, copy
from enum import Enum
from pytest import set_trace


class LoggedState:

    def __init__(self, ignored_keys=[]):
        super().__setattr__("_logger", None)
        super().__setattr__("_ignored_keys", set(ignored_keys))

    def __setattr__(self, key, to_value):
        if self.logger_initialized() and hasattr(self, key) and \
                key not in self._ignored_keys and to_value != getattr(self, key):
            from_value = getattr(self, key)
            to_value = add_logging(to_value, self._logger)
            self.log_this(AssignmentStep(self, key, from_value, to_value))
        super().__setattr__(key, to_value)

    def log_this(self, entry):
        self._logger.log_state_change(entry)

    def reset_to(self, key, value):
        super().__setattr__(key, value)

    def set_logger(self, logger):
        if self.logger_initialized():
            return

        super().__setattr__("_logger", logger)

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if attr_name[0] == "_" or callable(attr) or attr_name in self._ignored_keys:
                continue

            new_value = add_logging(attr, logger)

            super().__setattr__(attr_name, new_value)

    def logger_initialized(self):
        return self._logger is not None


class Trajectory:
    def __init__(self):
        self.action_log = [[]]
        self.current_step = 0
        self.enabled = False

    def log_state_change(self, log_entry):
        if self.enabled:
            self.action_log[self.current_step].append(log_entry)

    def step_backward(self, to_step=0):
        assert to_step <= self.current_step

        revert_actions = reversed([log_entry for step in self.action_log[to_step:] for log_entry in step])
        for log_entry in revert_actions:
            log_entry.undo()

        self.current_step = to_step

        # Reset log to current step
        self.action_log = self.action_log[:to_step]
        self.action_log.append([])

    def step_forward(self, to_step):
        raise NotImplementedError()

        assert to_step >= self.current_step

        revert_actions = [log_entry for step in self.action_log[self.current_step:to_step + 1] for log_entry in step]

        for log_entry in revert_actions:
            log_entry.redo()
        self.current_step = to_step

    def next_step(self):
        if self.enabled:
            self.action_log.append([])
            self.current_step += 1



class Step:
    def undo(self):
        raise NotImplementedError("Method to be overwritten by subclass")

    def redo(self):
        raise NotImplementedError("Method to be overwritten by subclass")


class CallableStep(Step):
    def __init__(self, owner, forward_func, forward_args, backward_func, backward_args):
        self.owner = owner
        self.forward_func = forward_func
        self.forward_args = forward_args
        self.backward_func = backward_func
        self.backward_args = backward_args

    def redo(self):
        self.forward_func(self.owner, *self.forward_args)

    def undo(self):
        self.backward_func(self.owner, *self.backward_args)

    def __repr__(self):
        return f"CallableStep(owner={self.owner}, redo={self.forward_func}, {self.forward_args}, " \
               f"undo={self.backward_func}, {self.backward_args}"


class AssignmentStep(Step):
    def __init__(self, owner, key, from_val, to_val):
        self.owner = owner
        self.key = key
        self.from_val = from_val
        self.to_val = to_val

    def undo(self):
        self.owner.reset_to(self.key, self.from_val)

    def redo(self):
        self.owner.reset_to(self.key, self.to_val)


class MovementStep(Step):
    def __init__(self, board, piece, square, put=True):
        self.board = board
        self.piece = piece
        self.square = square
        self.put = put

    def _put(self):
        self.board[self.square.y][self.square.x] = self.piece
        self.piece.position = self.square

    def _remove(self):
        self.board[self.square.y][self.square.x] = None
        self.piece.position = None

    def undo(self):
        if self.put:
            self._remove()
        else:
            self._put()

    def redo(self):
        if self.put:
            self._put()
        else:
            self._remove()


class LoggedList(list, LoggedState):

    def __init__(self, value):
        super().__init__(value)
        LoggedState.__init__(self)

    def set_logger(self, logger):
        if self.logger_initialized():
            return

        LoggedState.set_logger(self, logger)
        for i in range(len(self)):
            new_value = add_logging(self[i], logger)
            list.__setitem__(self, i, new_value)

    def append(self, value):
        if self.logger_initialized():
            value = add_logging(value, self._logger)
            log_entry = CallableStep(self, forward_func=list.append, forward_args=(value,),
                                     backward_func=list.pop, backward_args=())
            self.log_this(log_entry)

        list.append(self, value)

    def pop(self, i=None):
        if self.logger_initialized():
            if i is None:
                log_entry = CallableStep(self, forward_func=list.pop, forward_args=(),
                                         backward_func=list.append, backward_args=(self[-1],))
            else:
                log_entry = CallableStep(self, forward_func=list.pop, forward_args=(i,),
                                         backward_func=list.insert, backward_args=(i, self[i],))

            self.log_this(log_entry)
        return list.pop(self) if i is None else list.pop(self, i)

    def __setitem__(self, key, value):
        if self.logger_initialized():
            value = add_logging(value, self._logger)
            log_entry = CallableStep(self, forward_func=list.__setitem__, forward_args=(key, value),
                                     backward_func=list.__setitem__, backward_args=(key, self[key],))
            self.log_this(log_entry)
        return list.__setitem__(self, key, value)

    def clear(self):
        if self.logger_initialized():
            log_entry = CallableStep(self, forward_func=list.clear, forward_args=(),
                                     backward_func=list.extend, backward_args=(self[:],))
            self.log_this(log_entry)

        list.clear(self)

    def extend(self, value):
        raise NotImplementedError()

    def remove(self, value):
        if self.logger_initialized():
            log_entry = CallableStep(self, forward_func=list.remove, forward_args=(value,),
                                     backward_func=list.insert, backward_args=(self.index(value), value,))
            self.log_this(log_entry)

        list.remove(self, value)

    def __reduce__(self):
        func = LoggedList.init_LoggedList
        values = []
        values.extend(self)
        return func, (values, self._logger)

    def init_LoggedList(values, logger):  # Static method
        logged_list =  LoggedList(values)
        object.__setattr__(logged_list, "_logger", logger)
        return logged_list


class LoggedSet(set, LoggedState):
    def __init__(self, value):
        super().__init__(value)
        LoggedState.__init__(self)

    def set_logger(self, logger):
        if self.logger_initialized():
            return
        LoggedState.set_logger(self, logger)

        # TODO: call add logger for all items in the set.

    def add(self, value):
        if self.logger_initialized():
            value = add_logging(value, self._logger)
            log_entry = CallableStep(self, forward_func=set.add, forward_args=(value,),
                                     backward_func=set.remove, backward_args=(value,))
            self.log_this(log_entry)

        set.add(self, value)

    def clear(self):
        if self.logger_initialized():
            log_entry = CallableStep(self, forward_func=set.clear, forward_args=(),
                                     backward_func=set.update, backward_args=(copy(self),))
            self.log_this(log_entry)

        set.clear(self)

    def pop(self):
        raise NotImplementedError()

    def remove(self, value):
        raise NotImplementedError()

    def __reduce__(self):
        func = LoggedSet.init_LoggedSet
        values = set()
        values.update(self)
        return func, (values, self._logger)

    def init_LoggedSet(values, logger):  # Static method
        logged_set = LoggedSet(values)
        object.__setattr__(logged_set, "_logger", logger)
        return logged_set


class LoggedDict(dict, LoggedState):
    def __init__(self, value):
        super().__init__(value)
        LoggedState.__init__(self)

    def set_logger(self, logger):
        if self.logger_initialized():
            return

        LoggedState.set_logger(self, logger)
        for key in self:
            new_value = add_logging(self[key], logger)
            dict.__setitem__(self, key, new_value)

    def __setitem__(self, key, value):
        if key in self.keys():
            raise NotImplementedError()
        else:
            if self.logger_initialized():
                value = add_logging(value, self._logger)
            super().__setitem__(key, value)

    def pop(self, key):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def popitem(self):
        raise NotImplementedError()

    def __reduce__(self):
        func = LoggedDict.init_LoggedDict
        values = {}
        values.update(self)
        return func, (values, self._logger)

    def init_LoggedDict(values, logger):  # Static method
        logged_dict = LoggedDict(values)
        object.__setattr__(logged_dict, "_logger", logger)
        return logged_dict


def is_immutable(obj):
    return type(obj) in immutable_types or isinstance(obj, Enum) or isinstance(obj, Immutable)


replacement_type = [(list, LoggedList), (dict, LoggedDict), (set, LoggedSet)]
immutable_types = {int, float, str, tuple, bool, range, type(None)}


def add_logging(value, logger):
    if is_immutable(value):
        return value

    if isinstance(value, LoggedState):
        if not value.logger_initialized():
            value.set_logger(logger)
        return value

    new_types = [t[1] for t in replacement_type if type(value) == t[0]]
    if len(new_types) == 1:
        new_type = new_types.pop()
        new_value = new_type(value)
        new_value.set_logger(logger)
        return new_value
    else:
        raise AttributeError(f"Unable to add logging to {value}")


class Immutable:
    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError(f"{self} is an Immutable object. Its values can't be reassigned.")
        else:
            super().__setattr__(key, value)