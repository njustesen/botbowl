"""
==========================
Author: Mattias Bermell
Year: 2021
==========================
Classes and method for recursively tracing changes to objects.
"""
from abc import ABC, abstractmethod
from copy import copy
from enum import Enum
from typing import Any, List


class Step(ABC):
    @abstractmethod
    def undo(self):
        pass

    @abstractmethod
    def redo(self):
        pass


class Reversible:
    _trajectory: 'Trajectory'
    _ignored_keys: set

    def __init__(self, ignored_keys=None):
        if ignored_keys is None:
            ignored_keys = []
        super().__setattr__("_trajectory", None)
        super().__setattr__("_ignored_keys", set(ignored_keys))

    def __setattr__(self, key, to_value):
        if self.trajectory_initialized() and hasattr(self, key) and \
                key not in self._ignored_keys and to_value != getattr(self, key):
            from_value = getattr(self, key)
            to_value = add_reversibility(to_value, self._trajectory)
            self.log_this(AssignmentStep(self, key, from_value, to_value))
        super().__setattr__(key, to_value)

    def log_this(self, entry: Any):
        self._trajectory.log_state_change(entry)

    def reset_to(self, key, value):
        super().__setattr__(key, value)

    def set_trajectory(self, trajectory):
        if self.trajectory_initialized():
            return

        super().__setattr__("_trajectory", trajectory)

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if attr_name[0] == "_" or callable(attr) or attr_name in self._ignored_keys:
                continue

            new_value = add_reversibility(attr, trajectory)

            super().__setattr__(attr_name, new_value)

    def trajectory_initialized(self):
        return self._trajectory is not None


class Trajectory:
    action_log: List[Step]
    enabled: bool

    def __init__(self):
        self.action_log = []
        self.enabled = False

    def __len__(self):
        return len(self.action_log)

    def log_state_change(self, log_entry: Any):
        if self.enabled:
            self.action_log.append(log_entry)

    def revert(self, to_step: int) -> List[Step]:
        assert 0 <= to_step <= len(self.action_log)

        reverted_steps = self.action_log[to_step:]
        self.action_log = self.action_log[:to_step]

        for log_entry in reversed(reverted_steps):
            log_entry.undo()

        return reverted_steps

    def step_forward(self, steps: List[Step]):
        for log_entry in steps:
            log_entry.redo()
        self.action_log.extend(steps)


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


class ReversibleList(list, Reversible):

    def __init__(self, value):
        super().__init__(value)
        Reversible.__init__(self)

    def set_trajectory(self, trajectory):
        if self.trajectory_initialized():
            return

        Reversible.set_trajectory(self, trajectory)
        for i in range(len(self)):
            new_value = add_reversibility(self[i], trajectory)
            list.__setitem__(self, i, new_value)

    def append(self, value):
        if self.trajectory_initialized():
            value = add_reversibility(value, self._trajectory)
            log_entry = CallableStep(self, forward_func=list.append, forward_args=(value,),
                                     backward_func=list.pop, backward_args=())
            self.log_this(log_entry)

        list.append(self, value)

    def pop(self, i=None):
        if self.trajectory_initialized():
            if i is None:
                log_entry = CallableStep(self, forward_func=list.pop, forward_args=(),
                                         backward_func=list.append, backward_args=(self[-1],))
            else:
                log_entry = CallableStep(self, forward_func=list.pop, forward_args=(i,),
                                         backward_func=list.insert, backward_args=(i, self[i],))

            self.log_this(log_entry)
        return list.pop(self) if i is None else list.pop(self, i)

    def __setitem__(self, key, value):
        if self.trajectory_initialized():
            value = add_reversibility(value, self._trajectory)
            log_entry = CallableStep(self, forward_func=list.__setitem__, forward_args=(key, value),
                                     backward_func=list.__setitem__, backward_args=(key, self[key],))
            self.log_this(log_entry)
        return list.__setitem__(self, key, value)

    def clear(self):
        if self.trajectory_initialized():
            log_entry = CallableStep(self, forward_func=list.clear, forward_args=(),
                                     backward_func=list.extend, backward_args=(self[:],))
            self.log_this(log_entry)

        list.clear(self)

    def extend(self, value):
        raise NotImplementedError()

    def remove(self, value):
        if self.trajectory_initialized():
            log_entry = CallableStep(self, forward_func=list.remove, forward_args=(value,),
                                     backward_func=list.insert, backward_args=(self.index(value), value,))
            self.log_this(log_entry)

        list.remove(self, value)

    def __reduce__(self):
        func = ReversibleList.init_reversible_list
        values = []
        values.extend(self)
        return func, (values, self._trajectory)

    @staticmethod
    def init_reversible_list(values, trajectory):
        logged_list = ReversibleList(values)
        object.__setattr__(logged_list, "_trajectory", trajectory)
        return logged_list


class ReversibleSet(set, Reversible):
    def __init__(self, value):
        super().__init__(value)
        Reversible.__init__(self)

    def set_trajectory(self, trajectory):
        if self.trajectory_initialized():
            return
        Reversible.set_trajectory(self, trajectory)

        # TODO: call add trajectory for all items in the set?

    def add(self, value):
        if self.trajectory_initialized():
            value = add_reversibility(value, self._trajectory)
            log_entry = CallableStep(self, forward_func=set.add, forward_args=(value,),
                                     backward_func=set.remove, backward_args=(value,))
            self.log_this(log_entry)

        set.add(self, value)

    def clear(self):
        if self.trajectory_initialized():
            log_entry = CallableStep(self, forward_func=set.clear, forward_args=(),
                                     backward_func=set.update, backward_args=(copy(self),))
            self.log_this(log_entry)

        set.clear(self)

    def pop(self):
        raise NotImplementedError()

    def remove(self, value):
        raise NotImplementedError()

    def __reduce__(self):
        func = ReversibleSet.init_reversible_set
        values = set()
        values.update(self)
        return func, (values, self._trajectory)

    @staticmethod
    def init_reversible_set(values, trajectory):  # Static method
        logged_set = ReversibleSet(values)
        object.__setattr__(logged_set, "_trajectory", trajectory)
        return logged_set


class ReversibleDict(dict, Reversible):
    def __init__(self, value):
        super().__init__(value)
        Reversible.__init__(self)

    def set_trajectory(self, trajectory):
        if self.trajectory_initialized():
            return

        Reversible.set_trajectory(self, trajectory)
        for key in self:
            new_value = add_reversibility(self[key], trajectory)
            dict.__setitem__(self, key, new_value)

    def __setitem__(self, key, value):
        if key in self.keys():
            raise NotImplementedError()
        else:
            if self.trajectory_initialized():
                value = add_reversibility(value, self._trajectory)
            super().__setitem__(key, value)

    def pop(self, key):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def popitem(self):
        raise NotImplementedError()

    def __reduce__(self):
        func = ReversibleDict.init_reversible_dict
        values = {}
        values.update(self)
        return func, (values, self._trajectory)

    @staticmethod
    def init_reversible_dict(values, trajectory):  # Static method
        logged_dict = ReversibleDict(values)
        object.__setattr__(logged_dict, "_trajectory", trajectory)
        return logged_dict


def is_immutable(obj):
    return type(obj) in immutable_types or isinstance(obj, Enum) or isinstance(obj, Immutable)


replacement_type = [(list, ReversibleList), (dict, ReversibleDict), (set, ReversibleSet)]
immutable_types = {int, float, str, tuple, bool, range, type(None)}


def treat_as_immutable(cls):
    """Used as decorator for classes that should never be tracked by forward model"""
    immutable_types.add(cls)
    return cls


def add_reversibility(value, trajectory):
    if is_immutable(value):
        return value

    if isinstance(value, Reversible):
        if not value.trajectory_initialized():
            value.set_trajectory(trajectory)
        return value

    new_types = [t[1] for t in replacement_type if type(value) == t[0]]
    if len(new_types) == 1:
        new_type = new_types.pop()
        new_value = new_type(value)
        new_value.set_trajectory(trajectory)
        return new_value
    else:
        raise AttributeError(f"Unable to add logging to {value}")


class Immutable:
    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError(f"{self} is an Immutable object. Its values can't be reassigned.")
        else:
            super().__setattr__(key, value)


def immutable_after_init(cls):
    """ Used as decorator to disallow attribute assignments after init"""

    old_init = cls.__init__

#    def setattr_error(self, attr_name, value):
#        raise AttributeError(f"Can't assign new value to {self} because class is immutable after init")


    def _setattr(self, key, value):
        if self.__setattr__ is None:
            raise AttributeError()
        object.__setattr__(self, key, value)


    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.__setattr__ = None

    cls.__init__ = init
    cls.__setattr__ = _setattr
    treat_as_immutable(cls)
    return cls


