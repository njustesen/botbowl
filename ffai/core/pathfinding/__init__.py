try:
    from .cython_pathfinding import *
except ImportError:
    from .python_pathfinding import *
