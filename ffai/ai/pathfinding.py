try:
    from .fast_pathing import *
except ImportError:
    from .native_pathfinding import *