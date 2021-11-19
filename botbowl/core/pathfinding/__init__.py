"""
These lines will import the compiled pathfinding module if it's available
if it's not available the ImportError is caught and python pathfinding is imported instead
"""
try:
    from .cython_pathfinding import *
except ImportError:
    from .python_pathfinding import *
