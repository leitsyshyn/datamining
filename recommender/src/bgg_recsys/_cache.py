"""Disk cache for expensive results (tuning, robustness), keyed by the function's small arguments.

joblib re-runs a cached function when its arguments or its own code change. It does not see changes in
the functions it calls: after changing those, clear the cache with `memory.clear()`.
"""

from joblib import Memory

from .config import Paths

memory = Memory(Paths().cache, verbose=0)
