"""Compatibility facade: moved to :mod:`victor.agent.execution_checkpoint`.

Kept so the existing import path is unchanged. At runtime this name IS the
moved module (``sys.modules`` alias), so module state, singletons and
``mock.patch`` targets under this path all resolve to the one real module.
"""

import importlib
import sys

from victor.agent.execution_checkpoint import *  # noqa: F401,F403  (static re-export for type checkers)

_moved = importlib.import_module("victor.agent.execution_checkpoint")

sys.modules[__name__] = _moved
