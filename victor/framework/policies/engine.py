"""Compatibility facade: moved to :mod:`victor.agent.policies.engine`."""

import importlib
import sys

from victor.agent.policies.engine import *  # noqa: F401,F403  (static re-export for type checkers)

_moved = importlib.import_module("victor.agent.policies.engine")

sys.modules[__name__] = _moved
