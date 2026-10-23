"""Compatibility facade: moved to :mod:`victor.agent.policies.gate`."""

import importlib
import sys

from victor.agent.policies.gate import *  # noqa: F401,F403  (static re-export for type checkers)

_moved = importlib.import_module("victor.agent.policies.gate")

sys.modules[__name__] = _moved
