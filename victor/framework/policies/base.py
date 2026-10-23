"""Compatibility facade: moved to :mod:`victor.agent.policies.base`."""

import importlib
import sys

from victor.agent.policies.base import *  # noqa: F401,F403  (static re-export for type checkers)

_moved = importlib.import_module("victor.agent.policies.base")

sys.modules[__name__] = _moved
