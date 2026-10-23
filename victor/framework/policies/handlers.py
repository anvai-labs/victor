"""Compatibility facade: moved to :mod:`victor.agent.policies.handlers`."""

import importlib
import sys

from victor.agent.policies.handlers import *  # noqa: F401,F403  (static re-export for type checkers)

_moved = importlib.import_module("victor.agent.policies.handlers")

sys.modules[__name__] = _moved
