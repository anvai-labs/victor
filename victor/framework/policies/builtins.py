"""Compatibility facade: moved to :mod:`victor.agent.policies.builtins`."""

import importlib
import sys

from victor.agent.policies.builtins import *  # noqa: F401,F403  (static re-export for type checkers)

_moved = importlib.import_module("victor.agent.policies.builtins")

sys.modules[__name__] = _moved
