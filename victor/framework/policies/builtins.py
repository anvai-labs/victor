"""Compatibility facade: moved to :mod:`victor.agent.policies.builtins`."""

import sys

from victor.agent.policies import builtins as _moved
from victor.agent.policies.builtins import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
