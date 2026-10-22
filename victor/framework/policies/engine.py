"""Compatibility facade: moved to :mod:`victor.agent.policies.engine`."""

import sys

from victor.agent.policies import engine as _moved
from victor.agent.policies.engine import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
