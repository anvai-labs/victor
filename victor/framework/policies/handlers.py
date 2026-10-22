"""Compatibility facade: moved to :mod:`victor.agent.policies.handlers`."""

import sys

from victor.agent.policies import handlers as _moved
from victor.agent.policies.handlers import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
