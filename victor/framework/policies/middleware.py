"""Compatibility facade: moved to :mod:`victor.agent.policies.middleware`."""

import sys

from victor.agent.policies import middleware as _moved
from victor.agent.policies.middleware import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
