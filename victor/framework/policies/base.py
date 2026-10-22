"""Compatibility facade: moved to :mod:`victor.agent.policies.base`."""

import sys

from victor.agent.policies import base as _moved
from victor.agent.policies.base import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
