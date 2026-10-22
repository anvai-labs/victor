"""Compatibility facade: moved to :mod:`victor.agent.policies.gate`."""

import sys

from victor.agent.policies import gate as _moved
from victor.agent.policies.gate import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
