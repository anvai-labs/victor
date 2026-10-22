"""Compatibility facade: moved to :mod:`victor.agent.policies.types`."""

import sys

from victor.agent.policies import types as _moved
from victor.agent.policies.types import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
