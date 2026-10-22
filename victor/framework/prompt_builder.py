"""Compatibility facade: moved to :mod:`victor.agent.prompting.builder`.

Kept so the existing import path is unchanged. At runtime this name IS the
moved module (``sys.modules`` alias), so module state, singletons and
``mock.patch`` targets under this path all resolve to the one real module.
"""

import sys

from victor.agent.prompting import builder as _moved
from victor.agent.prompting.builder import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
