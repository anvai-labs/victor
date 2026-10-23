"""Compatibility facade: moved to :mod:`victor.config.classification_backend`.

Kept so the existing import path is unchanged. At runtime this name IS the
moved module (``sys.modules`` alias), so module state, singletons and
``mock.patch`` targets under this path all resolve to the one real module.
"""

import sys

from victor.config import classification_backend as _moved
from victor.config.classification_backend import *  # noqa: F401,F403  (static re-export for type checkers)

sys.modules[__name__] = _moved
