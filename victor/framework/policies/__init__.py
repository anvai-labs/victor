"""Compatibility facade: moved to :mod:`victor.agent.policies`.

Kept so the existing import paths are unchanged. At runtime this package and
each of its submodules ARE the moved modules (``sys.modules`` aliases), so a
submodule imported by its old dotted path is never loaded a second time.
"""

import importlib
import sys

from victor.agent import policies as _moved
from victor.agent.policies import *  # noqa: F401,F403  (static re-export for type checkers)

for _name in ("base", "builtins", "engine", "gate", "handlers", "middleware", "types"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"victor.agent.policies.{_name}")
sys.modules[__name__] = _moved
