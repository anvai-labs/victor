# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backward-compatibility shim: ``victor_sdk`` was renamed to ``victor_contracts``.

External plugins pinned to the old import path keep working — both top-level
(``from victor_sdk import PluginContext``) and submodule
(``from victor_sdk.capabilities import ...``) imports resolve to the matching
``victor_contracts`` module — but emit a ``DeprecationWarning``. New code should
import from ``victor_contracts`` directly.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import victor_contracts

warnings.warn(
    "The 'victor_sdk' package was renamed to 'victor_contracts'. Importing from "
    "'victor_sdk' is deprecated and will be removed in a future release; import "
    "from 'victor_contracts' instead.",
    DeprecationWarning,
    stacklevel=2,
)

_PREFIX = "victor_sdk"
_TARGET = "victor_contracts"


class _VictorSdkRedirect(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve any ``victor_sdk.<sub>`` import to the same ``victor_contracts.<sub>``.

    Aliases to the *same* module object (no duplicate classes), so isinstance and
    identity checks across the old/new names stay consistent.
    """

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname != _PREFIX and not fullname.startswith(_PREFIX + "."):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):  # noqa: ANN001
        target_name = _TARGET + spec.name[len(_PREFIX) :]
        module = importlib.import_module(target_name)  # nosemgrep
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):  # noqa: ANN001
        pass


if not any(isinstance(finder, _VictorSdkRedirect) for finder in sys.meta_path):
    sys.meta_path.insert(0, _VictorSdkRedirect())

# Eagerly re-export the documented public API so ``from victor_sdk import *`` and
# direct ``from victor_sdk import Name`` both work without hitting the finder.
from victor_contracts import *  # noqa: F401,F403,E402  (warn before re-export)

try:
    __all__ = list(victor_contracts.__all__)  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - victor_contracts always defines __all__
    __all__ = []


def __getattr__(name: str):  # PEP 562 — delegate any other attribute access.
    return getattr(victor_contracts, name)
