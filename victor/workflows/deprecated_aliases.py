# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Module-level ``__getattr__``/``__dir__`` hooks for warning-backed renamed aliases.

Each shim module keeps its own alias map and states its removal contract
(version, date, migration guide) at the call site, where the repository
hygiene check (``scripts/ci/repo_hygiene_check.py``) expects to find it.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Mapping, MutableMapping

__all__ = ["make_deprecated_alias_getattr"]


def make_deprecated_alias_getattr(
    module_name: str,
    module_globals: MutableMapping[str, Any],
    alias_map: Mapping[str, Any],
    *,
    removal_version: str,
    removal_date: str,
    migration_guide: str,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build ``(__getattr__, __dir__)`` hooks that resolve deprecated aliases.

    Accessing an alias emits a ``DeprecationWarning`` attributed to the
    accessing code and returns the replacement object. Any other missing
    attribute raises ``AttributeError`` exactly like a plain module.

    Args:
        module_name: The shim module's ``__name__``.
        module_globals: The shim module's ``globals()`` (read live by ``__dir__``).
        alias_map: Deprecated name to replacement object; each replacement
            must have a ``__name__``.
        removal_version: Last release that keeps the aliases, e.g. ``"v0.9.0"``.
        removal_date: Target removal date, e.g. ``"2027-03-31"``.
        migration_guide: Repository path of the migration guide.

    Returns:
        The module ``__getattr__`` and ``__dir__`` functions.
    """

    def __getattr__(name: str) -> Any:
        if name in alias_map:
            warnings.warn(
                f"{name} is deprecated; use "
                f"{alias_map[name].__name__} instead. "
                f"This compatibility alias remains supported through "
                f"{removal_version} ({removal_date}) and "
                "will be removed after that milestone. "
                f"See {migration_guide} for migration guidance.",
                DeprecationWarning,
                stacklevel=2,
            )
            return alias_map[name]
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted(set(module_globals) | set(alias_map))

    return __getattr__, __dir__
