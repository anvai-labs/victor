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

"""Default complexity-to-mode mapping for verticals.

Provides ``DEFAULT_COMPLEXITY_MAP`` and a factory so external verticals can
bootstrap mode configuration without hardcoding the base mapping::

    from victor.framework.defaults import DEFAULT_COMPLEXITY_MAP, create_complexity_map

    CODING_COMPLEXITY = create_complexity_map(
        complex="thorough",
        highly_complex="architect",
    )
"""

from __future__ import annotations

from typing import Dict

DEFAULT_COMPLEXITY_MAP: Dict[str, str] = {
    "trivial": "quick",
    "simple": "quick",
    "moderate": "standard",
    "complex": "comprehensive",
    "highly_complex": "extended",
}
"""Mapping from complexity level to mode name.

Used by ``RegistryBasedModeConfigProvider.get_mode_for_complexity()``
and available to verticals as a starting point for customisation.
"""


def create_complexity_map(**overrides: str) -> Dict[str, str]:
    """Create a complexity map starting from defaults with overrides.

    Args:
        **overrides: Complexity level keys to override with new mode names.

    Returns:
        New dict with defaults plus any overrides applied.

    Example::

        CODING_COMPLEXITY = create_complexity_map(
            complex="thorough",
            highly_complex="architect",
        )
    """
    mapping = dict(DEFAULT_COMPLEXITY_MAP)
    mapping.update(overrides)
    return mapping


__all__ = [
    "DEFAULT_COMPLEXITY_MAP",
    "create_complexity_map",
]
