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

"""Tiered import-boundary manifests shared by contract auditors and SDK test helpers.

Co-design review item 22a/22b: this module is the single source of truth that
the repo's import-boundary guards should converge on, replacing independently
maintained forbidden-prefix lists of differing strictness. Two tiers, by scope:

- ``DEFINITION_LAYER_FORBIDDEN_PREFIXES``: strictest tier, for a vertical's pure
  definition files (``assistant.py``, ``plugin.py``) that declare capabilities
  and must stay free of any runtime import. Consumed by
  ``tests/unit/contracts/test_contracts_import_boundaries.py`` in the victor
  monorepo.
- ``RUNTIME_LAYER_FORBIDDEN_PREFIXES``: looser tier, for a vertical's runtime/tool
  code, which may legitimately touch a documented extension surface (see
  ``ALLOWED_RUNTIME_IMPORT_PREFIXES`` in ``victor.core.verticals.contract_audit``).
  This is the union of the two prefix lists that independently existed before
  this module: ``victor.core.verticals.contract_audit``'s
  ``FORBIDDEN_RUNTIME_IMPORT_PREFIXES`` and this package's own (pre-existing)
  ``_DEFAULT_FORBIDDEN_PREFIXES`` in ``victor_contracts.testing``. The union
  only ever adds forbidden prefixes relative to either source list, so
  migrating a consumer from its local list onto this one can only tighten
  enforcement, never loosen it.

``KNOWN_VERTICAL_PACKAGE_NAMES`` is a separate, orthogonal list: the set of
first-party vertical *package names* (not ``victor.*`` prefixes) recognized by
the monorepo's own boundary guards. It replaces three independently
maintained copies of the same 6 names that existed across
``tests/unit/contracts/test_core_vertical_import_boundary.py``,
``tests/unit/contracts/test_contracts_import_boundaries.py``, and
``tests/unit/core/verticals/test_external_vertical_import_boundaries.py`` (the
last of which had drifted stale at only 3 of the 6 names).
"""

from __future__ import annotations

DEFINITION_LAYER_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "victor.agent",
    "victor.core",
    "victor.framework",
    "victor.providers",
    "victor.security",
    "victor.storage",
    "victor.tools",
    "victor.workflows",
    "victor.evaluation",
    "victor.observability",
)

RUNTIME_LAYER_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "victor.framework",
    "victor.core",
    "victor.security",
    "victor.agent",
    "victor.workflows",
    "victor.providers",
    "victor.evaluation",
    "victor.storage",
    "victor.config.settings",
    "victor.config.api_keys",
)

KNOWN_VERTICAL_PACKAGE_NAMES: tuple[str, ...] = (
    "victor_coding",
    "victor_devops",
    "victor_research",
    "victor_rag",
    "victor_dataanalysis",
    "victor_invest",
)
