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

"""Victor runtime vertical compatibility package.

This module exists primarily for runtime consumption and backward compatibility.
Historically it exposed in-repo and contrib vertical implementations directly, and
those imports remain available for compatibility and internal runtime integration.

New external vertical packages should not be authored against ``victor.verticals``.
Instead:
- define verticals against ``victor_contracts.VerticalBase``
- decorate them with ``victor_contracts.register_vertical``
- publish a thin ``VictorPlugin`` through the canonical ``victor.plugins``
  entry-point group

Core runtime loading, activation, and compatibility handling now live under
``victor.core.verticals``.
"""

from __future__ import annotations

__all__ = []
