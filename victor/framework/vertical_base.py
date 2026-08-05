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

"""Compatibility shim for core vertical runtime types.

This module remains as a stable framework-facing surface for in-repo code and
incremental migrations, but new external vertical packages should prefer
``victor_contracts`` definition-layer contracts directly. In particular, external
authors should import ``VerticalBase`` from ``victor_contracts`` and publish
``victor.plugins`` entry points via a ``VictorPlugin`` wrapper.

Usage::

    # Compatibility shim for framework-side code
    from victor.framework.vertical_base import (
        VerticalBase,
        VerticalConfig,
        StageDefinition,
        StageBuilder,
        register_vertical,
        ExtensionDependency,
    )

    # Preferred contract for external vertical packages
    from victor_contracts import ExtensionDependency, VerticalBase, register_vertical
"""

from victor.core.verticals.base import VerticalBase, VerticalConfig
from victor.core.vertical_types import StageDefinition, StageBuilder
from victor_contracts import ExtensionDependency, register_vertical

__all__ = [
    "VerticalBase",
    "VerticalConfig",
    "StageDefinition",
    "StageBuilder",
    "register_vertical",
    "ExtensionDependency",
]
