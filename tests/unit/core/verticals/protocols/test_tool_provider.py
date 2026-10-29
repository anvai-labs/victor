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

"""victor.core.verticals.protocols.tool_provider re-exports the SDK selection types (one definition)."""

import importlib

import pytest

MODULE = "victor.core.verticals.protocols.tool_provider"

PAIRS = [
    (
        "VerticalToolSelectionContext",
        "victor_contracts.verticals.protocols.promoted_types",
        "ToolSelectionContext",
    ),
    (
        "ToolSelectionContext",
        "victor_contracts.verticals.protocols.promoted_types",
        "ToolSelectionContext",
    ),
    (
        "ToolSelectionResult",
        "victor_contracts.verticals.protocols.promoted_types",
        "ToolSelectionResult",
    ),
]


@pytest.mark.parametrize("core_name, sdk_mod, sdk_name", PAIRS)
def test_core_name_is_the_sdk_object(core_name, sdk_mod, sdk_name):
    core = getattr(importlib.import_module(MODULE), core_name)
    assert core is getattr(importlib.import_module(sdk_mod), sdk_name)
