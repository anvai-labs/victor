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

"""victor.core.tool_types re-exports the SDK tool-dependency types (one definition)."""

import pytest


class TestSingleDefinition:
    """victor.core.tool_types re-exports the SDK types instead of copying them."""

    @pytest.mark.parametrize(
        "name, sdk_module",
        [
            ("ToolDependency", "victor_contracts.verticals.protocols.tools"),
            ("ToolDependencyProviderProtocol", "victor_contracts.verticals.protocols.tools"),
            ("EmptyToolDependencyProvider", "victor_contracts.verticals.tool_dependencies"),
        ],
    )
    def test_core_name_is_the_sdk_object(self, name, sdk_module):
        import importlib

        import victor.core.tool_types as core

        assert getattr(core, name) is getattr(importlib.import_module(sdk_module), name)

    def test_objects_interoperate_across_import_paths(self):
        from victor_contracts.verticals.protocols.tools import (
            ToolDependency as SdkToolDependency,
            ToolDependencyProviderProtocol as SdkProtocol,
        )

        from victor.core.tool_types import EmptyToolDependencyProvider, ToolDependency

        dep = ToolDependency(tool_name="edit", depends_on={"read"})
        assert isinstance(dep, SdkToolDependency)
        assert dep == SdkToolDependency(tool_name="edit", depends_on={"read"})
        assert isinstance(EmptyToolDependencyProvider("coding"), SdkProtocol)
