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

"""Unit tests for removed SystemPromptCoordinator compatibility surface."""

from __future__ import annotations

import pytest


class TestSystemPromptCoordinator:
    """SystemPromptCoordinator should no longer be importable anywhere."""

    def test_system_prompt_runtime_module_removed(self) -> None:
        with pytest.raises(ImportError, match="system_prompt_runtime"):
            import victor.agent.services.system_prompt_runtime  # noqa: F401

    def test_services_package_no_longer_exports_system_prompt_coordinator(self) -> None:
        with pytest.raises(ImportError, match="SystemPromptCoordinator"):
            from victor.agent.services import SystemPromptCoordinator  # noqa: F401

    def test_coordinators_package_no_longer_exports_system_prompt_coordinator(
        self,
    ) -> None:
        with pytest.raises(ImportError, match="SystemPromptCoordinator"):
            from victor.agent.coordinators import SystemPromptCoordinator  # noqa: F401

    def test_canonical_state_passed_system_prompt_surface_still_exists(self) -> None:
        from victor.agent.coordinators.system_prompt_state_passed import (
            SystemPromptStatePassedCoordinator,
        )

        assert SystemPromptStatePassedCoordinator is not None
