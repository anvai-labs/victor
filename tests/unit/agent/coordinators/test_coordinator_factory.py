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

"""Regression coverage for CoordinatorFactory compatibility seams."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from victor.agent.coordinators.coordinator_factory import CoordinatorFactory
from victor_contracts.conversation import ConversationCoordinator
from victor_contracts.safety import SafetyCoordinator


class ExplodingContainer:
    """Container stub that fails if compatibility surfaces reach into DI."""

    def get(self, service_type: Any) -> Any:
        raise AssertionError(f"unexpected get({service_type!r}) for compatibility surface")

    def get_optional(self, service_type: Any) -> Any:
        raise AssertionError(f"unexpected get_optional({service_type!r}) for compatibility surface")


class TestCoordinatorFactoryCompatibility:
    """Ensure the factory's legacy coordinator methods stay compatibility-only."""

    def test_create_system_prompt_coordinator_removed(self) -> None:
        factory = CoordinatorFactory(ExplodingContainer())

        assert not hasattr(factory, "create_system_prompt_coordinator")

    def test_create_safety_coordinator_returns_sdk_surface(self) -> None:
        factory = CoordinatorFactory(ExplodingContainer())

        with pytest.warns(
            DeprecationWarning,
            match=r"create_safety_coordinator\(\) is a deprecated SDK-owned compatibility surface",
        ):
            coordinator = factory.create_safety_coordinator()

        assert isinstance(coordinator, SafetyCoordinator)

    def test_create_conversation_coordinator_returns_sdk_surface(self) -> None:
        factory = CoordinatorFactory(ExplodingContainer())

        with pytest.warns(
            DeprecationWarning,
            match=(
                r"create_conversation_coordinator\(\) is a deprecated SDK-owned "
                r"compatibility surface"
            ),
        ):
            coordinator = factory.create_conversation_coordinator()

        assert isinstance(coordinator, ConversationCoordinator)

    def test_factory_no_longer_imports_removed_local_conversation_or_safety_modules(
        self,
    ) -> None:
        source = Path("victor/agent/coordinators/coordinator_factory.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        assert "victor.agent.coordinators.safety_coordinator" not in imported_modules
        assert "victor.agent.coordinators.conversation_coordinator" not in imported_modules
