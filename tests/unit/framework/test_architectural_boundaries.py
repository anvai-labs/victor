# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Architectural boundary tests to prevent regression to legacy patterns.

These tests enforce:
1. UI layer (CLI, Web) must NOT import AgentOrchestrator directly
2. UI layer must use VictorClient or Agent facade
3. VictorClient must use SessionConfig (not VictorConfig)
4. VictorClient must access services (not bypass to orchestrator)
5. SessionConfig must be used for CLI/runtime overrides

Run with: pytest tests/unit/framework/test_architectural_boundaries.py -v
"""

import pytest
import ast
import os
from pathlib import Path


class TestUILayerArchitecturalBoundaries:
    """Test that UI layer doesn't violate architectural boundaries."""

    @pytest.fixture
    def ui_layer_files(self):
        """Get all UI layer Python files."""
        repo_root = Path(__file__).parent.parent.parent.parent
        ui_dirs = [
            repo_root / "victor" / "ui",
            repo_root / "victor" / "commands",
            # The Textual observability dashboard is a UI surface too (UX P5):
            # it must consume framework/client surfaces, never victor.agent.*.
            repo_root / "victor" / "observability" / "dashboard",
        ]
        files = []
        for ui_dir in ui_dirs:
            if ui_dir.exists():
                files.extend(ui_dir.rglob("*.py"))
        return files

    def test_ui_layer_must_not_import_orchestrator_directly(self, ui_layer_files):
        """UI layer MUST NOT import AgentOrchestrator directly.

        This ensures UI layer goes through VictorClient or Agent facade,
        not bypassing to internal orchestrator.
        """
        violations = []

        for file_path in ui_layer_files:
            # Skip test files
            if "test_" in file_path.name or "__tests__" in str(file_path):
                continue

            try:
                with open(file_path, "r") as f:
                    tree = ast.parse(f.read(), filename=str(file_path))

                for node in ast.walk(tree):
                    # Check for direct imports of AgentOrchestrator
                    if isinstance(node, ast.ImportFrom):
                        if node.module and "orchestrator" in node.module:
                            for alias in node.names:
                                if "AgentOrchestrator" in alias.name:
                                    violations.append(
                                        {
                                            "file": str(
                                                file_path.relative_to(
                                                    Path(__file__).parent.parent.parent.parent
                                                )
                                            ),
                                            "line": node.lineno,
                                            "import": f"from {node.module} import {alias.name}",
                                        }
                                    )

                    # Check for direct imports
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if "orchestrator" in alias.name and "AgentOrchestrator" in alias.name:
                                violations.append(
                                    {
                                        "file": str(
                                            file_path.relative_to(
                                                Path(__file__).parent.parent.parent.parent
                                            )
                                        ),
                                        "line": node.lineno,
                                        "import": f"import {alias.name}",
                                    }
                                )
            except Exception as e:
                # Skip files that can't be parsed
                pass

        if violations:
            pytest.fail(
                "UI layer must NOT import AgentOrchestrator directly.\n"
                "Use VictorClient or Agent facade instead.\n\n"
                "Violations:\n"
                + "\n".join(f"  - {v['file']}:{v['line']}: {v['import']}" for v in violations)
            )

    def test_ui_layer_must_not_import_framework_shim(self, ui_layer_files):
        """UI layer MUST NOT import FrameworkShim (deprecated)."""
        violations = []

        for file_path in ui_layer_files:
            if "test_" in file_path.name or "__tests__" in str(file_path):
                continue

            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    if "from victor.framework.shim import FrameworkShim" in content:
                        violations.append(
                            {
                                "file": str(
                                    file_path.relative_to(
                                        Path(__file__).parent.parent.parent.parent
                                    )
                                ),
                                "import": "FrameworkShim import found",
                            }
                        )
            except Exception:
                pass

        if violations:
            pytest.fail(
                "UI layer must NOT import FrameworkShim (deprecated).\n"
                "Use VictorClient or Agent.create() instead.\n\n"
                "Violations:\n" + "\n".join(f"  - {v['file']}: {v['import']}" for v in violations)
            )

    def test_ui_layer_must_not_import_legacy_session_persistence(self, ui_layer_files):
        """UI layer must use ConversationStore, never the deprecated session shim."""
        violations = []

        for file_path in ui_layer_files:
            if "test_" in file_path.name or "__tests__" in str(file_path):
                continue

            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "victor.agent.sqlite_session_persistence"
                ):
                    violations.append(
                        f"{file_path.relative_to(Path(__file__).parent.parent.parent.parent)}:{node.lineno}"
                    )

        if violations:
            pytest.fail(
                "UI layer must use ConversationStore instead of the deprecated "
                "SQLiteSessionPersistence shim.\n\nViolations:\n  - " + "\n  - ".join(violations)
            )

    def test_ui_layer_must_not_call_private_victor_client_methods(self, ui_layer_files):
        """UI layer MUST NOT reach into VictorClient private helpers."""
        violations = []

        for file_path in ui_layer_files:
            if "test_" in file_path.name or "__tests__" in str(file_path):
                continue

            try:
                with open(file_path, "r") as f:
                    for lineno, line in enumerate(f, start=1):
                        if "._ensure_initialized(" in line:
                            violations.append(
                                {
                                    "file": str(
                                        file_path.relative_to(
                                            Path(__file__).parent.parent.parent.parent
                                        )
                                    ),
                                    "line": lineno,
                                    "content": line.strip(),
                                }
                            )
            except Exception:
                pass

        if violations:
            pytest.fail(
                "UI layer must NOT call private VictorClient methods.\n"
                "Use VictorClient.initialize() or other public surfaces instead.\n\n"
                "Violations:\n"
                + "\n".join(f"  - {v['file']}:{v['line']}: {v['content']}" for v in violations)
            )


class TestVictorClientArchitecturalBoundaries:
    """Test that VictorClient follows architectural patterns."""

    def test_victor_client_must_accept_session_config(self):
        """VictorClient.__init__ MUST accept SessionConfig (not VictorConfig)."""
        from victor.framework.client import VictorClient
        import inspect

        sig = inspect.signature(VictorClient.__init__)
        params = sig.parameters

        # Check for config parameter
        if "config" not in params:
            pytest.fail("VictorClient.__init__ must have 'config' parameter")

        # Check type hint
        param = params["config"]
        if param.annotation and "SessionConfig" not in str(param.annotation):
            # Allow generic annotations
            if not any(x in str(param.annotation) for x in ["Any", "object", "SessionConfig"]):
                pytest.fail(
                    f"VictorClient.__init__ 'config' parameter should accept SessionConfig, "
                    f"but has annotation: {param.annotation}"
                )

    def test_victor_client_must_use_session_config_in_docstring(self):
        """VictorClient must document SessionConfig usage."""
        from victor.framework.client import VictorClient

        docstring = VictorClient.__doc__ or ""
        if "SessionConfig" not in docstring:
            pytest.fail(
                "VictorClient docstring must mention SessionConfig usage.\n"
                "This documents the proper pattern for CLI/runtime overrides."
            )

    def test_victor_client_must_not_use_victor_config(self):
        """VictorClient MUST NOT use VictorConfig (legacy)."""
        from victor.framework import client

        # Read the source file
        source_file = Path(client.__file__)
        with open(source_file, "r") as f:
            content = f.read()

        # Check for VictorConfig usage
        if (
            "VictorConfig" in content
            and "from victor.framework.config_models import VictorConfig" in content
        ):
            pytest.fail(
                "VictorClient must use SessionConfig, not VictorConfig.\n"
                "VictorConfig is the legacy pattern."
            )

    def test_victor_client_chat_returns_canonical_task_result(self):
        """VictorClient.chat() must expose TaskResult, not a client-only wrapper."""
        from victor.framework.client import VictorClient
        import inspect

        signature = inspect.signature(VictorClient.chat)
        return_annotation = signature.return_annotation

        if "TaskResult" not in str(return_annotation):
            pytest.fail(
                "VictorClient.chat() must return TaskResult.\n"
                "Framework clients should reuse the canonical framework execution result "
                "instead of defining parallel chat-result wrappers."
            )


class TestSessionConfigArchitecturalBoundaries:
    """Test that SessionConfig is used properly."""

    def test_session_config_must_be_immutable(self):
        """SessionConfig must be frozen=True (immutable)."""
        from victor.framework.session_config import SessionConfig
        import dataclasses

        # Check that SessionConfig is a frozen dataclass
        if not dataclasses.is_dataclass(SessionConfig):
            pytest.fail("SessionConfig must be a dataclass")

        # Try to create an instance and check if it's frozen
        try:
            config = SessionConfig()
            # Try to mutate (should fail)
            config.tool_budget = 999
            pytest.fail("SessionConfig must be frozen=True (immutable)")
        except (dataclasses.FrozenInstanceError, AttributeError):
            # Expected - frozen dataclass
            pass

    def test_session_config_has_apply_to_settings_method(self):
        """SessionConfig must have apply_to_settings() method."""
        from victor.framework.session_config import SessionConfig

        if not hasattr(SessionConfig, "apply_to_settings"):
            pytest.fail(
                "SessionConfig must have apply_to_settings() method.\n"
                "This is the ONLY place where Settings should be mutated from session config."
            )

    def test_session_config_has_from_cli_flags_method(self):
        """SessionConfig must have from_cli_flags() factory method."""
        from victor.framework.session_config import SessionConfig

        if not hasattr(SessionConfig, "from_cli_flags"):
            pytest.fail(
                "SessionConfig must have from_cli_flags() class method.\n"
                "This is the primary factory for CLI code."
            )


class TestAgentFacadeArchitecturalBoundaries:
    """Test that Agent facade follows architectural patterns."""

    def test_agent_create_must_accept_session_config(self):
        """Agent.create() MUST accept session_config parameter."""
        from victor.framework.agent import Agent
        import inspect

        sig = inspect.signature(Agent.create)
        params = sig.parameters

        if "session_config" not in params:
            pytest.fail(
                "Agent.create() must accept session_config parameter.\n"
                "This allows CLI/runtime overrides via SessionConfig."
            )

    def test_agent_create_docstring_mentions_session_config(self):
        """Agent.create() docstring must mention SessionConfig."""
        from victor.framework.agent import Agent

        docstring = Agent.create.__doc__ or ""
        if "SessionConfig" not in docstring:
            pytest.fail(
                "Agent.create() docstring must mention SessionConfig.\n"
                "This documents the proper pattern for CLI/runtime overrides."
            )


class TestServiceLayerArchitecturalBoundaries:
    """Test that service layer is properly structured."""

    def test_services_exist_in_services_module(self):
        """Core services must exist in victor.agent.services."""
        service_modules = [
            "victor.agent.services.chat_service",
            "victor.agent.services.tool_service",
            "victor.agent.services.session_service",
            "victor.agent.services.context_service",
            "victor.agent.services.provider_service",
            "victor.agent.services.recovery_service",
        ]

        for module_name in service_modules:
            try:
                module = __import__(module_name, fromlist=[""])
                # Check if the service class exists
                service_class_name = (
                    module_name.split(".")[-1].replace("_service", "").title().replace("_", "")
                    + "Service"
                )
                if not hasattr(module, service_class_name):
                    # Try alternate naming
                    if not hasattr(module, "ChatService") and "chat" in module_name:
                        pytest.fail(f"{module_name} must export ChatService")
                    elif not hasattr(module, "ToolService") and "tool" in module_name:
                        pytest.fail(f"{module_name} must export ToolService")
            except ImportError as e:
                pytest.fail(f"Service module {module_name} must exist: {e}")

    def test_service_accessor_exists(self):
        """ServiceAccessor must exist for accessing services."""
        from victor.runtime.context import ServiceAccessor

        # Check that ServiceAccessor has service properties
        required_services = [
            "chat",
            "tool",
            "session",
            "context",
            "provider",
            "recovery",
        ]
        for service in required_services:
            if not hasattr(ServiceAccessor, service):
                pytest.fail(
                    f"ServiceAccessor must have '{service}' property.\n"
                    f"This allows accessing {service.upper()}Service."
                )


class TestRegressionGuards:
    """Regression guards to prevent sliding back to legacy patterns."""

    @pytest.fixture
    def ui_layer_files(self):
        """Get all UI layer Python files."""
        repo_root = Path(__file__).parent.parent.parent.parent
        ui_dirs = [
            repo_root / "victor" / "ui",
            repo_root / "victor" / "commands",
            # The Textual observability dashboard is a UI surface too (UX P5):
            # it must consume framework/client surfaces, never victor.agent.*.
            repo_root / "victor" / "observability" / "dashboard",
        ]
        files = []
        for ui_dir in ui_dirs:
            if ui_dir.exists():
                files.extend(ui_dir.rglob("*.py"))
        return files

    def test_no_agent_factory_in_ui_layer(self, ui_layer_files):
        """UI layer must NOT directly instantiate AgentFactory."""
        violations = []

        for file_path in ui_layer_files:
            if "test_" in file_path.name or "__tests__" in str(file_path):
                continue

            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    # Check for AgentFactory instantiation
                    if "AgentFactory(" in content:
                        violations.append(
                            {
                                "file": str(
                                    file_path.relative_to(
                                        Path(__file__).parent.parent.parent.parent
                                    )
                                ),
                                "pattern": "AgentFactory instantiation",
                            }
                        )
            except Exception:
                pass

        if violations:
            pytest.fail(
                "UI layer must NOT instantiate AgentFactory directly.\n"
                "Use VictorClient or Agent.create() instead.\n\n"
                "Violations:\n" + "\n".join(f"  - {v['file']}: {v['pattern']}" for v in violations)
            )

    def test_no_settings_mutation_in_ui_layer(self, ui_layer_files):
        """UI layer must NOT mutate Settings directly (use SessionConfig)."""
        # This is a soft check - we can't catch all mutations, but we can check for patterns
        pass  # Implement with AST analysis if needed

    def test_session_config_is_frozen(self):
        """SessionConfig must be frozen (immutable) at definition."""
        from victor.framework.session_config import SessionConfig
        import dataclasses

        dc_fields = dataclasses.fields(SessionConfig)
        # Check if the dataclass is frozen
        # Note: We can't directly check if it's frozen, but we can test behavior
        try:
            config = SessionConfig()
            config.tool_budget = 999  # Try to mutate
            pytest.fail("SessionConfig must be frozen=True")
        except (dataclasses.FrozenInstanceError, AttributeError):
            pass  # Expected
