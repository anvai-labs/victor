# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Test that chat_refactored.py has zero architectural violations

import pytest
from pathlib import Path


class TestChatRefactoredCompliance:
    """Verify that chat_refactored.py follows all architectural rules."""

    def test_chat_refactored_has_no_orchestrator_imports(self):
        """chat_refactored.py must NOT import AgentOrchestrator."""
        chat_file = (
            Path(__file__).parent.parent.parent.parent
            / "victor"
            / "ui"
            / "commands"
            / "chat_refactored.py"
        )

        if not chat_file.exists():
            pytest.skip("chat_refactored.py not found")

        with open(chat_file, "r") as f:
            content = f.read()

        # Check for forbidden imports
        assert (
            "from victor.agent.orchestrator import AgentOrchestrator" not in content
        ), "chat_refactored.py must NOT import AgentOrchestrator"
        assert (
            "from victor.framework.shim import FrameworkShim" not in content
        ), "chat_refactored.py must NOT import FrameworkShim"
        assert (
            "AgentFactory(" not in content
        ), "chat_refactored.py must NOT instantiate AgentFactory"

    def test_chat_refactored_uses_framework_client_factory(self):
        """chat_refactored.py MUST use the framework client creation seam."""
        chat_file = (
            Path(__file__).parent.parent.parent.parent
            / "victor"
            / "ui"
            / "commands"
            / "chat_refactored.py"
        )

        if not chat_file.exists():
            pytest.skip("chat_refactored.py not found")

        with open(chat_file, "r") as f:
            content = f.read()

        # Check for proper imports
        assert (
            "from victor.framework.session_runner import create_victor_client" in content
        ), "chat_refactored.py MUST import create_victor_client"
        assert (
            "from victor.framework.session_config import SessionConfig" in content
        ), "chat_refactored.py MUST import SessionConfig"

    def test_chat_refactored_uses_session_config(self):
        """chat_refactored.py MUST use SessionConfig.from_cli_flags()."""
        chat_file = (
            Path(__file__).parent.parent.parent.parent
            / "victor"
            / "ui"
            / "commands"
            / "chat_refactored.py"
        )

        if not chat_file.exists():
            pytest.skip("chat_refactored.py not found")

        with open(chat_file, "r") as f:
            content = f.read()

        # Check for SessionConfig usage
        assert (
            "SessionConfig.from_cli_flags(" in content
        ), "chat_refactored.py MUST use SessionConfig.from_cli_flags()"

    def test_chat_refactored_no_settings_mutation(self):
        """chat_refactored.py MUST NOT mutate settings directly."""
        chat_file = (
            Path(__file__).parent.parent.parent.parent
            / "victor"
            / "ui"
            / "commands"
            / "chat_refactored.py"
        )

        if not chat_file.exists():
            pytest.skip("chat_refactored.py not found")

        with open(chat_file, "r") as f:
            content = f.read()

        # Check for settings mutations (should not exist)
        forbidden_patterns = [
            "settings.tool_settings.tool_output_preview_enabled =",
            "settings.tool_settings.tool_output_pruning_enabled =",
            "settings.smart_routing_enabled =",
            "settings.tool_budget =",
        ]

        for pattern in forbidden_patterns:
            assert (
                pattern not in content
            ), f"chat_refactored.py MUST NOT mutate settings directly (found: {pattern})"
