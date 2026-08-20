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

"""Tests for the safety module."""

import pytest
from victor.agent.safety import (
    ConfirmationRequest,
    OperationalRiskLevel,
    SafetyChecker,
    _RISK_ORDER,
    get_safety_checker,
    set_confirmation_callback,
)


class TestOperationalRiskLevel:
    """Tests for OperationalRiskLevel enum and ordering."""

    def test_risk_order_completeness(self):
        """All OperationalRiskLevel values should be in _RISK_ORDER."""
        for level in OperationalRiskLevel:
            assert level in _RISK_ORDER

    def test_risk_order_values(self):
        """Risk levels should be ordered correctly."""
        assert _RISK_ORDER[OperationalRiskLevel.SAFE] < _RISK_ORDER[OperationalRiskLevel.LOW]
        assert _RISK_ORDER[OperationalRiskLevel.LOW] < _RISK_ORDER[OperationalRiskLevel.MEDIUM]
        assert _RISK_ORDER[OperationalRiskLevel.MEDIUM] < _RISK_ORDER[OperationalRiskLevel.HIGH]
        assert _RISK_ORDER[OperationalRiskLevel.HIGH] < _RISK_ORDER[OperationalRiskLevel.CRITICAL]


class TestSafetyChecker:
    """Tests for SafetyChecker class."""

    def test_check_bash_safe_commands(self):
        """Safe commands should return SAFE risk level."""
        checker = SafetyChecker()
        safe_commands = [
            "ls -la",
            "pwd",
            "echo hello",
            "cat file.txt",
            "grep pattern file.txt",
            "find . -name '*.py'",
        ]
        for cmd in safe_commands:
            level, details = checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.SAFE, f"'{cmd}' should be SAFE but got {level}"
            assert details == []

    def test_check_bash_medium_risk(self):
        """Medium risk commands should be detected."""
        checker = SafetyChecker()
        medium_commands = [
            ("rm file.txt", "Delete file(s)"),
            ("> output.txt", "Overwrite file with redirection"),
            ("pip uninstall package", "Uninstall Python package"),
        ]
        for cmd, expected_detail in medium_commands:
            level, details = checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.MEDIUM, f"'{cmd}' should be MEDIUM but got {level}"
            assert any(
                expected_detail in d for d in details
            ), f"Expected '{expected_detail}' in {details}"

    def test_check_bash_high_risk(self):
        """High risk commands should be detected."""
        checker = SafetyChecker()
        high_commands = [
            ("rm -rf /tmp/test", "Recursively delete"),
            ("git reset --hard", "Discard all uncommitted changes"),
            ("git push --force", "Force push"),
            ("sudo apt install", "elevated privileges"),
        ]
        for cmd, expected_detail in high_commands:
            level, details = checker.check_bash_command(cmd)
            assert level == OperationalRiskLevel.HIGH, f"'{cmd}' should be HIGH but got {level}"
            assert any(
                expected_detail.lower() in d.lower() for d in details
            ), f"Expected '{expected_detail}' in {details}"

    def test_check_bash_critical_risk(self):
        """Critical risk commands should be detected."""
        checker = SafetyChecker()
        critical_commands = [
            ("rm -rf /", "Delete entire filesystem"),
            ("rm -rf /*", "Delete all root-level directories"),
            ("dd if=/dev/zero of=/dev/sda", "Write directly to disk device"),
        ]
        for cmd, expected_detail in critical_commands:
            level, details = checker.check_bash_command(cmd)
            assert (
                level == OperationalRiskLevel.CRITICAL
            ), f"'{cmd}' should be CRITICAL but got {level}"
            assert any(
                expected_detail.lower() in d.lower() for d in details
            ), f"Expected '{expected_detail}' in {details}"

    def test_check_file_operation_sensitive_files(self):
        """Sensitive file operations should be HIGH risk."""
        checker = SafetyChecker()
        sensitive_files = [
            "/path/to/secrets.env",
            "/path/to/private.key",
            "/path/to/cert.pem",
            "/path/to/database.db",
        ]
        for file_path in sensitive_files:
            level, details = checker.check_file_operation("write", file_path)
            assert (
                level == OperationalRiskLevel.HIGH
            ), f"'{file_path}' should be HIGH risk but got {level}"

    def test_check_file_operation_delete(self):
        """Delete operations should be HIGH risk."""
        checker = SafetyChecker()
        level, details = checker.check_file_operation("delete", "/path/to/file.txt")
        assert level == OperationalRiskLevel.HIGH
        assert any("deleting" in d.lower() for d in details)

    def test_check_file_operation_system_paths(self):
        """System path operations should be HIGH risk."""
        checker = SafetyChecker()
        level, details = checker.check_file_operation("write", "/etc/config")
        assert level == OperationalRiskLevel.HIGH
        assert any("system" in d.lower() for d in details)

    def test_is_write_tool_recognizes_canonical_names(self):
        """Canonical runtime tool names should be treated as write tools."""
        checker = SafetyChecker()
        assert checker.is_write_tool("write") is True
        assert checker.is_write_tool("edit") is True
        assert checker.is_write_tool("shell") is True

    @pytest.mark.asyncio
    async def test_check_and_confirm_safe_operation(self):
        """Safe operations should auto-approve without callback."""
        checker = SafetyChecker()
        should_proceed, reason = await checker.check_and_confirm(
            "read_file", {"path": "/some/file.txt"}
        )
        assert should_proceed is True
        assert reason is None

    @pytest.mark.asyncio
    async def test_check_and_confirm_dangerous_bash(self):
        """Dangerous bash commands should trigger confirmation."""
        confirmed = False

        async def mock_callback(request):
            nonlocal confirmed
            confirmed = True
            assert request.risk_level == OperationalRiskLevel.HIGH
            return True  # User confirms

        checker = SafetyChecker(
            confirmation_callback=mock_callback,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )
        assert confirmed is True
        assert should_proceed is True

    @pytest.mark.asyncio
    async def test_check_and_confirm_canonical_shell_uses_cmd(self):
        """Canonical shell tool should read the cmd argument."""
        confirmed = False

        async def mock_callback(request):
            nonlocal confirmed
            confirmed = True
            assert request.tool_name == "shell"
            assert request.arguments["cmd"] == "rm -rf /tmp/test"
            assert request.risk_level == OperationalRiskLevel.HIGH
            return True

        checker = SafetyChecker(
            confirmation_callback=mock_callback,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        should_proceed, reason = await checker.check_and_confirm(
            "shell", {"cmd": "rm -rf /tmp/test"}
        )
        assert confirmed is True
        assert should_proceed is True

    @pytest.mark.asyncio
    async def test_check_and_confirm_user_cancels(self):
        """User cancellation should block operation."""

        async def mock_callback(request):
            return False  # User cancels

        checker = SafetyChecker(
            confirmation_callback=mock_callback,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )
        assert should_proceed is False
        assert "cancelled" in reason.lower()

    @pytest.mark.asyncio
    async def test_check_and_confirm_no_callback_allows_with_warning(self):
        """Without callback, operations proceed with warning."""
        checker = SafetyChecker(
            confirmation_callback=None,
            require_confirmation_threshold=OperationalRiskLevel.HIGH,
        )
        should_proceed, reason = await checker.check_and_confirm(
            "execute_bash", {"command": "rm -rf /tmp/test"}
        )
        # Should proceed even without callback (with logged warning)
        assert should_proceed is True


class TestConfirmationRequest:
    """Tests for ConfirmationRequest dataclass."""

    def test_format_message(self):
        """format_message should produce readable output."""
        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=OperationalRiskLevel.HIGH,
            description="Delete files",
            details=["Recursively delete files/directories"],
            arguments={"command": "rm -rf /tmp/test"},
        )
        message = request.format_message()
        assert "HIGH RISK" in message
        assert "execute_bash" in message
        assert "Delete files" in message
        assert "Recursively delete" in message


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_safety_checker_singleton(self):
        """get_safety_checker should return same instance."""
        checker1 = get_safety_checker()
        checker2 = get_safety_checker()
        assert checker1 is checker2

    def test_set_confirmation_callback(self):
        """set_confirmation_callback should update global checker."""

        async def test_callback(request):
            return True

        set_confirmation_callback(test_callback)
        checker = get_safety_checker()
        assert checker.confirmation_callback == test_callback
