"""Tests for HITL tool approval gate."""

from unittest.mock import AsyncMock
import pytest


class TestApprovalMode:
    def test_enum_exists(self):
        from victor.agent.tool_approval import ApprovalMode

        assert ApprovalMode.AUTO == "auto"
        assert ApprovalMode.DANGEROUS == "dangerous"
        assert ApprovalMode.ALL == "all"

    def test_auto_mode_approves_all(self):
        from victor.agent.tool_approval import ToolApprovalGate, ApprovalMode

        gate = ToolApprovalGate(mode=ApprovalMode.AUTO)
        assert gate.needs_approval("shell", "high") is False
        assert gate.needs_approval("read", "safe") is False

    def test_dangerous_mode_blocks_high(self):
        from victor.agent.tool_approval import ToolApprovalGate, ApprovalMode

        gate = ToolApprovalGate(mode=ApprovalMode.DANGEROUS)
        assert gate.needs_approval("shell", "high") is True
        assert gate.needs_approval("edit", "medium") is True

    def test_dangerous_mode_allows_safe(self):
        from victor.agent.tool_approval import ToolApprovalGate, ApprovalMode

        gate = ToolApprovalGate(mode=ApprovalMode.DANGEROUS)
        assert gate.needs_approval("read", "safe") is False
        assert gate.needs_approval("ls", "safe") is False

    def test_all_mode_blocks_everything(self):
        from victor.agent.tool_approval import ToolApprovalGate, ApprovalMode

        gate = ToolApprovalGate(mode=ApprovalMode.ALL)
        assert gate.needs_approval("read", "safe") is True
        assert gate.needs_approval("shell", "high") is True


class TestApprovalGate:
    @pytest.mark.asyncio
    async def test_auto_approved_for_safe_tool(self):
        from victor.agent.tool_approval import ToolApprovalGate, ApprovalMode

        gate = ToolApprovalGate(mode=ApprovalMode.DANGEROUS)
        decision = await gate.request_approval("read", {}, "safe")
        assert decision.approved is True
        assert "auto" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_handler_called_for_dangerous_tool(self):
        from victor.agent.tool_approval import (
            ToolApprovalGate,
            ApprovalMode,
            ApprovalDecision,
        )

        handler = AsyncMock(return_value=ApprovalDecision(approved=True, reason="user approved"))
        gate = ToolApprovalGate(mode=ApprovalMode.DANGEROUS, approval_handler=handler)
        decision = await gate.request_approval("shell", {"command": "rm -rf"}, "high")
        handler.assert_called_once()
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_no_handler_auto_approves_with_warning(self):
        from victor.agent.tool_approval import ToolApprovalGate, ApprovalMode

        gate = ToolApprovalGate(mode=ApprovalMode.DANGEROUS, approval_handler=None)
        decision = await gate.request_approval("shell", {}, "high")
        assert decision.approved is True
        assert "no handler" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_rejection(self):
        from victor.agent.tool_approval import (
            ToolApprovalGate,
            ApprovalMode,
            ApprovalDecision,
        )

        handler = AsyncMock(return_value=ApprovalDecision(approved=False, reason="too dangerous"))
        gate = ToolApprovalGate(mode=ApprovalMode.ALL, approval_handler=handler)
        decision = await gate.request_approval("shell", {"cmd": "rm -rf /"}, "critical")
        assert decision.approved is False
        assert "dangerous" in decision.reason


class TestApprovalDecision:
    def test_dataclass_fields(self):
        from victor.agent.tool_approval import ApprovalDecision

        d = ApprovalDecision(approved=True, reason="ok")
        assert d.approved is True
        assert d.reason == "ok"
        assert d.modified_args is None


class TestSettingIntegration:
    def test_setting_exists(self):
        from victor.config.settings import ContextSettings

        settings = ContextSettings()
        assert hasattr(settings, "tool_approval_mode")

    def test_default_is_auto(self):
        from victor.config.settings import ContextSettings

        settings = ContextSettings()
        assert settings.tool_approval_mode == "auto"
