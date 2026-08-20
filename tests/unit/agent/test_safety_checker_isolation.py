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

"""Global safety-checker singleton isolation.

Regression net for an order-dependent CI failure: the global ``_default_checker``
singleton (``get_safety_checker()``) accumulates mutable state and is never reset,
so a test that left it *denying* made a later, unrelated ``edit`` execution return
``success=False`` "Operation cancelled by safety check" — only in the full suite,
never in isolation. ``reset_safety_checker()`` + the autouse conftest fixture fix it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import victor.agent.safety as safety
from victor.agent.safety import get_safety_checker, reset_safety_checker
from victor.agent.tool_executor import ToolExecutor, ValidationMode
from victor.tools.base import AccessMode, BaseTool, ToolValidationResult
from victor.tools.registry import ToolRegistry


def _edit_executor() -> ToolExecutor:
    registry = ToolRegistry()
    tool = MagicMock(spec=BaseTool)
    tool.name = "edit"
    tool.access_mode = AccessMode.WRITE
    tool.parameters = {
        "type": "object",
        "properties": {"ops": {"type": "array"}, "preview": {"type": "boolean"}},
        "required": ["ops"],
        "additionalProperties": False,
    }
    tool.validate_parameters_detailed = MagicMock(return_value=ToolValidationResult.success())
    tool.execute = AsyncMock(return_value={"success": True})
    registry.register(tool)
    return ToolExecutor(tool_registry=registry, validation_mode=ValidationMode.STRICT)


_EDIT_ARGS = {"value": {"ops": [{"type": "create", "path": "test.txt", "content": "hi"}]}}


class _DenyChecker:
    """Stand-in for a globally-polluted, denying safety checker."""

    async def check_and_confirm(self, *args, **kwargs):
        return (False, "denied by prior-test pollution")


def test_reset_rebuilds_singleton():
    first = get_safety_checker()
    reset_safety_checker()
    assert safety._default_checker is None
    second = get_safety_checker()
    assert second is not first


class TestSafetyCheckerDoesNotLeakIntoExecutor:
    @pytest.mark.asyncio
    async def test_polluted_global_checker_blocks_edit(self):
        # Documents the failure mode: a denying global checker blocks the edit.
        safety._default_checker = _DenyChecker()
        result = await _edit_executor().execute("edit", _EDIT_ARGS)
        assert result.success is False
        assert result.error  # blocked with the checker's rejection reason

    @pytest.mark.asyncio
    async def test_reset_restores_execution(self):
        # The fix: after reset, a fresh (default, non-denying) checker lets it run.
        safety._default_checker = _DenyChecker()
        reset_safety_checker()
        result = await _edit_executor().execute("edit", _EDIT_ARGS)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_autouse_fixture_gives_each_test_a_clean_checker(self):
        # The autouse `reset_safety_checker_singleton` fixture ran before this test,
        # so even if an earlier test polluted the global, edit succeeds here.
        result = await _edit_executor().execute("edit", _EDIT_ARGS)
        assert result.success is True
