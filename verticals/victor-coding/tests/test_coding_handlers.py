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

"""Pre-move characterization for victor-coding compute handlers (dedup F5).

Pins the observable contract before the envelope is extracted to the shared
kernel: command construction, output-key handling, the two handlers' DIFFERENT
failure policies (code_validation always COMPLETED with all_passed inside the
output; test_runner FAILED when the suite fails), and registration semantics.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor_contracts.testing import MockComputeNode, MockToolResult, MockWorkflowContext
from victor_coding.handlers import HANDLERS, CodeValidationHandler, TestRunnerHandler


def registry(result: MockToolResult) -> MagicMock:
    r = MagicMock()
    r.execute = AsyncMock(return_value=result)
    return r


class TestCodeValidationHandler:
    @pytest.mark.asyncio
    async def test_lint_check_passes_through_shell_command(self):
        reg = registry(MockToolResult(success=True, output="clean"))
        ctx = MockWorkflowContext()
        node = MockComputeNode(
            node_id="v1",
            input_mapping={"files": ["a.py"], "checks": ["lint"]},
            output_key="validation",
        )
        result = await CodeValidationHandler()(node, ctx, reg)
        cmd = reg.execute.call_args.kwargs["command"]
        assert cmd == "ruff check a.py"
        assert result.status.value == "completed"
        assert ctx.get("validation")["all_passed"] is True
        assert ctx.get("validation")["checks"]["lint"]["passed"] is True

    @pytest.mark.asyncio
    async def test_failed_check_still_completes_with_all_passed_false(self):
        """code_validation's fail policy: COMPLETED status, failure in output."""
        reg = registry(MockToolResult(success=False, error="E501"))
        ctx = MockWorkflowContext()
        node = MockComputeNode(node_id="v2", input_mapping={"files": ["b.py"], "checks": ["lint"]})
        result = await CodeValidationHandler()(node, ctx, reg)
        assert result.status.value == "completed"
        assert ctx.get("v2")["all_passed"] is False

    @pytest.mark.asyncio
    async def test_unknown_check_passes_with_message(self):
        reg = registry(MockToolResult(success=True))
        node = MockComputeNode(
            node_id="v3", input_mapping={"files": ["x.py"], "checks": ["nonexistent"]}
        )
        ctx = MockWorkflowContext()
        result = await CodeValidationHandler()(node, ctx, reg)
        assert result.status.value == "completed"
        assert ctx.get("v3")["checks"]["nonexistent"]["passed"] is True

    @pytest.mark.asyncio
    async def test_exception_is_contained_per_check(self):
        """Actual behavior: _run_check catches exceptions and reports them as
        failed checks — the node still COMPLETES (no outer try/except)."""
        reg = MagicMock()
        reg.execute = AsyncMock(side_effect=RuntimeError("boom"))
        node = MockComputeNode(node_id="v4", input_mapping={"files": ["c.py"]})
        ctx = MockWorkflowContext()
        result = await CodeValidationHandler()(node, ctx, reg)
        assert result.status.value == "completed"
        assert ctx.get("v4")["all_passed"] is False
        assert "boom" in ctx.get("v4")["checks"]["lint"]["error"]


class TestTestRunnerHandler:
    @pytest.mark.asyncio
    async def test_pytest_command_with_coverage(self):
        reg = registry(MockToolResult(success=True, output="3 passed"))
        node = MockComputeNode(
            node_id="t1",
            input_mapping={"test_path": "tests/", "framework": "pytest", "coverage": True},
            output_key="test_results",
        )
        ctx = MockWorkflowContext()
        result = await TestRunnerHandler()(node, ctx, reg)
        cmd = reg.execute.call_args.kwargs["command"]
        assert cmd == "pytest tests/ -v --cov --cov-report=json"
        assert result.status.value == "completed"
        assert ctx.get("test_results")["passed"] is True

    @pytest.mark.asyncio
    async def test_failed_suite_fails_the_node(self):
        """test_runner's fail policy: FAILED status on suite failure (unlike
        code_validation — the kernel must parameterize this)."""
        reg = registry(MockToolResult(success=False, output="1 failed"))
        node = MockComputeNode(node_id="t2", input_mapping={"framework": "pytest"})
        ctx = MockWorkflowContext()
        result = await TestRunnerHandler()(node, ctx, reg)
        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_unittest_framework_command(self):
        reg = registry(MockToolResult(success=True))
        node = MockComputeNode(node_id="t3", input_mapping={"framework": "unittest"})
        ctx = MockWorkflowContext()
        await TestRunnerHandler()(node, ctx, reg)
        cmd = reg.execute.call_args.kwargs["command"]
        assert cmd.startswith("python -m unittest discover")


class TestRegistration:
    def test_handlers_table_and_registrar_signature(self):
        assert set(HANDLERS) == {"code_validation", "test_runner"}
        # The UI introspects this signature; the kernel must keep it.
        import inspect

        from victor_coding.handlers import register_handlers

        params = inspect.signature(register_handlers).parameters
        assert "registrar" in params

    def test_register_via_explicit_registrar(self):
        entries: Dict[str, Any] = {}

        class Registrar:
            def __call__(self, name: str, handler: Any) -> None:
                entries[name] = handler

        from victor_coding.handlers import register_handlers

        register_handlers(Registrar())
        assert set(entries) == set(HANDLERS)
