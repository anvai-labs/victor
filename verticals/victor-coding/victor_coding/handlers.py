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

"""Coding vertical compute handlers.

Domain-specific handlers for coding workflows:
- code_validation: Syntax and lint checking
- test_runner: Test execution and reporting

Usage:
    from victor_coding import handlers
    handlers.register_handlers()

    # In YAML workflow:
    - id: validate
      type: compute
      handler: code_validation
      inputs:
        files: $ctx.changed_files
        checks: [lint, type]
      output: validation_results
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor_contracts.workflows import (
    ComputeHandlerRegistrar,
    ExecutorNodeStatus,
    NodeResult,
    register_compute_handlers,
)

if TYPE_CHECKING:
    from victor_contracts.verticals.protocols.tools import (
        ToolRegistryProtocol as ToolRegistry,
    )
    from victor_contracts.workflows import ComputeNodeProtocol as ComputeNode
    from victor_contracts.workflows import WorkflowContextProtocol as WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class CodeValidationHandler:
    """Validate code syntax and style.

    Runs linters and type checkers on code files without LLM.

    Example YAML:
        - id: validate_code
          type: compute
          handler: code_validation
          inputs:
            files: $ctx.changed_files
            checks: [syntax, lint, type]
          output: validation_results
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()
        tool_calls = 0

        files = node.input_mapping.get("files", [])
        checks = node.input_mapping.get("checks", ["lint"])

        if isinstance(files, str):
            files = context.get(files) or [files]

        results = {}
        all_passed = True

        for check in checks:
            check_result = await self._run_check(check, files, tool_registry)
            tool_calls += 1
            results[check] = check_result
            if not check_result.get("passed", False):
                all_passed = False

        output = {"checks": results, "all_passed": all_passed}
        output_key = node.output_key or node.id
        context.set(output_key, output)

        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=output,
            duration_seconds=time.time() - start_time,
            tool_calls_used=tool_calls,
        )

    async def _run_check(
        self, check: str, files: List[str], tool_registry: "ToolRegistry"
    ) -> Dict[str, Any]:
        """Run a specific check type."""
        try:
            file_args = " ".join(files) if files else "."

            if check == "lint":
                cmd = f"ruff check {file_args}"
            elif check == "type":
                cmd = f"mypy {file_args}"
            elif check == "syntax":
                cmd = f"python -m py_compile {file_args}"
            elif check == "format":
                cmd = f"ruff format --check {file_args}"
            else:
                return {"passed": True, "message": f"Unknown check: {check}"}

            result = await tool_registry.execute("shell", command=cmd)
            return {
                "passed": result.success,
                "output": result.output,
                "error": result.error,
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}


@dataclass
class TestRunnerHandler:
    """Run tests and collect results.

    Executes test suites and parses results.

    Example YAML:
        - id: run_tests
          type: compute
          handler: test_runner
          inputs:
            test_path: tests/
            framework: pytest
            coverage: true
          output: test_results
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        test_path = node.input_mapping.get("test_path", "tests/")
        framework = node.input_mapping.get("framework", "pytest")
        coverage = node.input_mapping.get("coverage", False)

        if framework == "pytest":
            cmd = f"pytest {test_path} -v"
            if coverage:
                cmd += " --cov --cov-report=json"
        elif framework == "unittest":
            cmd = f"python -m unittest discover {test_path}"
        else:
            cmd = f"{framework} {test_path}"

        try:
            result = await tool_registry.execute("shell", command=cmd)

            output = {
                "framework": framework,
                "passed": result.success,
                "output": result.output,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED
                ),
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


HANDLERS = {
    "code_validation": CodeValidationHandler(),
    "test_runner": TestRunnerHandler(),
}


def register_handlers(
    registrar: Optional[ComputeHandlerRegistrar] = None,
):
    """Register Coding handlers through an explicit host-side registrar."""
    registered = register_compute_handlers(registrar, HANDLERS)
    if registrar is not None:
        for name in HANDLERS:
            logger.debug(f"Registered Coding handler: {name}")
    return registered


__all__ = [
    "CodeValidationHandler",
    "TestRunnerHandler",
    "HANDLERS",
    "register_handlers",
]
