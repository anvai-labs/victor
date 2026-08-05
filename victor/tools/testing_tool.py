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

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.subprocess_executor import run_command_async, CommandErrorType
from victor.tools.formatters import format_test_results


@tool(
    category="testing",
    priority=Priority.MEDIUM,  # Task-specific testing tool
    access_mode=AccessMode.EXECUTE,  # Runs pytest subprocess
    danger_level=DangerLevel.SAFE,  # No file modifications
    keywords=["test", "pytest", "unittest", "tests", "testing", "check", "validate"],
    mandatory_keywords=["run tests", "run test", "execute tests"],  # Force inclusion
    task_types=["testing", "verification"],  # Classification-aware selection
    stages=["verification"],  # Conversation stages where relevant
)
async def test(
    path: Optional[str] = None, pytest_args: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Runs tests using pytest and returns a structured summary of the results.

    NOTE: For simple test runs, PREFER the ``shell`` tool:
        shell(cmd="pytest -q")
        shell(cmd="pytest tests/unit -v", cwd=".")
        shell(cmd="pytest -k test_foo --tb=short")

    This tool wraps pytest via subprocess. Use it when you want structured
    JSON test output (pass/fail summary). For quick ad-hoc test runs,
    ``shell(cmd="pytest ...")`` is more efficient.

    This tool runs pytest on the specified path and captures the output in JSON format,
    providing a clean summary of test outcomes.

    Args:
        path: The specific file or directory to run tests on. If not provided,
              pytest will run on the entire project based on its configuration.
        pytest_args: A list of additional command-line arguments to pass to pytest
                     (e.g., ["-k", "my_test_name", "-v"]).

    Returns:
        A dictionary summarizing the test results, including counts of passed,
        failed, and skipped tests, and detailed error reports for failures.
    """
    report_file = Path(".pytest_report.json")
    command = ["pytest", f"--json-report-file={report_file}"]

    if path:
        command.append(path)

    if pytest_args:
        command.extend(pytest_args)

    try:
        # Build command string for async execution
        cmd_str = " ".join(command)
        result = await run_command_async(
            cmd_str,
            timeout=300,  # 5-minute timeout for tests
            check_dangerous=False,
        )

        # Handle timeout
        if result.error_type == CommandErrorType.TIMEOUT:
            return {"error": "Test execution timed out after 5 minutes."}

        # Handle command not found
        if result.error_type == CommandErrorType.NOT_FOUND:
            return {"error": "pytest is not installed or not in the system's PATH."}

        if not report_file.exists():
            return {
                "error": "pytest execution failed or report was not generated.",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        with open(report_file, "r") as f:
            report = json.load(f)

        # Clean up the report file
        report_file.unlink()

        return _summarize_report(report)

    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


def _summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Parses the JSON report from pytest and creates a concise summary."""
    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed_count = summary.get("failed", 0)

    failures = []
    if failed_count > 0:
        for test in report.get("tests", []):
            if test.get("outcome") == "failed":
                call = test.get("call", {})
                long_repr = call.get("longrepr", "")
                # Ensure long_repr is a string before splitting
                if isinstance(long_repr, str):
                    error_lines = long_repr.split("\n")
                    error_message = error_lines[-1] if error_lines else "No error message captured."
                else:
                    error_message = "Error representation was not a string."

                failures.append(
                    {
                        "test_name": test.get("nodeid"),
                        "error_message": error_message,
                        "full_error": long_repr,
                    }
                )

    # Build summary dictionary
    result_summary = {
        "total_tests": total,
        "passed": passed,
        "failed": failed_count,
        "skipped": summary.get("skipped", 0),
        "xfailed": summary.get("xfailed", 0),
        "xpassed": summary.get("xpassed", 0),
    }

    # Use unified formatter system
    formatted_result = format_test_results(
        {
            "summary": result_summary,
            "failures": failures,
        }
    )

    return {
        "summary": result_summary,
        "failures": failures,
        "formatted_summary": formatted_result.content,  # Rich-formatted for console
        "contains_markup": formatted_result.contains_markup,
    }
