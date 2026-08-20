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

"""Tests for testing_tool module."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.testing_tool import test, _summarize_report
from victor.tools.subprocess_executor import CommandResult, CommandErrorType


class TestRunTests:
    """Tests for run_tests function."""

    @pytest.mark.asyncio
    async def test_run_tests_success_with_report(self, tmp_path):
        """Test successful test execution with valid report."""
        report_data = {
            "summary": {
                "total": 10,
                "passed": 8,
                "failed": 2,
                "skipped": 0,
            },
            "tests": [
                {
                    "nodeid": "tests/test_example.py::test_success",
                    "outcome": "passed",
                },
                {
                    "nodeid": "tests/test_example.py::test_failure",
                    "outcome": "failed",
                    "call": {
                        "longrepr": "AssertionError: Expected 5 but got 4",
                    },
                },
            ],
        }

        mock_result = CommandResult(
            success=True,
            stdout="Test output",
            stderr="",
            return_code=1,  # Some tests failed
            error_type=CommandErrorType.SUCCESS,
        )

        with (
            patch(
                "victor.tools.testing_tool.run_command_async",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
            patch("pathlib.Path.unlink"),
        ):

            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                report_data
            )

            result = await test()

            assert result["summary"]["total_tests"] == 10
            assert result["summary"]["passed"] == 8
            assert result["summary"]["failed"] == 2
            assert len(result["failures"]) == 1
            assert "test_failure" in result["failures"][0]["test_name"]

    @pytest.mark.asyncio
    async def test_run_tests_with_path(self):
        """Test running tests with specific path."""
        report_data = {
            "summary": {
                "total": 5,
                "passed": 5,
                "failed": 0,
            },
            "tests": [],
        }

        mock_result = CommandResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
        )

        with (
            patch(
                "victor.tools.testing_tool.run_command_async",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_run,
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
            patch("pathlib.Path.unlink"),
        ):

            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                report_data
            )

            result = await test(path="tests/unit")

            # Verify subprocess was called with correct path
            call_args = mock_run.call_args[0][0]
            assert "tests/unit" in call_args
            assert result["summary"]["passed"] == 5

    @pytest.mark.asyncio
    async def test_run_tests_with_pytest_args(self):
        """Test running tests with custom pytest arguments."""
        report_data = {
            "summary": {
                "total": 3,
                "passed": 3,
                "failed": 0,
            },
            "tests": [],
        }

        mock_result = CommandResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
        )

        with (
            patch(
                "victor.tools.testing_tool.run_command_async",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_run,
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
            patch("pathlib.Path.unlink"),
        ):

            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                report_data
            )

            await test(pytest_args=["-v", "-x"])

            # Verify subprocess was called with correct args
            call_args = mock_run.call_args[0][0]
            assert "-v" in call_args
            assert "-x" in call_args

    @pytest.mark.asyncio
    async def test_run_tests_missing_pytest(self):
        """Test handling of missing pytest binary."""
        mock_result = CommandResult(
            success=False,
            stdout="",
            stderr="pytest not found",
            return_code=127,
            error_type=CommandErrorType.NOT_FOUND,
        )

        with patch(
            "victor.tools.testing_tool.run_command_async",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await test()

            assert "error" in result
            assert "pytest is not installed" in result["error"]

    @pytest.mark.asyncio
    async def test_run_tests_timeout(self):
        """Test handling of test execution timeout."""
        mock_result = CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.TIMEOUT,
        )

        with patch(
            "victor.tools.testing_tool.run_command_async",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await test()

            assert "error" in result
            assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_run_tests_missing_report_file(self):
        """Test handling of missing report file."""
        mock_result = CommandResult(
            success=True,
            stdout="Test output",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
        )

        with (
            patch(
                "victor.tools.testing_tool.run_command_async",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):

            result = await test()

            assert "error" in result
            assert "report was not generated" in result["error"]
            assert "stdout" in result
            assert "stderr" in result

    @pytest.mark.asyncio
    async def test_run_tests_invalid_json_report(self):
        """Test handling of invalid JSON in report file."""
        mock_result = CommandResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
        )

        with (
            patch(
                "victor.tools.testing_tool.run_command_async",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
            patch("pathlib.Path.unlink"),
        ):

            mock_open.return_value.__enter__.return_value.read.return_value = "invalid json"

            result = await test()

            assert "error" in result
            assert "unexpected error" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_tests_all_passed(self):
        """Test successful test execution with all tests passing."""
        report_data = {
            "summary": {
                "total": 15,
                "passed": 15,
                "failed": 0,
                "skipped": 0,
            },
            "tests": [
                {
                    "nodeid": f"tests/test_example.py::test_{i}",
                    "outcome": "passed",
                }
                for i in range(15)
            ],
        }

        mock_result = CommandResult(
            success=True,
            stdout="All tests passed",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
        )

        with (
            patch(
                "victor.tools.testing_tool.run_command_async",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
            patch("pathlib.Path.unlink"),
        ):

            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                report_data
            )

            result = await test()

            assert result["summary"]["total_tests"] == 15
            assert result["summary"]["passed"] == 15
            assert result["summary"]["failed"] == 0
            assert len(result["failures"]) == 0

    @pytest.mark.asyncio
    async def test_run_tests_with_skipped(self):
        """Test handling of skipped tests."""
        report_data = {
            "summary": {
                "total": 10,
                "passed": 7,
                "failed": 0,
                "skipped": 3,
            },
            "tests": [],
        }

        mock_result = CommandResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
        )

        with (
            patch(
                "victor.tools.testing_tool.run_command_async",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
            patch("pathlib.Path.unlink"),
        ):

            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
                report_data
            )

            result = await test()

            assert result["summary"]["total_tests"] == 10
            assert result["summary"]["skipped"] == 3


class TestSummarizeReport:
    """Tests for _summarize_report helper function."""

    def test_summarize_report_all_passed(self):
        """Test summarizing report with all tests passing."""
        report = {
            "summary": {
                "total": 5,
                "passed": 5,
                "failed": 0,
            },
            "tests": [],
        }

        result = _summarize_report(report)

        assert result["summary"]["total_tests"] == 5
        assert result["summary"]["passed"] == 5
        assert result["summary"]["failed"] == 0
        assert len(result["failures"]) == 0

    def test_summarize_report_with_failures(self):
        """Test summarizing report with failures."""
        report = {
            "summary": {
                "total": 10,
                "passed": 7,
                "failed": 3,
            },
            "tests": [
                {
                    "nodeid": "tests/test_a.py::test_1",
                    "outcome": "failed",
                    "call": {
                        "longrepr": "AssertionError: Test failed",
                    },
                },
                {
                    "nodeid": "tests/test_b.py::test_2",
                    "outcome": "failed",
                    "call": {
                        "longrepr": "ValueError: Invalid input",
                    },
                },
                {
                    "nodeid": "tests/test_c.py::test_3",
                    "outcome": "passed",
                },
            ],
        }

        result = _summarize_report(report)

        assert result["summary"]["total_tests"] == 10
        assert result["summary"]["failed"] == 3
        assert len(result["failures"]) == 2
        assert "test_1" in result["failures"][0]["test_name"]
        assert "test_2" in result["failures"][1]["test_name"]
        assert "AssertionError" in result["failures"][0]["error_message"]

    def test_summarize_report_failure_without_longrepr(self):
        """Test summarizing report with failure but no longrepr."""
        report = {
            "summary": {
                "total": 3,
                "passed": 2,
                "failed": 1,
            },
            "tests": [
                {
                    "nodeid": "tests/test_a.py::test_1",
                    "outcome": "failed",
                    "call": {},
                },
            ],
        }

        result = _summarize_report(report)

        assert len(result["failures"]) == 1
        # Empty longrepr results in empty string error message
        assert result["failures"][0]["error_message"] == ""

    def test_summarize_report_empty(self):
        """Test summarizing empty report."""
        report = {
            "summary": {},
            "tests": [],
        }

        result = _summarize_report(report)

        assert result["summary"]["total_tests"] == 0
        assert result["summary"]["passed"] == 0
        assert result["summary"]["failed"] == 0
        assert len(result["failures"]) == 0

    def test_summarize_report_with_skipped(self):
        """Test summarizing report with skipped tests."""
        report = {
            "summary": {
                "total": 8,
                "passed": 5,
                "failed": 0,
                "skipped": 3,
            },
            "tests": [],
        }

        result = _summarize_report(report)

        assert result["summary"]["total_tests"] == 8
        assert result["summary"]["skipped"] == 3

    def test_summarize_report_extracts_error_message(self):
        """Test that error message is correctly extracted from longrepr."""
        report = {
            "summary": {
                "total": 1,
                "passed": 0,
                "failed": 1,
            },
            "tests": [
                {
                    "nodeid": "tests/test_example.py::test_division",
                    "outcome": "failed",
                    "call": {
                        "longrepr": "ZeroDivisionError: division by zero\nExtra details here",
                    },
                },
            ],
        }

        result = _summarize_report(report)

        # Implementation extracts the LAST line as error_message
        assert result["failures"][0]["error_message"] == "Extra details here"
        assert "division by zero" in result["failures"][0]["full_error"]

    def test_summarize_report_longrepr_not_string(self):
        """Test handling of non-string longrepr."""
        report = {
            "summary": {
                "total": 1,
                "passed": 0,
                "failed": 1,
            },
            "tests": [
                {
                    "nodeid": "tests/test_example.py::test_with_dict_error",
                    "outcome": "failed",
                    "call": {
                        "longrepr": {"error": "Some error", "line": 42},
                    },
                },
            ],
        }

        result = _summarize_report(report)

        assert len(result["failures"]) == 1
        assert result["failures"][0]["error_message"] == "Error representation was not a string."
        assert result["failures"][0]["full_error"] == {
            "error": "Some error",
            "line": 42,
        }
