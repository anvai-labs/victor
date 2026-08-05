# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for StepHandlerStatus enum and StepExceptionDetails TypedDict — Wave K."""

import pytest

from victor.framework.step_handlers import (
    StepHandlerStatus,
    StepExceptionDetails,
    BaseStepHandler,
)


class TestStepHandlerStatus:
    """StepHandlerStatus enum provides typed status values."""

    def test_enum_values(self):
        """StepHandlerStatus should have SUCCESS, WARNING, ERROR values."""
        assert StepHandlerStatus.SUCCESS.value == "success"
        assert StepHandlerStatus.WARNING.value == "warning"
        assert StepHandlerStatus.ERROR.value == "error"

    def test_enum_members(self):
        """All three required status enum members should exist."""
        assert hasattr(StepHandlerStatus, "SUCCESS")
        assert hasattr(StepHandlerStatus, "WARNING")
        assert hasattr(StepHandlerStatus, "ERROR")


class TestStepExceptionDetails:
    """StepExceptionDetails TypedDict provides typed error detail structure."""

    def test_typed_dict_structure(self):
        """StepExceptionDetails should have required error_type and error_message fields."""
        details: StepExceptionDetails = {
            "error_type": "ValueError",
            "error_message": "test error",
            "step_name": "TestStep",
        }
        assert details["error_type"] == "ValueError"
        assert details["error_message"] == "test error"
        assert details["step_name"] == "TestStep"

    def test_optional_fields(self):
        """error_traceback should be optional (total=False)."""
        details: StepExceptionDetails = {
            "error_type": "RuntimeError",
            "error_message": "test",
        }
        # Should not raise KeyError for missing optional fields
        assert "error_type" in details


class TestStepHandlerExecution:
    """Step handlers use typed status and exception details."""

    def test_success_status_by_default(self, mocker):
        """Step should succeed with status=success when no exception raised."""
        from victor.framework.step_handlers import ToolStepHandler

        handler = ToolStepHandler()

        # Mock _do_apply to succeed
        handler._do_apply = mocker.MagicMock()
        handler._get_step_details = mocker.MagicMock(return_value={"test": "details"})

        result = mocker.MagicMock()
        result.record_step_status = mocker.MagicMock()

        handler.apply(None, None, None, result, strict_mode=False)

        # Verify status was recorded as "success"
        result.record_step_status.assert_called_once()
        call_args = result.record_step_status.call_args
        assert call_args[0][1] == "success"  # status argument

    def test_strict_mode_returns_error_status(self, mocker):
        """Step should return ERROR status in strict_mode when exception occurs."""
        from victor.framework.step_handlers import ToolStepHandler

        handler = ToolStepHandler()

        # Mock _do_apply to raise exception
        test_error = ValueError("test failure")
        handler._do_apply = mocker.MagicMock(side_effect=test_error)

        result = mocker.MagicMock()
        result.record_step_status = mocker.MagicMock()

        handler.apply(None, None, None, result, strict_mode=True)

        # Verify error was added and status was ERROR
        result.add_error.assert_called_once()
        call_args = result.record_step_status.call_args
        assert call_args[0][1] == "error"  # status argument

    def test_non_strict_mode_returns_warning_status(self, mocker):
        """Step should return WARNING status in non-strict_mode when exception occurs."""
        from victor.framework.step_handlers import ToolStepHandler

        handler = ToolStepHandler()

        # Mock _do_apply to raise exception
        test_error = RuntimeError("test warning")
        handler._do_apply = mocker.MagicMock(side_effect=test_error)

        result = mocker.MagicMock()
        result.record_step_status = mocker.MagicMock()

        handler.apply(None, None, None, result, strict_mode=False)

        # Verify warning was added and status was WARNING
        result.add_warning.assert_called_once()
        call_args = result.record_step_status.call_args
        assert call_args[0][1] == "warning"  # status argument

    def test_exception_details_include_type_and_message(self, mocker):
        """Exception details should include error_type and error_message."""
        from victor.framework.step_handlers import ToolStepHandler

        handler = ToolStepHandler()

        # Mock _do_apply to raise a specific exception
        test_error = ValueError("detailed error message")
        handler._do_apply = mocker.MagicMock(side_effect=test_error)

        result = mocker.MagicMock()
        result.record_step_status = mocker.MagicMock()

        handler.apply(None, None, None, result, strict_mode=False)

        # Verify details dict has error_type and error_message
        call_args = result.record_step_status.call_args
        details = call_args[1]["details"]
        assert isinstance(details, dict)
        assert "error_type" in details
        assert details["error_type"] == "ValueError"
        assert "error_message" in details
        assert details["error_message"] == "detailed error message"
        assert details["step_name"] == "tools"

    def test_exception_details_traceback_optional(self, mocker):
        """error_traceback should be optional in exception details."""
        from victor.framework.step_handlers import ToolStepHandler

        handler = ToolStepHandler()

        # Mock _do_apply to raise exception
        handler._do_apply = mocker.MagicMock(side_effect=RuntimeError("test"))

        result = mocker.MagicMock()
        result.record_step_status = mocker.MagicMock()

        handler.apply(None, None, None, result, strict_mode=False)

        # Should not raise error if error_traceback is missing
        call_args = result.record_step_status.call_args
        details = call_args[1]["details"]
        # error_traceback is optional, so it may or may not be present
        # The test just verifies the details dict is well-formed
        assert isinstance(details, dict)
