# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for WorkflowExecutionResult error typing — Wave S."""

import pytest

from victor.framework.workflow_engine import (
    WorkflowExecutionResult,
    WorkflowErrorType,
)


class TestWorkflowErrorType:
    """WorkflowErrorType enum provides typed error categories for workflow execution."""

    def test_workflow_error_type_enum_values(self):
        """WorkflowErrorType should have expected error types."""
        assert WorkflowErrorType.VALIDATION.value == "validation"
        assert WorkflowErrorType.EXECUTION.value == "execution"
        assert WorkflowErrorType.TIMEOUT.value == "timeout"
        assert WorkflowErrorType.CANCELLED.value == "cancelled"
        assert WorkflowErrorType.RESOURCE.value == "resource"


class TestWorkflowExecutionResultTyping:
    """WorkflowExecutionResult should support typed error fields."""

    def test_result_with_error_type(self):
        """WorkflowExecutionResult should accept error_type field."""
        result = WorkflowExecutionResult(
            success=False,
            error_type=WorkflowErrorType.VALIDATION,
            error_message="Invalid workflow definition",
        )

        assert result.success is False
        assert result.error_type == WorkflowErrorType.VALIDATION
        assert result.error_message == "Invalid workflow definition"

    def test_result_backward_compat(self):
        """WorkflowExecutionResult should maintain backward compatibility."""
        # Old-style error field should still work
        result = WorkflowExecutionResult(success=False, error="Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"
        # New fields should be None by default
        assert result.error_type is None
        assert result.error_message is None

    def test_error_type_enum_values(self):
        """WorkflowErrorType enum should have all expected values."""
        assert hasattr(WorkflowErrorType, "VALIDATION")
        assert hasattr(WorkflowErrorType, "EXECUTION")
        assert hasattr(WorkflowErrorType, "TIMEOUT")
        assert hasattr(WorkflowErrorType, "CANCELLED")
        assert hasattr(WorkflowErrorType, "RESOURCE")
