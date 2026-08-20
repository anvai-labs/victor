# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for ToolDependencyValidator."""

from dataclasses import dataclass
import logging
from typing import Any, List, Optional, Set
from unittest.mock import MagicMock

import pytest

from victor.workflows.validation import (
    ToolDependencyValidator,
    ToolValidationError,
    ToolValidationResult,
    validate_workflow_tools,
)


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self, tools: List[str]):
        self._tools = {name: MagicMock() for name in tools}

    def get(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    def list_tools(self, only_enabled: bool = True) -> List[Any]:
        tools = []
        for name, mock in self._tools.items():
            mock.name = name
            tools.append(mock)
        return tools


@dataclass
class MockNode:
    """Mock workflow node for testing."""

    id: str
    allowed_tools: Optional[Set[str]] = None
    tools: Optional[Set[str]] = None


@dataclass
class MockWorkflow:
    """Mock workflow definition for testing."""

    nodes: dict


class TestToolValidationError:
    """Test ToolValidationError dataclass."""

    def test_creation(self):
        """Should create error with all fields."""
        error = ToolValidationError(
            node_id="node_1",
            tool_name="missing_tool",
            error_type="missing",
            message="Tool not found",
        )
        assert error.node_id == "node_1"
        assert error.tool_name == "missing_tool"
        assert error.error_type == "missing"
        assert error.severity == "error"

    def test_str_format(self):
        """Should format as readable string."""
        error = ToolValidationError(
            node_id="node_1",
            tool_name="tool",
            error_type="missing",
            message="Test message",
            severity="error",
        )
        result = str(error)
        assert "ERROR" in result
        assert "node_1" in result

    def test_warning_severity(self):
        """Should support warning severity."""
        error = ToolValidationError(
            node_id="node_1",
            tool_name="tool",
            error_type="missing",
            message="Test",
            severity="warning",
        )
        assert error.severity == "warning"


class TestToolValidationResult:
    """Test ToolValidationResult dataclass."""

    def test_initial_state(self):
        """Should start valid with empty collections."""
        result = ToolValidationResult()
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_marks_invalid(self):
        """Adding error should mark result as invalid."""
        result = ToolValidationResult()
        error = ToolValidationError(
            node_id="n1",
            tool_name="t1",
            error_type="missing",
            message="test",
            severity="error",
        )
        result.add_error(error)
        assert result.valid is False
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Adding warning should keep result valid."""
        result = ToolValidationResult()
        warning = ToolValidationError(
            node_id="n1",
            tool_name="t1",
            error_type="missing",
            message="test",
            severity="warning",
        )
        result.add_error(warning)
        assert result.valid is True
        assert len(result.warnings) == 1

    def test_summary_valid(self):
        """Summary should indicate valid state."""
        result = ToolValidationResult()
        result.validated_tools.add("tool1")
        result.validated_tools.add("tool2")
        summary = result.summary()
        assert "Valid" in summary
        assert "2" in summary

    def test_summary_invalid(self):
        """Summary should indicate invalid state."""
        result = ToolValidationResult()
        result.add_error(ToolValidationError("n1", "t1", "missing", "test", "error"))
        result.missing_tools.add("t1")
        summary = result.summary()
        assert "Invalid" in summary

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = ToolValidationResult()
        result.validated_tools.add("tool1")
        result.missing_tools.add("tool2")
        d = result.to_dict()
        assert d["valid"] is True
        assert "tool1" in d["validated_tools"]
        assert "tool2" in d["missing_tools"]


class TestToolDependencyValidator:
    """Test ToolDependencyValidator class."""

    def test_init_without_registry(self):
        """Should initialize without registry."""
        validator = ToolDependencyValidator()
        assert validator.tool_registry is None

    def test_init_with_registry(self):
        """Should initialize with registry."""
        registry = MockToolRegistry(["tool1", "tool2"])
        validator = ToolDependencyValidator(registry)
        assert validator.tool_registry is registry

    def test_available_tools_cached(self):
        """Available tools should be cached."""
        registry = MockToolRegistry(["tool1", "tool2"])
        validator = ToolDependencyValidator(registry)
        tools1 = validator.available_tools
        tools2 = validator.available_tools
        assert tools1 is tools2

    def test_set_registry_clears_cache(self):
        """Setting registry should clear cache."""
        registry1 = MockToolRegistry(["tool1"])
        registry2 = MockToolRegistry(["tool2"])
        validator = ToolDependencyValidator(registry1)
        _ = validator.available_tools
        validator.set_registry(registry2)
        assert validator._available_tools is None

    def test_validate_empty_workflow(self):
        """Should validate workflow with no nodes."""
        registry = MockToolRegistry(["tool1"])
        validator = ToolDependencyValidator(registry)
        workflow = MockWorkflow(nodes={})
        result = validator.validate(workflow)
        assert result.valid is True

    def test_validate_all_tools_present(self):
        """Should validate when all tools exist."""
        registry = MockToolRegistry(["tool1", "tool2"])
        validator = ToolDependencyValidator(registry)
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"tool1", "tool2"}),
            }
        )
        result = validator.validate(workflow)
        assert result.valid is True
        assert "tool1" in result.validated_tools
        assert "tool2" in result.validated_tools

    def test_validate_missing_tool_strict(self):
        """Should error on missing tool in strict mode."""
        registry = MockToolRegistry(["tool1"])
        validator = ToolDependencyValidator(registry, strict=True)
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"tool1", "missing_tool"}),
            }
        )
        result = validator.validate(workflow)
        assert result.valid is False
        assert "missing_tool" in result.missing_tools
        assert len(result.errors) == 1

    def test_validate_missing_tool_lenient(self):
        """Should warn on missing tool in lenient mode."""
        registry = MockToolRegistry(["tool1"])
        validator = ToolDependencyValidator(registry, strict=False)
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"tool1", "missing_tool"}),
            }
        )
        result = validator.validate(workflow)
        assert result.valid is True  # Warnings don't invalidate
        assert "missing_tool" in result.missing_tools
        assert len(result.warnings) == 1

    def test_validate_missing_tool_logs_debug_not_warning_or_error(self, caplog):
        """Missing-tool validation should not spam operational logs."""
        registry = MockToolRegistry(["tool1"])
        validator = ToolDependencyValidator(registry, strict=True)
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"tool1", "missing_tool"}),
            }
        )

        caplog.set_level(logging.DEBUG, logger="victor.workflows.validation.tool_validator")

        result = validator.validate(workflow)

        assert result.valid is False
        assert any(
            record.levelno == logging.DEBUG and "missing_tool" in record.message
            for record in caplog.records
        )
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_validate_compute_node_tools(self):
        """Should validate tools attribute on compute nodes."""
        registry = MockToolRegistry(["compute_tool"])
        validator = ToolDependencyValidator(registry)
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", tools={"compute_tool"}),
            }
        )
        result = validator.validate(workflow)
        assert result.valid is True
        assert "compute_tool" in result.validated_tools

    def test_validate_tools_exist_helper(self):
        """Should validate list of tool names."""
        registry = MockToolRegistry(["tool1", "tool2"])
        validator = ToolDependencyValidator(registry)
        result = validator.validate_tools_exist(["tool1", "tool2", "missing"])
        assert result.valid is False
        assert "tool1" in result.validated_tools
        assert "missing" in result.missing_tools

    def test_validate_without_registry(self):
        """Should track tools as validated without registry."""
        validator = ToolDependencyValidator()
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"any_tool"}),
            }
        )
        result = validator.validate(workflow)
        assert result.valid is True
        assert "any_tool" in result.validated_tools


class TestValidateWorkflowToolsFunction:
    """Test validate_workflow_tools convenience function."""

    def test_with_registry(self):
        """Should validate with provided registry."""
        registry = MockToolRegistry(["tool1"])
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"tool1"}),
            }
        )
        result = validate_workflow_tools(workflow, registry)
        assert result.valid is True

    def test_without_registry(self):
        """Should work without registry."""
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"tool1"}),
            }
        )
        result = validate_workflow_tools(workflow)
        assert result.valid is True

    def test_strict_mode(self):
        """Should respect strict parameter."""
        registry = MockToolRegistry([])
        workflow = MockWorkflow(
            nodes={
                "node1": MockNode(id="node1", allowed_tools={"missing"}),
            }
        )
        result_strict = validate_workflow_tools(workflow, registry, strict=True)
        result_lenient = validate_workflow_tools(workflow, registry, strict=False)
        assert result_strict.valid is False
        assert result_lenient.valid is True
