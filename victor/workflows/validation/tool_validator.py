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

"""Tool dependency validation for YAML workflows.

This module validates that all tools referenced in workflow nodes exist
in the tool registry before workflow execution begins.

Design Principles (SOLID):
- SRP: Focused solely on tool validation
- DIP: Uses Protocol for tool registry abstraction
- OCP: Extensible via strict mode and severity levels

Benefits for all verticals:
- Early error detection at workflow load time (not runtime)
- Clear error messages identifying missing tools by node
- Support for strict (fail-fast) and lenient (warn) modes
- Works with any ToolRegistry implementation via Protocol

Example:
    from victor.workflows.validation import ToolDependencyValidator
    from victor.tools.registry import ToolRegistry

    # Get tool registry
    registry = ToolRegistry.get_shared_instance()

    # Create validator
    validator = ToolDependencyValidator(registry, strict=True)

    # Validate workflow
    result = validator.validate(workflow)
    if not result.valid:
        for error in result.errors:
            print(f"Node '{error.node_id}': tool '{error.tool_name}' not found")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set
import logging

logger = logging.getLogger(__name__)


class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry - allows any registry implementation.

    This protocol enables dependency injection and testing without
    coupling to a specific ToolRegistry implementation.
    """

    def get(self, name: str) -> Optional[Any]:
        """Get tool by name.

        Args:
            name: Tool name to look up

        Returns:
            Tool instance or None if not found
        """
        ...

    def list_tools(self, only_enabled: bool = True) -> List[Any]:
        """List all available tools.

        Args:
            only_enabled: If True, only return enabled tools

        Returns:
            List of tool instances
        """
        ...


@dataclass
class ToolValidationError:
    """Error found during tool validation.

    Attributes:
        node_id: ID of the workflow node with the error
        tool_name: Name of the tool that failed validation
        error_type: Type of error ("missing", "disabled", "incompatible")
        message: Human-readable error message
        severity: Error severity ("error" or "warning")
    """

    node_id: str
    tool_name: str
    error_type: str  # "missing", "disabled", "incompatible"
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        """Format as human-readable string."""
        return f"[{self.severity.upper()}] Node '{self.node_id}': {self.message}"


@dataclass
class ToolValidationResult:
    """Result of tool dependency validation.

    Attributes:
        valid: True if no errors found (warnings allowed)
        errors: List of error-level validation issues
        warnings: List of warning-level validation issues
        validated_tools: Set of tools that passed validation
        missing_tools: Set of tools that were not found
    """

    valid: bool = True
    errors: List[ToolValidationError] = field(default_factory=list)
    warnings: List[ToolValidationError] = field(default_factory=list)
    validated_tools: Set[str] = field(default_factory=set)
    missing_tools: Set[str] = field(default_factory=set)

    def add_error(self, error: ToolValidationError) -> None:
        """Add validation error."""
        if error.severity == "warning":
            self.warnings.append(error)
        else:
            self.errors.append(error)
            self.valid = False

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.valid:
            return f"Valid: {len(self.validated_tools)} tools verified"
        return (
            f"Invalid: {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings, "
            f"{len(self.missing_tools)} missing tools"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": self.valid,
            "errors": [
                {
                    "node_id": e.node_id,
                    "tool_name": e.tool_name,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.errors
            ],
            "warnings": [
                {
                    "node_id": w.node_id,
                    "tool_name": w.tool_name,
                    "error_type": w.error_type,
                    "message": w.message,
                }
                for w in self.warnings
            ],
            "validated_tools": sorted(self.validated_tools),
            "missing_tools": sorted(self.missing_tools),
        }


class ToolDependencyValidator:
    """Validates tool dependencies in workflows at load time.

    Ensures all tools referenced in workflow nodes exist and are
    available before workflow execution begins.

    Attributes:
        tool_registry: Registry to check for tool existence
        strict: If True, missing tools are errors; if False, warnings

    Example:
        validator = ToolDependencyValidator(tool_registry)
        result = validator.validate(workflow)
        if not result.valid:
            raise WorkflowValidationError(result.errors)
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistryProtocol] = None,
        strict: bool = True,
    ):
        """Initialize validator.

        Args:
            tool_registry: Registry to check for tool existence.
                If None, validator will only track requested tools.
            strict: If True, missing tools cause errors. If False, warnings.
        """
        self.tool_registry = tool_registry
        self.strict = strict
        self._available_tools: Optional[Set[str]] = None

    def set_registry(self, registry: ToolRegistryProtocol) -> None:
        """Set or update tool registry.

        Args:
            registry: New registry to use for validation
        """
        self.tool_registry = registry
        self._available_tools = None  # Reset cache

    @property
    def available_tools(self) -> Set[str]:
        """Get set of available tool names (cached)."""
        if self._available_tools is None:
            if self.tool_registry:
                tools = self.tool_registry.list_tools(only_enabled=True)
                self._available_tools = {getattr(t, "name", str(t)) for t in tools}
            else:
                self._available_tools = set()
        return self._available_tools

    def validate(self, workflow: Any) -> ToolValidationResult:
        """Validate all tool dependencies in workflow.

        Args:
            workflow: WorkflowDefinition to validate

        Returns:
            ToolValidationResult with validation status and any errors
        """
        result = ToolValidationResult()

        # Get all nodes from workflow
        nodes = getattr(workflow, "nodes", {})
        if not nodes:
            return result

        for node_id, node in nodes.items():
            self._validate_node(node_id, node, result)

        return result

    def _validate_node(
        self,
        node_id: str,
        node: Any,
        result: ToolValidationResult,
    ) -> None:
        """Validate tools for a single node.

        Args:
            node_id: ID of the node being validated
            node: Node object to validate
            result: Result object to add errors to
        """
        tools_to_check: Set[str] = set()

        # Extract tools based on node attributes
        # AgentNode has allowed_tools
        allowed_tools = getattr(node, "allowed_tools", None)
        if allowed_tools:
            if isinstance(allowed_tools, (list, set)):
                tools_to_check.update(allowed_tools)

        # ComputeNode may have tools
        node_tools = getattr(node, "tools", None)
        if node_tools:
            if isinstance(node_tools, (list, set)):
                tools_to_check.update(node_tools)

        # Validate each tool
        for tool_name in tools_to_check:
            self._validate_tool(node_id, tool_name, result)

    def _validate_tool(
        self,
        node_id: str,
        tool_name: str,
        result: ToolValidationResult,
    ) -> None:
        """Validate a single tool exists.

        Args:
            node_id: ID of the node referencing the tool
            tool_name: Name of the tool to validate
            result: Result object to add errors to
        """
        # If no registry, we can't validate - track as validated
        if not self.tool_registry:
            result.validated_tools.add(tool_name)
            return

        if tool_name in self.available_tools:
            result.validated_tools.add(tool_name)
        else:
            result.missing_tools.add(tool_name)
            severity = "error" if self.strict else "warning"
            result.add_error(
                ToolValidationError(
                    node_id=node_id,
                    tool_name=tool_name,
                    error_type="missing",
                    message=f"Tool '{tool_name}' not found in registry",
                    severity=severity,
                )
            )
            logger.debug(
                "Workflow validation: tool '%s' in node '%s' not found in registry "
                "(severity=%s)",
                tool_name,
                node_id,
                severity,
            )

    def validate_tools_exist(
        self,
        tool_names: List[str],
        context: str = "",
    ) -> ToolValidationResult:
        """Validate a list of tool names exist (without workflow context).

        Useful for validating tool configurations outside of workflows.

        Args:
            tool_names: List of tool names to validate
            context: Optional context string for error messages

        Returns:
            ToolValidationResult with validation status
        """
        result = ToolValidationResult()
        node_id = context or "config"

        for name in tool_names:
            self._validate_tool(node_id, name, result)

        return result


def validate_workflow_tools(
    workflow: Any,
    tool_registry: Optional[ToolRegistryProtocol] = None,
    strict: bool = True,
) -> ToolValidationResult:
    """Convenience function to validate workflow tools.

    Args:
        workflow: WorkflowDefinition to validate
        tool_registry: Optional registry for validation
        strict: If True, missing tools are errors

    Returns:
        ToolValidationResult
    """
    validator = ToolDependencyValidator(tool_registry, strict=strict)
    return validator.validate(workflow)


__all__ = [
    "ToolRegistryProtocol",
    "ToolValidationError",
    "ToolValidationResult",
    "ToolDependencyValidator",
    "validate_workflow_tools",
]
