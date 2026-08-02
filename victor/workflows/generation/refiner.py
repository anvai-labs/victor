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

"""Automated workflow refinement strategies.

This module provides automated fixes for common validation errors,
reducing the need for manual intervention or LLM refinement.

Design Principles (SOLID):
    - SRP: Each refiner handles one error category
    - OCP: Extensible via new refinement strategies
    - LSP: All refiners implement the same interface
    - ISP: Focused refinement methods per category
    - DIP: Depends on WorkflowValidationError abstractions

Key Features:
    - Schema fixes: Add missing fields, remove invalid fields
    - Structure fixes: Remove orphans, connect nodes
    - Semantic fixes: Replace invalid roles, remove unknown tools
    - Conservative mode: Only safe fixes

Example:
    from victor.workflows.generation import WorkflowRefiner

    refiner = WorkflowRefiner(conservative=True)
    refined_workflow, changes = refiner.refine(
        workflow_dict,
        validation_result
    )

    print(f"Applied {len(changes)} automated fixes")
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from victor.workflows.generation.types import (
    ErrorCategory,
    ErrorSeverity,
    WorkflowValidationError,
    WorkflowGenerationValidationResult,
    RefinementResult,
    WorkflowFix,
)

logger = logging.getLogger(__name__)


class SchemaRefiner:
    """Automated schema-level fixes.

    Handles:
    - Add missing required fields with defaults
    - Remove invalid extra fields
    - Convert types (string → int, etc.)
    - Normalize values (clamp ranges, trim strings)
    """

    # Default values for missing fields by node type
    DEFAULTS = {
        "agent": {
            "role": "executor",
            "goal": "Execute the assigned task",
            "tool_budget": 15,
        },
        "compute": {
            "tools": [],
        },
        "condition": {
            "branches": {},
        },
        "parallel": {
            "parallel_nodes": [],
            "join_strategy": "all",
        },
        "transform": {
            "transform": "identity",
        },
        "team": {
            "team_formation": "sequential",
        },
        "hitl": {
            "approval_type": "manual",
        },
    }

    # Valid values for enum-like fields
    VALID_AGENT_ROLES = {
        "researcher",
        "planner",
        "executor",
        "reviewer",
        "writer",
        "analyst",
        "coordinator",
    }

    VALID_NODE_TYPES = {
        "agent",
        "compute",
        "condition",
        "parallel",
        "transform",
        "team",
        "hitl",
    }

    def __init__(self, conservative: bool = True):
        """Initialize schema refiner.

        Args:
            conservative: If True, only safe fixes. If False, aggressive fixes.
        """
        self.conservative = conservative

    def refine(
        self, schema: Dict[str, Any], errors: List[WorkflowValidationError]
    ) -> Tuple[Dict[str, Any], List[WorkflowFix]]:
        """Apply schema-level fixes.

        Args:
            schema: Workflow schema dict
            errors: Schema validation errors

        Returns:
            Tuple of (refined_schema, list_of_fixes)
        """
        refined = copy.deepcopy(schema)
        fixes = []

        for error in errors:
            if error.category != ErrorCategory.SCHEMA:
                continue

            fix = self._fix_error(refined, error)
            if fix:
                fixes.append(fix)

        return refined, fixes

    def _fix_error(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix a single schema error.

        Args:
            schema: Workflow schema (modified in place)
            error: Validation error to fix

        Returns:
            WorkflowFix if fix was applied, None if not fixable
        """
        # Missing required field
        if "missing required field" in error.message.lower():
            return self._fix_missing_field(schema, error)

        # Invalid type
        if "must be" in error.message.lower() and "integer" in error.message.lower():
            return self._fix_type_conversion(schema, error)

        # Invalid value (out of range)
        if "should be between" in error.message.lower() or "exceeds" in error.message.lower():
            return self._fix_range_clamp(schema, error)

        # Invalid enum value
        if "invalid" in error.message.lower():
            return self._fix_invalid_enum(schema, error)

        return None

    def _fix_missing_field(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix missing required field."""
        location = error.location

        # Extract node info from location
        node_match = re.search(r"nodes\[([^\]]+)\]", location)
        if not node_match:
            return None

        node_id_or_index = node_match.group(1)
        field_match = re.search(r"'(\w+)'", error.message)
        if not field_match:
            return None

        missing_field = field_match.group(1)

        # Find the node
        nodes = schema.get("nodes", [])
        node = None
        node_index = None

        # Try by index first
        try:
            node_index = int(node_id_or_index)
            if 0 <= node_index < len(nodes):
                node = nodes[node_index]
        except ValueError:
            # Try by ID
            for i, n in enumerate(nodes):
                if n.get("id") == node_id_or_index:
                    node = n
                    node_index = i
                    break

        if not node:
            return None

        # Get default value for this field
        node_type = node.get("type", "")
        defaults = self.DEFAULTS.get(node_type, {})

        if missing_field in defaults:
            default_value = defaults[missing_field]
            node[missing_field] = default_value

            return WorkflowFix(
                fix_type="schema_add",
                description=f"Added missing '{missing_field}' field with default value",
                location=location,
                after_value=default_value,
                auto_applied=True,
            )

        return None

    def _fix_type_conversion(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix type conversion errors."""
        location = error.location

        # Navigate to the field
        if not self._navigate_to_location(schema, location):
            return None

        # Extract field path
        parts = location.split(".")
        if len(parts) < 2:
            return None

        field_name = parts[-1]

        # Try to find the value
        node_match = re.search(r"nodes\[(\d+)\]", location)
        if not node_match:
            return None

        node_index = int(node_match.group(1))
        nodes = schema.get("nodes", [])
        if node_index >= len(nodes):
            return None

        node = nodes[node_index]
        current_value = node.get(field_name)

        if current_value is None:
            return None

        # Try int conversion
        if field_name in ["tool_budget", "timeout", "max_iterations"]:
            try:
                if isinstance(current_value, str):
                    new_value = int(current_value)
                    node[field_name] = new_value

                    return WorkflowFix(
                        fix_type="schema_convert",
                        description=f"Converted '{field_name}' from string to int",
                        location=location,
                        before_value=current_value,
                        after_value=new_value,
                        auto_applied=True,
                    )
            except (ValueError, TypeError):
                # Use default if conversion fails
                defaults = {
                    "tool_budget": 15,
                    "timeout": 300,
                    "max_iterations": 50,
                }
                if field_name in defaults:
                    node[field_name] = defaults[field_name]

                    return WorkflowFix(
                        fix_type="schema_default",
                        description=f"Set '{field_name}' to default (conversion failed)",
                        location=location,
                        before_value=current_value,
                        after_value=defaults[field_name],
                        auto_applied=True,
                    )

        return None

    def _fix_range_clamp(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix out-of-range values by clamping."""
        location = error.location

        # Navigate to the field
        node_match = re.search(r"nodes\[(\d+)\]", location)
        if not node_match:
            return None

        node_index = int(node_match.group(1))
        nodes = schema.get("nodes", [])
        if node_index >= len(nodes):
            return None

        node = nodes[node_index]
        field_match = re.search(r"(\w+) should be between", error.message)
        if not field_match:
            return None

        field_name = field_match.group(1)
        current_value = node.get(field_name)

        if current_value is None:
            return None

        # Clamp to valid ranges
        if field_name == "tool_budget":
            clamped_value = max(1, min(500, int(current_value)))
            node[field_name] = clamped_value

            return WorkflowFix(
                fix_type="schema_clamp",
                description=f"Clamped '{field_name}' to valid range [1, 500]",
                location=location,
                before_value=current_value,
                after_value=clamped_value,
                auto_applied=True,
            )

        elif field_name == "timeout":
            clamped_value = max(0, float(current_value))
            node[field_name] = clamped_value

            return WorkflowFix(
                fix_type="schema_clamp",
                description=f"Clamped '{field_name}' to non-negative value",
                location=location,
                before_value=current_value,
                after_value=clamped_value,
                auto_applied=True,
            )

        return None

    def _fix_invalid_enum(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix invalid enum values."""
        location = error.location

        node_match = re.search(r"nodes\[(\d+)\]", location)
        if not node_match:
            return None

        node_index = int(node_match.group(1))
        nodes = schema.get("nodes", [])
        if node_index >= len(nodes):
            return None

        node = nodes[node_index]
        field_match = re.search(r"Invalid (\w+): '(\w+)'", error.message)
        if not field_match:
            return None

        field_name = field_match.group(1)
        invalid_value = field_match.group(2)
        current_value = node.get(field_name)

        if current_value != invalid_value:
            return None

        # Find closest valid value
        if field_name == "role":
            # Simple similarity mapping
            role_mapping = {
                "developer": "executor",
                "coder": "executor",
                "planner": "planner",
                "research": "researcher",
                "analyst": "analyst",
            }

            new_value = role_mapping.get(invalid_value.lower(), "executor")  # Default fallback

            node[field_name] = new_value

            return WorkflowFix(
                fix_type="schema_replace",
                description=f"Replaced invalid role '{invalid_value}' with '{new_value}'",
                location=location,
                before_value=invalid_value,
                after_value=new_value,
                auto_applied=True,
            )

        return None

    def _navigate_to_location(self, schema: Dict[str, Any], location: str) -> bool:
        """Navigate to a location in the schema (placeholder for navigation logic)."""
        # This is a placeholder - actual navigation depends on schema structure
        return True


class StructureRefiner:
    """Automated structure-level fixes.

    Handles:
    - Remove orphan nodes
    - Add missing entry point
    - Connect dangling edges
    - Remove duplicate edges
    """

    def __init__(self, conservative: bool = True):
        """Initialize structure refiner.

        Args:
            conservative: If True, only safe fixes
        """
        self.conservative = conservative

    def refine(
        self, schema: Dict[str, Any], errors: List[WorkflowValidationError]
    ) -> Tuple[Dict[str, Any], List[WorkflowFix]]:
        """Apply structure-level fixes.

        Args:
            schema: Workflow schema dict
            errors: Structure validation errors

        Returns:
            Tuple of (refined_schema, list_of_fixes)
        """
        refined = copy.deepcopy(schema)
        fixes = []

        # Build node map
        nodes = refined.get("nodes", [])
        nodes_map = {node.get("id"): node for node in nodes if node.get("id")}

        for error in errors:
            if error.category != ErrorCategory.STRUCTURE:
                continue

            fix = self._fix_error(refined, nodes_map, error)
            if fix:
                fixes.append(fix)
                # Rebuild node map after each fix
                nodes = refined.get("nodes", [])
                nodes_map = {node.get("id"): node for node in nodes if node.get("id")}

        return refined, fixes

    def _fix_error(
        self,
        schema: Dict[str, Any],
        nodes_map: Dict[str, Any],
        error: WorkflowValidationError,
    ) -> Optional[WorkflowFix]:
        """Fix a single structure error."""
        # Orphan node
        if "not reachable" in error.message.lower():
            return self._fix_orphan_node(schema, nodes_map, error)

        # Missing entry point
        if "entry point" in error.message.lower() and "not found" in error.message.lower():
            return self._fix_entry_point(schema, nodes_map, error)

        # Invalid cycle
        if "cycle" in error.message.lower():
            # Don't auto-fix cycles - too risky
            return None

        return None

    def _fix_orphan_node(
        self,
        schema: Dict[str, Any],
        nodes_map: Dict[str, Any],
        error: WorkflowValidationError,
    ) -> Optional[WorkflowFix]:
        """Fix orphan node by removing it."""
        # Extract node ID
        node_match = re.search(r"Node '([^']+)' is not reachable", error.message)
        if not node_match:
            return None

        node_id = node_match.group(1)
        if node_id not in nodes_map:
            return None

        # Remove orphan node
        nodes = schema.get("nodes", [])
        original_count = len(nodes)
        schema["nodes"] = [n for n in nodes if n.get("id") != node_id]

        if len(schema["nodes"]) < original_count:
            return WorkflowFix(
                fix_type="structure_remove",
                description=f"Removed unreachable node '{node_id}'",
                location=f"nodes[{node_id}]",
                before_value=f"Node with {len(nodes_map.get(node_id, {}))} fields",
                auto_applied=True,
            )

        return None

    def _fix_entry_point(
        self,
        schema: Dict[str, Any],
        nodes_map: Dict[str, Any],
        error: WorkflowValidationError,
    ) -> Optional[WorkflowFix]:
        """Fix missing entry point by setting first node."""
        nodes = schema.get("nodes", [])
        if not nodes:
            return None

        # Set first node as entry point
        first_node_id = nodes[0].get("id")
        if not first_node_id:
            return None

        old_entry = schema.get("entry_point", "")
        schema["entry_point"] = first_node_id

        return WorkflowFix(
            fix_type="structure_set_entry",
            description=f"Set entry point to first node '{first_node_id}'",
            location="workflow.entry_point",
            before_value=old_entry or "(missing)",
            after_value=first_node_id,
            auto_applied=True,
        )


class SemanticRefiner:
    """Automated semantic-level fixes.

    Handles:
    - Remove unknown tools
    - Replace invalid roles
    - Remove invalid handlers
    - Add missing optional fields
    """

    def __init__(self, conservative: bool = True, strict_mode: bool = True):
        """Initialize semantic refiner.

        Args:
            conservative: If True, only safe fixes
            strict_mode: If True, errors are failures
        """
        self.conservative = conservative
        self.strict_mode = strict_mode

        # Valid values for enum-like fields
        self.valid_agent_roles = {
            "researcher",
            "planner",
            "executor",
            "reviewer",
            "writer",
            "analyst",
            "coordinator",
        }

        self.valid_team_formations = {
            "sequential",
            "parallel",
            "hierarchical",
            "pipeline",
            "consensus",
        }

    def refine(
        self, schema: Dict[str, Any], errors: List[WorkflowValidationError]
    ) -> Tuple[Dict[str, Any], List[WorkflowFix]]:
        """Apply semantic-level fixes.

        Args:
            schema: Workflow schema dict
            errors: Semantic validation errors

        Returns:
            Tuple of (refined_schema, list_of_fixes)
        """
        refined = copy.deepcopy(schema)
        fixes = []

        for error in errors:
            if error.category != ErrorCategory.SEMANTIC:
                continue

            # Skip critical errors in conservative mode
            if self.conservative and error.severity == ErrorSeverity.CRITICAL:
                continue

            fix = self._fix_error(refined, error)
            if fix:
                fixes.append(fix)

        return refined, fixes

    def _fix_error(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix a single semantic error."""
        # Unknown tool
        if "not found in registry" in error.message.lower() or "tool" in error.message.lower():
            return self._fix_unknown_tool(schema, error)

        # Invalid role
        if "role" in error.location.lower() and "invalid" in error.message.lower():
            return self._fix_invalid_role(schema, error)

        return None

    def _fix_unknown_tool(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix unknown tool by removing it."""
        location = error.location

        node_match = re.search(r"nodes\[([^\]]+)\]", location)
        if not node_match:
            return None

        node_id = node_match.group(1)
        nodes = schema.get("nodes", [])

        # Find the node
        node = None
        for n in nodes:
            if n.get("id") == node_id:
                node = n
                break

        if not node:
            return None

        # Extract tool name
        tool_match = re.search(r"Tool '([^']+)'", error.message)
        if not tool_match:
            return None

        tool_name = tool_match.group(1)
        tools = node.get("tools", [])

        if tool_name not in tools:
            return None

        # Remove tool
        node["tools"] = [t for t in tools if t != tool_name]

        return WorkflowFix(
            fix_type="semantic_remove_tool",
            description=f"Removed unknown tool '{tool_name}'",
            location=location,
            before_value=tool_name,
            after_value="(removed)",
            auto_applied=True,
        )

    def _fix_invalid_role(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix invalid role by mapping to valid one."""
        location = error.location

        node_match = re.search(r"nodes\[([^\]]+)\]", location)
        if not node_match:
            return None

        node_id = node_match.group(1)
        nodes = schema.get("nodes", [])

        # Find the node
        node = None
        for n in nodes:
            if n.get("id") == node_id:
                node = n
                break

        if not node:
            return None

        current_role = node.get("role")
        if not current_role:
            return None

        # Map to closest valid role
        role_mapping = {
            "developer": "executor",
            "coder": "executor",
            "programmer": "executor",
            "planner": "planner",
            "research": "researcher",
            "researcher": "researcher",
        }

        new_role = role_mapping.get(current_role.lower(), "executor")
        node["role"] = new_role

        return WorkflowFix(
            fix_type="semantic_replace_role",
            description=f"Replaced invalid role '{current_role}' with '{new_role}'",
            location=location,
            before_value=current_role,
            after_value=new_role,
            auto_applied=True,
        )


class SecurityRefiner:
    """Automated security-level fixes.

    Handles:
    - Clamp resource limits
    - Remove dangerous tool combinations
    - Warn about privileged tools
    """

    def __init__(self, conservative: bool = True):
        """Initialize security refiner.

        Args:
            conservative: If True, only safe fixes
        """
        self.conservative = conservative

    def refine(
        self, schema: Dict[str, Any], errors: List[WorkflowValidationError]
    ) -> Tuple[Dict[str, Any], List[WorkflowFix]]:
        """Apply security-level fixes.

        Args:
            schema: Workflow schema dict
            errors: Security validation errors

        Returns:
            Tuple of (refined_schema, list_of_fixes)
        """
        refined = copy.deepcopy(schema)
        fixes = []

        for error in errors:
            if error.category != ErrorCategory.SECURITY:
                continue

            # Skip critical security issues
            if error.severity == ErrorSeverity.CRITICAL:
                continue

            fix = self._fix_error(refined, error)
            if fix:
                fixes.append(fix)

        return refined, fixes

    def _fix_error(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix a single security error."""
        # Resource limit exceeded
        if "exceeds" in error.message.lower() or "exceeded" in error.message.lower():
            return self._fix_resource_limit(schema, error)

        return None

    def _fix_resource_limit(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Fix resource limit by clamping."""
        # Extract field and limit
        field_match = re.search(r"(\w+) (\d+) exceeds limit (\d+)", error.message)
        if not field_match:
            return None

        _resource_type = field_match.group(1)  # noqa: F841
        current_value = int(field_match.group(2))
        max_value = int(field_match.group(3))

        if "tool budget" in error.message.lower():
            # Reduce individual node budgets proportionally
            nodes = schema.get("nodes", [])
            agent_nodes = [n for n in nodes if n.get("type") == "agent"]

            if agent_nodes:
                scale_factor = max_value / current_value
                for node in agent_nodes:
                    if "tool_budget" in node:
                        old_budget = node["tool_budget"]
                        new_budget = max(1, int(old_budget * scale_factor))
                        node["tool_budget"] = new_budget

                return WorkflowFix(
                    fix_type="security_clamp_budget",
                    description=f"Scaled down tool budgets to meet limit {max_value}",
                    location="workflow",
                    before_value=f"Total: {current_value}",
                    after_value=f"Total: {max_value}",
                    auto_applied=True,
                )

        return None


class WorkflowRefiner:
    """Main refiner coordinating all refinement strategies.

    This is the primary interface for automated workflow refinement.

    Example:
        refiner = WorkflowRefiner(conservative=True)

        refined_schema, fixes = refiner.refine(
            workflow_dict,
            validation_result
        )

        print(f"Applied {len(fixes)} automated fixes")
    """

    def __init__(self, conservative: bool = True, strict_mode: bool = True):
        """Initialize workflow refiner.

        Args:
            conservative: If True, only safe fixes
            strict_mode: If True, errors are failures
        """
        self.conservative = conservative
        self.strict_mode = strict_mode

        self.schema_refiner = SchemaRefiner(conservative=conservative)
        self.structure_refiner = StructureRefiner(conservative=conservative)
        self.semantic_refiner = SemanticRefiner(conservative=conservative, strict_mode=strict_mode)
        self.security_refiner = SecurityRefiner(conservative=conservative)

    def refine(
        self,
        schema: Dict[str, Any],
        validation_result: WorkflowGenerationValidationResult,
    ) -> RefinementResult:
        """Apply automated refinements to workflow.

        Args:
            schema: Workflow schema dict
            validation_result: Validation result with errors

        Returns:
            RefinementResult with refined schema and applied fixes
        """
        all_fixes: List[WorkflowFix] = []
        refined_schema = copy.deepcopy(schema)
        original_errors = validation_result.all_errors.copy()

        # Apply fixes by category (in priority order)
        # 1. Schema first (foundational)
        if validation_result.schema_errors:
            refined_schema, schema_fixes = self.schema_refiner.refine(
                refined_schema, validation_result.schema_errors
            )
            all_fixes.extend(schema_fixes)

        # 2. Structure second (affects execution)
        if validation_result.structure_errors:
            refined_schema, structure_fixes = self.structure_refiner.refine(
                refined_schema, validation_result.structure_errors
            )
            all_fixes.extend(structure_fixes)

        # 3. Semantic third (node-specific)
        if validation_result.semantic_errors:
            refined_schema, semantic_fixes = self.semantic_refiner.refine(
                refined_schema, validation_result.semantic_errors
            )
            all_fixes.extend(semantic_fixes)

        # 4. Security last (safety)
        if validation_result.security_errors:
            refined_schema, security_fixes = self.security_refiner.refine(
                refined_schema, validation_result.security_errors
            )
            all_fixes.extend(security_fixes)

        # Build result
        success = len(all_fixes) > 0
        fix_descriptions = [f.description for f in all_fixes]

        return RefinementResult(
            success=success,
            refined_schema=refined_schema,
            iterations=1,
            fixes_applied=fix_descriptions,
            original_errors=original_errors,
            remaining_errors=[],  # Would need revalidation to populate
            convergence_achieved=len(all_fixes) > 0,
        )

    def refine_single_error(
        self, schema: Dict[str, Any], error: WorkflowValidationError
    ) -> Optional[WorkflowFix]:
        """Apply refinement for a single error.

        Useful for selective refinement.

        Args:
            schema: Workflow schema dict
            error: Single validation error

        Returns:
            WorkflowFix if fix was applied, None otherwise
        """
        # Route to appropriate refiner
        if error.category == ErrorCategory.SCHEMA:
            refined_schema, fixes = self.schema_refiner.refine(schema, [error])
        elif error.category == ErrorCategory.STRUCTURE:
            refined_schema, fixes = self.structure_refiner.refine(schema, [error])
        elif error.category == ErrorCategory.SEMANTIC:
            refined_schema, fixes = self.semantic_refiner.refine(schema, [error])
        elif error.category == ErrorCategory.SECURITY:
            refined_schema, fixes = self.security_refiner.refine(schema, [error])
        else:
            return None

        return fixes[0] if fixes else None


__all__ = [
    "WorkflowRefiner",
    "SchemaRefiner",
    "StructureRefiner",
    "SemanticRefiner",
    "SecurityRefiner",
]
