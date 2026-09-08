"""Pydantic models for workflow execution context.

Provides type-safe, validated state models for workflow execution,
replacing TypedDict with better type checking and validation.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


class WorkflowExecutionContextModel(BaseModel):
    """Unified execution context for all workflow runtimes (Pydantic v2).

    This model replaces the WorkflowExecutionContext TypedDict with:
    - Type checking with mypy
    - Runtime validation
    - Better error messages
    - Automatic serialization
    - IDE autocomplete

    Attributes:
        # Core data (user-defined workflow data)
        data: Dict[str, Any] — Shared context data dictionary
        messages: List[Dict[str, Any]] — Conversation messages (for agent workflows)

        # Execution metadata (system-managed)
        workflow_id: str — Unique workflow execution ID (auto-generated UUID)
        workflow_name: str — Name of the workflow being executed
        current_node: str — Currently executing node ID
        node_results: Dict[str, Any] — Results from each executed node
        error: Optional[str] — Error message if execution failed

        # Iteration tracking (for loop detection)
        iteration: int — Current iteration count (default: 0)
        visited_nodes: List[str] — List of visited node IDs (default: [])

        # Parallel execution
        parallel_results: Dict[str, Any] — Results from parallel node execution

        # Human-in-the-loop
        hitl_pending: bool — Whether waiting for human input (default: False)
        hitl_response: Optional[Dict[str, Any]] — Human response data

        # Temporal context (for backtesting)
        as_of_date: Optional[str] — Point-in-time date for temporal queries
        lookback_periods: Optional[int] — Number of periods to look back
        include_end_date: bool — Whether to include end date in ranges

        # Completion tracking
        is_complete: bool — Whether workflow has completed (default: False)
        success: bool — Whether workflow completed successfully (default: False)

    Note:
        Field names use underscore prefix (e.g., workflow_id not _workflow_id)
        for consistency with TypedDict version. This differs from typical
        Pydantic conventions but maintains compatibility during migration.
    """

    # Model configuration
    model_config = ConfigDict(
        # Allow arbitrary types for flexibility during migration
        arbitrary_types_allowed=True,
        # Validate assignment values
        validate_assignment=True,
        # Use enum values (not their names)
        use_enum_values=True,
    )

    # Core data
    data: Dict[str, Any] = Field(default_factory=dict)
    messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Execution metadata
    workflow_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    workflow_name: str = Field(default="")
    current_node: str = Field(default="")
    node_results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    # Iteration tracking
    iteration: int = Field(default=0, ge=0)
    visited_nodes: List[str] = Field(default_factory=list)

    # Parallel execution
    parallel_results: Dict[str, Any] = Field(default_factory=dict)

    # Human-in-the-loop
    hitl_pending: bool = Field(default=False)
    hitl_response: Optional[Dict[str, Any]] = None

    # Temporal context
    as_of_date: Optional[str] = None
    lookback_periods: Optional[int] = Field(default=None, ge=0)
    include_end_date: bool = Field(default=True)

    # Completion tracking
    is_complete: bool = Field(default=False)
    success: bool = Field(default=False)

    # Validators
    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id(cls, v: Optional[str]) -> str:
        """Validate workflow_id is a valid UUID or identifier."""
        # None is allowed at initialization (will use default factory)
        if v is None:
            return uuid.uuid4().hex
        if not v:
            raise ValueError("workflow_id cannot be empty")
        # Allow UUID format or simple identifiers
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(f"workflow_id must be alphanumeric: {v}")
        return v

    @field_validator("visited_nodes")
    @classmethod
    def validate_visited_nodes(cls, v: List[str]) -> List[str]:
        """Validate visited_nodes list."""
        if v and len(set(v)) != len(v):
            raise ValueError("visited_nodes must be unique (no duplicates)")
        return v

    # Utility methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict compatible with TypedDict-based code.

        Adds underscore prefixes to match TypedDict field naming convention.
        """
        return {
            "data": self.data,
            "messages": self.messages,
            "_workflow_id": self.workflow_id,
            "_workflow_name": self.workflow_name,
            "_current_node": self.current_node,
            "_node_results": self.node_results,
            "_error": self.error,
            "_iteration": self.iteration,
            "_visited_nodes": self.visited_nodes,
            "_parallel_results": self.parallel_results,
            "_hitl_pending": self.hitl_pending,
            "_hitl_response": self.hitl_response,
            "_as_of_date": self.as_of_date,
            "_lookback_periods": self.lookback_periods,
            "_include_end_date": self.include_end_date,
            "_is_complete": self.is_complete,
            "_success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowExecutionContextModel:
        """Create Pydantic model from dict (TypedDict-compatible).

        Removes underscore prefixes from TypedDict field names.
        """
        return cls(
            data=data.get("data", {}),
            messages=data.get("messages", []),
            workflow_id=data.get("_workflow_id", uuid.uuid4().hex),
            workflow_name=data.get("_workflow_name", ""),
            current_node=data.get("_current_node", ""),
            node_results=data.get("_node_results", {}),
            error=data.get("_error"),
            iteration=data.get("_iteration", 0),
            visited_nodes=data.get("_visited_nodes", []),
            parallel_results=data.get("_parallel_results", {}),
            hitl_pending=data.get("_hitl_pending", False),
            hitl_response=data.get("_hitl_response"),
            as_of_date=data.get("_as_of_date"),
            lookback_periods=data.get("_lookback_periods"),
            include_end_date=data.get("_include_end_date", True),
            is_complete=data.get("_is_complete", False),
            success=data.get("_success", False),
        )

    def add_node_result(self, node_id: str, result: Any) -> None:
        """Add a node result to the context."""
        self.node_results[node_id] = result

    def visit_node(self, node_id: str) -> None:
        """Mark a node as visited."""
        if node_id not in self.visited_nodes:
            self.visited_nodes.append(node_id)

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration += 1

    def mark_complete(self, success: bool = True) -> None:
        """Mark the workflow as complete."""
        self.is_complete = True
        self.success = success

    # Dict-like methods for compatibility with StateGraph
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key (dict-like interface)."""
        # First check if it's a direct attribute (data, messages)
        if hasattr(self, key):
            return getattr(self, key)
        # Otherwise convert to dict format with underscore prefixes
        dict_key = f"_{key}" if not key.startswith("_") else key
        dict_repr = self.to_dict()
        return dict_repr.get(dict_key, default)

    def keys(self) -> list:
        """Return list of keys (dict-like interface)."""
        dict_repr = self.to_dict()
        # Include data field contents at top level
        keys = list(dict_repr.keys())
        if hasattr(self, "data"):
            # Remove "data" key and add its contents at top level
            keys.remove("data")
            keys.extend(self.data.keys())
        return keys

    def items(self) -> list:
        """Return list of (key, value) tuples (dict-like interface)."""
        dict_repr = self.to_dict()
        # Include data field contents at top level
        items = list(dict_repr.items())
        if hasattr(self, "data") and "data" in dict_repr:
            # Remove "data" entry and add its contents at top level
            items = [kv for kv in items if kv[0] != "data"]
            items.extend(self.data.items())
        return items

    def values(self) -> list:
        """Return list of values (dict-like interface)."""
        dict_repr = self.to_dict()
        # Include data field contents at top level
        values = list(dict_repr.values())
        if hasattr(self, "data") and "data" in dict_repr:
            # Remove "data" value and add its contents at top level
            values = [v for k, v in dict_repr.items() if k != "data"]
            values.extend(self.data.values())
        return values

    def __getitem__(self, key: str) -> Any:
        """Get item by key (dict-like subscript access)."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key (dict-like subscript access)."""
        # First check if it's a direct attribute (data, messages)
        if hasattr(self.__class__, key):
            setattr(self, key, value)
        else:
            # Convert underscore-prefixed keys to attribute names
            attr_name = key.lstrip("_")
            if hasattr(self.__class__, attr_name):
                setattr(self, attr_name, value)
            else:
                # For unknown keys, store them in the data field
                self.data[key] = value


class WorkflowStateModel(BaseModel):
    """Generic state for compiled workflow execution (Pydantic v2).

    Simpler model for compiled workflow execution without all the
    workflow execution context fields.

    Attributes:
        data: User data dictionary for workflow state
        workflow_id: Unique workflow execution ID
        workflow_name: Name of the workflow being executed
        current_node: Currently executing node ID
        node_results: Results from each executed node
        error: Error message if execution failed
        iteration: Current iteration count
        parallel_results: Results from parallel node execution
        hitl_pending: Whether waiting for human input
        hitl_response: Human response data
    """

    # Model configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    # Fields
    data: Dict[str, Any] = Field(default_factory=dict)
    workflow_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    workflow_name: str = Field(default="")
    current_node: str = Field(default="")
    node_results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    iteration: int = Field(default=0, ge=0)
    parallel_results: Dict[str, Any] = Field(default_factory=dict)
    hitl_pending: bool = Field(default=False)
    hitl_response: Optional[Dict[str, Any]] = None

    # Validators
    @field_validator("workflow_id")
    @classmethod
    def validate_workflow_id(cls, v: str) -> str:
        """Validate workflow_id is a valid identifier."""
        if not v:
            raise ValueError("workflow_id cannot be empty")
        return v

    # Utility methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict compatible with TypedDict-based code."""
        return {
            "data": self.data,
            "_workflow_id": self.workflow_id,
            "_workflow_name": self.workflow_name,
            "_current_node": self.current_node,
            "_node_results": self.node_results,
            "_error": self.error,
            "_iteration": self.iteration,
            "_parallel_results": self.parallel_results,
            "_hitl_pending": self.hitl_pending,
            "_hitl_response": self.hitl_response,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowStateModel:
        """Create Pydantic model from dict (TypedDict-compatible)."""
        return cls(
            data=data.get("data", {}),
            workflow_id=data.get("_workflow_id", uuid.uuid4().hex),
            workflow_name=data.get("_workflow_name", ""),
            current_node=data.get("_current_node", ""),
            node_results=data.get("_node_results", {}),
            error=data.get("_error"),
            iteration=data.get("_iteration", 0),
            parallel_results=data.get("_parallel_results", {}),
            hitl_pending=data.get("_hitl_pending", False),
            hitl_response=data.get("_hitl_response"),
        )

    def __getitem__(self, key: str) -> Any:
        """Get item by key (dict-like subscript access)."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key (dict-like subscript access)."""
        # Convert underscore-prefixed keys to attribute names
        attr_name = key.lstrip("_")
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)
        else:
            # For unknown keys, store them in the data field
            self.data[key] = value

    # Dict-like methods for compatibility with StateGraph
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key (dict-like interface)."""
        # User keys may share names with mapping methods or model fields.
        # The flat mapping must return the payload, never a bound method.
        if key in self.data:
            return self.data[key]
        if key in type(self).model_fields:
            return getattr(self, key)

        # Otherwise convert to dict format with underscore prefixes
        dict_key = f"_{key}" if not key.startswith("_") else key
        dict_repr = self.to_dict()
        return dict_repr.get(dict_key, default)

    def keys(self) -> list:
        """Return list of keys (dict-like interface)."""
        dict_repr = self.to_dict()
        # Include data field contents at top level
        keys = list(dict_repr.keys())
        if hasattr(self, "data"):
            # Remove "data" key and add its contents at top level
            keys.remove("data")
            keys.extend(self.data.keys())
        return keys

    def items(self) -> list:
        """Return list of (key, value) tuples (dict-like interface)."""
        dict_repr = self.to_dict()
        # Include data field contents at top level
        items = list(dict_repr.items())
        if hasattr(self, "data") and "data" in dict_repr:
            # Remove "data" entry and add its contents at top level
            items = [kv for kv in items if kv[0] != "data"]
            items.extend(self.data.items())
        return items

    def values(self) -> list:
        """Return list of values (dict-like interface)."""
        dict_repr = self.to_dict()
        # Include data field contents at top level
        values = list(dict_repr.values())
        if hasattr(self, "data") and "data" in dict_repr:
            # Remove "data" value and add its contents at top level
            values = [v for k, v in dict_repr.items() if k != "data"]
            values.extend(self.data.values())
        return values


__all__ = [
    "WorkflowExecutionContextModel",
    "WorkflowStateModel",
]
