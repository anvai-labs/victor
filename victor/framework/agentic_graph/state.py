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

"""AgenticLoopState - Unified state model for StateGraph-based agentic loop.

This module provides the single canonical Pydantic state model for the agentic
loop execution in StateGraph.  The state tracks all information needed for the
PERCEIVE-PLAN-ACT-EVALUATE-DECIDE loop.

Consolidation Note (Design Decision):
    The legacy ``AgenticLoopState`` TypedDict has been replaced by a
    deprecation alias pointing to ``AgenticLoopStateModel``.  All code
    should use ``AgenticLoopStateModel`` directly; the alias exists solely
    for backward-compatible imports.

    Rationale for removing the TypedDict:
    - TypedDict provides **no runtime validation** (all fields optional via
      ``total=False``), so malformed state was silently accepted.
    - ``AgenticLoopStateModel`` already provides a dict-like interface
      (``__getitem__``, ``__setitem__``, ``get``, ``keys``, ``values``,
      ``items``), making the TypedDict redundant for StateGraph compatibility.
    - Legacy dict and ``CopyOnWriteState`` inputs are normalized once at the
      callable boundary instead of forcing every node body to handle multiple
      runtime shapes.
    - Pydantic v2 validators (``validate_assignment``, ``field_validator``)
      catch invalid state at the point of mutation rather than failing
      silently and exploding downstream.

Design Principles:
- Pydantic v2 for runtime validation and serialization
- Dict-like interface for StateGraph compatibility
- Immutable via model_copy() for functional updates
- Checkpointable via model_dump() / model_validate()
- Single source of truth (no parallel TypedDict)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from victor.framework.evaluation_nodes import EvaluationDecision

# =============================================================================
# Deprecation Alias (PEP 562)
# =============================================================================


def __getattr__(name: str) -> Any:
    """Module-level ``__getattr__`` to emit a deprecation warning for
    ``AgenticLoopState`` imports while keeping backward compatibility.

    This is the Python-accepted pattern for lazy deprecation aliases
    (PEP 562).  Importing ``AgenticLoopState`` will still resolve to
    ``AgenticLoopStateModel`` but will emit a ``DeprecationWarning``.
    """
    if name == "AgenticLoopState":
        warnings.warn(
            "AgenticLoopState (TypedDict) is deprecated. "
            "Use AgenticLoopStateModel instead. "
            "The TypedDict provided no runtime validation; "
            "AgenticLoopStateModel is a full Pydantic model with "
            "the same dict-like interface. Will be removed in v0.10.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AgenticLoopStateModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# Canonical State Model
# =============================================================================


class AgenticLoopStateModel(BaseModel):
    """Unified Pydantic model for agentic loop state.

    This is the **single canonical** state type for the agentic loop graph.
    It replaces the former ``AgenticLoopState`` TypedDict, providing the same
    dict-like interface with the addition of runtime validation.

    Capabilities over the old TypedDict:
    - Runtime field validation (max_iterations bounds, progress_scores range)
    - Immutable updates via ``model_copy(update={...})``
    - Serialization / deserialization via ``model_dump()`` / ``model_validate()``
    - Dict-like access (``state["key"]``, ``state.get("key")``, ``"key" in state``)

    Example:
        state = AgenticLoopStateModel(query="Fix the bug")
        state = state.model_copy(update={"iteration": 1, "stage": "perceive"})
        assert state["query"] == "Fix the bug"  # Dict-like access

    Attributes:
        query: User's input query or task (required)
        iteration: Current loop iteration (default: 0)
        max_iterations: Maximum iterations (default: 10)
        stage: Current stage name (optional)
        perception: Perception result dict (optional)
        task_type: Task type classification (optional)
        complexity: Complexity level (optional)
        plan: Execution plan dict (optional)
        action_result: Action execution result (optional)
        tool_results: List of tool results (default: [])
        evaluation: Evaluation result dict (optional)
        progress_scores: Progress score history (default: [])
        fulfillment: Fulfillment check result (optional)
        context: Additional context dict (optional)
        conversation_history: Conversation messages (optional)
        planning_events: Planning decision metadata emitted by graph nodes
        plan_execution_state: Serializable planning execution/checkpoint state
        planning_routing_hints: Learned or injected planning hints
        structured_routing_policy: Serialized structured routing policy snapshot
        topology_events: Serialized topology telemetry emitted during execution
        degradation_events: Runtime degradation/error records captured in state
        _execution_context: Internal ExecutionContext (excluded from serialization)
    """

    # Core loop state
    query: str
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=10, ge=1, le=100)

    # Stage tracking
    stage: Optional[str] = Field(default=None)

    # Perception results
    perception: Optional[Dict[str, Any]] = Field(default=None)
    task_type: Optional[str] = Field(default=None)
    complexity: Optional[str] = Field(default=None)

    # Planning
    plan: Optional[Dict[str, Any]] = Field(default=None)

    # Execution results
    action_result: Optional[Dict[str, Any]] = Field(default=None)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Evaluation
    evaluation: Optional[Dict[str, Any]] = Field(default=None)
    progress_scores: List[float] = Field(default_factory=list)

    # Fulfillment
    fulfillment: Optional[Dict[str, Any]] = Field(default=None)

    # Metadata
    context: Optional[Dict[str, Any]] = Field(default=None)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None)
    planning_events: List[Dict[str, Any]] = Field(default_factory=list)
    plan_execution_state: Dict[str, Any] = Field(default_factory=dict)
    planning_routing_hints: Dict[str, Any] = Field(default_factory=dict)
    structured_routing_policy: Dict[str, Any] = Field(default_factory=dict)
    topology_events: List[Dict[str, Any]] = Field(default_factory=list)
    degradation_events: List[Dict[str, Any]] = Field(default_factory=list)

    # Internal ExecutionContext (private attribute, excluded from serialization)
    _execution_context: Optional[Any] = PrivateAttr(default=None)

    # Private attribute for service injection (not serialized)
    _execution_context_private: Optional[Any] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max_iterations is positive."""
        if v < 1:
            raise ValueError("max_iterations must be at least 1")
        if v > 100:
            raise ValueError("max_iterations cannot exceed 100")
        return v

    @field_validator("progress_scores")
    @classmethod
    def validate_progress_scores(cls, v: List[float]) -> List[float]:
        """Validate progress scores are in valid range."""
        for score in v:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"progress score {score} not in [0.0, 1.0]")
        return v

    # ==========================================================================
    # Dict-like interface for StateGraph compatibility
    # ==========================================================================

    def __getitem__(self, key: str) -> Any:
        """Get item by key (dict-like access)."""
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key (dict-like mutation)."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists (dict-like membership)."""
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        """Get item with default (dict-like)."""
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        """Return list of keys (dict-like)."""
        return list(type(self).model_fields.keys())

    def values(self) -> List[Any]:
        """Return list of values (dict-like)."""
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> List[tuple[str, Any]]:
        """Return list of key-value tuples (dict-like)."""
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict (excluding internal fields)."""
        return {key: getattr(self, key) for key in self.keys()}


# =============================================================================
# Helper Functions
# =============================================================================


def create_initial_state(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 10,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> AgenticLoopStateModel:
    """Create initial state for agentic loop execution.

    Args:
        query: User's input query or task
        context: Optional additional context
        max_iterations: Maximum iterations (default: 10)
        conversation_history: Optional prior conversation turns

    Returns:
        Initial AgenticLoopStateModel ready for execution
    """
    return AgenticLoopStateModel(
        query=query,
        iteration=0,
        max_iterations=max_iterations,
        context=context or {},
        progress_scores=[],
        tool_results=[],
        conversation_history=conversation_history,
    )


def should_continue_loop(state: AgenticLoopStateModel) -> bool:
    """Check if agentic loop should continue iterating.

    Args:
        state: Current agentic loop state

    Returns:
        True if loop should continue, False otherwise
    """
    # Check max iterations
    if state.iteration >= state.max_iterations:
        return False

    # Check evaluation decision (handle both string and enum)
    if state.evaluation:
        decision = state.evaluation.get("decision")
        # Normalize to string for comparison
        if isinstance(decision, EvaluationDecision):
            decision_str = decision.value
        else:
            decision_str = str(decision).lower()

        if decision_str == EvaluationDecision.COMPLETE.value or decision_str == "complete":
            return False
        if decision_str == EvaluationDecision.FAIL.value or decision_str == "fail":
            return False

    # Default: continue
    return True


def get_evaluation_decision(state: AgenticLoopStateModel) -> str:
    """Get the evaluation decision from state.

    Args:
        state: Current agentic loop state

    Returns:
        Evaluation decision string (default: "continue")
    """
    if state.evaluation:
        return state.evaluation.get("decision", "continue")
    return "continue"
