"""Adapter utilities for Pydantic state models.

Provides conversion utilities between Pydantic models and TypedDict-compatible
dicts for backward compatibility during migration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from victor.workflows.models import WorkflowExecutionContextModel

# Generic type variable for BaseModel subclasses
T = TypeVar("T", bound=BaseModel)


class StateAdapter:
    """Adapter for converting between Pydantic models and dict-like objects.

    Provides utilities to migrate between TypedDict-based state and
    Pydantic models during the transition period.

    Usage:
        # Convert Pydantic to dict
        model = WorkflowExecutionContextModel(...)
        state_dict = StateAdapter.to_dict(model)

        # Convert dict to Pydantic
        model = StateAdapter.from_dict(state_dict, WorkflowExecutionContextModel)
    """

    @staticmethod
    def to_dict(model: BaseModel) -> Dict[str, Any]:
        """Convert Pydantic model to dict (TypedDict-compatible).

        Args:
            model: Pydantic model instance

        Returns:
            Dict compatible with TypedDict-based code

        Example:
            model = WorkflowExecutionContextModel(workflow_name="test")
            state_dict = StateAdapter.to_dict(model)
            # Can now pass to functions expecting TypedDict
        """
        if hasattr(model, "to_dict"):
            return model.to_dict()
        # Fallback for models without custom to_dict method
        return model.model_dump()

    @staticmethod
    def from_dict(data: Dict[str, Any], model_class: Type[T]) -> T:
        """Create Pydantic model from dict (TypedDict-compatible).

        Args:
            data: Dict from TypedDict-based code
            model_class: Pydantic model class

        Returns:
            Pydantic model instance

        Raises:
            ValidationError: If data fails Pydantic validation

        Example:
            state_dict = {"_workflow_id": "123", "_workflow_name": "test"}
            model = StateAdapter.from_dict(state_dict, WorkflowStateModel)
        """
        if hasattr(model_class, "from_dict"):
            return model_class.from_dict(data)
        # Fallback for models without custom from_dict method
        return model_class(**data)

    @staticmethod
    def to_legacy_format(model: BaseModel) -> Dict[str, Any]:
        """Convert Pydantic model to legacy TypedDict format with underscore prefixes.

        This is the primary method for backward compatibility with existing
        StateGraph and workflow executors that expect TypedDict-like objects.

        Args:
            model: Pydantic model instance

        Returns:
            Dict with underscore-prefixed keys (TypedDict convention)
        """
        return StateAdapter.to_dict(model)

    @staticmethod
    def from_legacy_format(data: Dict[str, Any], model_class: Type[T]) -> T:
        """Create Pydantic model from legacy TypedDict format.

        This is the primary method for migrating existing TypedDict-based
        state to Pydantic models.

        Args:
            data: Dict with underscore-prefixed keys (TypedDict convention)
            model_class: Pydantic model class

        Returns:
            Pydantic model instance

        Raises:
            ValidationError: If data fails Pydantic validation
        """
        return StateAdapter.from_dict(data, model_class)


class WorkflowExecutionContextAdapter:
    """Specialized adapter for WorkflowExecutionContext conversions.

    Provides additional workflow-specific utility methods beyond the
    generic StateAdapter.
    """

    @staticmethod
    def create_initial(
        workflow_id: Optional[str] = None,
        workflow_name: str = "",
        current_node: str = "",
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecutionContextModel:
        """Create initial workflow execution context with sensible defaults.

        Args:
            workflow_id: Optional workflow ID (auto-generated UUID if None)
            workflow_name: Name of the workflow
            current_node: Current node ID
            initial_data: Optional initial data dictionary

        Returns:
            WorkflowExecutionContextModel with defaults applied
        """
        # Only pass workflow_id if provided (let default factory handle None)
        kwargs = {
            "workflow_name": workflow_name,
            "current_node": current_node,
        }
        if workflow_id is not None:
            kwargs["workflow_id"] = workflow_id

        model = WorkflowExecutionContextModel(**kwargs)
        if initial_data:
            model.data.update(initial_data)
        return model

    @staticmethod
    def update_from_legacy(
        model: WorkflowExecutionContextModel,
        legacy_update: Dict[str, Any],
    ) -> WorkflowExecutionContextModel:
        """Update Pydantic model from legacy TypedDict-style update.

        Allows incremental updates from legacy code that modifies
        state dictionaries directly.

        Args:
            model: Existing Pydantic model to update
            legacy_update: Dict with underscore-prefixed keys

        Returns:
            Updated WorkflowExecutionContextModel
        """
        # Extract underscore-prefixed fields
        workflow_id = legacy_update.get("_workflow_id")
        workflow_name = legacy_update.get("_workflow_name")
        current_node = legacy_update.get("_current_node")
        node_results = legacy_update.get("_node_results")
        error = legacy_update.get("_error")
        iteration = legacy_update.get("_iteration")
        visited_nodes = legacy_update.get("_visited_nodes")
        parallel_results = legacy_update.get("_parallel_results")
        hitl_pending = legacy_update.get("_hitl_pending")
        hitl_response = legacy_update.get("_hitl_response")
        as_of_date = legacy_update.get("_as_of_date")
        lookback_periods = legacy_update.get("_lookback_periods")
        include_end_date = legacy_update.get("_include_end_date")
        is_complete = legacy_update.get("_is_complete")
        success = legacy_update.get("_success")

        # Create updated model (Pydantic will validate)
        return WorkflowExecutionContextModel(
            workflow_id=workflow_id if workflow_id else model.workflow_id,
            workflow_name=(workflow_name if workflow_name is not None else model.workflow_name),
            current_node=current_node if current_node else model.current_node,
            node_results=(node_results if node_results is not None else model.node_results),
            error=error if error is not None else model.error,
            iteration=iteration if iteration is not None else model.iteration,
            visited_nodes=(visited_nodes if visited_nodes is not None else model.visited_nodes),
            parallel_results=(
                parallel_results if parallel_results is not None else model.parallel_results
            ),
            hitl_pending=(hitl_pending if hitl_pending is not None else model.hitl_pending),
            hitl_response=(hitl_response if hitl_response is not None else model.hitl_response),
            as_of_date=as_of_date if as_of_date is not None else model.as_of_date,
            lookback_periods=(
                lookback_periods if lookback_periods is not None else model.lookback_periods
            ),
            include_end_date=(
                include_end_date if include_end_date is not None else model.include_end_date
            ),
            is_complete=is_complete if is_complete is not None else model.is_complete,
            success=success if success is not None else model.success,
            data={**model.data, **legacy_update.get("data", {})},
            messages=[*model.messages, *legacy_update.get("messages", [])],
        )


__all__ = [
    "StateAdapter",
    "WorkflowExecutionContextAdapter",
]
