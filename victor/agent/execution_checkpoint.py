# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unified execution checkpoint contract.

This module defines the framework-level envelope that links existing checkpoint
owners together. It does not persist graph, conversation, filesystem, or HITL
state itself; those responsibilities stay with their canonical managers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional
import uuid


class ApprovalState(str, Enum):
    """Approval status associated with an execution checkpoint."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionCheckpoint:
    """Identifier envelope for a resumable execution point.

    Distinct checkpoint managers continue to own their data:
    - ``WorkflowCheckpoint`` owns graph state.
    - ``ConversationCheckpointManager`` owns conversation/session state.
    - ``GitCheckpointManager`` or another filesystem owner owns file snapshots.
    - HITL controllers own approval interruption state.
    """

    id: str
    session_id: str
    graph_checkpoint_id: Optional[str] = None
    conversation_checkpoint_id: Optional[str] = None
    filesystem_checkpoint_id: Optional[str] = None
    triggering_tool_call: Optional[Dict[str, Any]] = None
    approval_state: ApprovalState = ApprovalState.NOT_REQUIRED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        graph_checkpoint_id: Optional[str] = None,
        conversation_checkpoint_id: Optional[str] = None,
        filesystem_checkpoint_id: Optional[str] = None,
        triggering_tool_call: Optional[Mapping[str, Any]] = None,
        approval_state: ApprovalState | str = ApprovalState.NOT_REQUIRED,
        metadata: Optional[Mapping[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> "ExecutionCheckpoint":
        """Create a new execution checkpoint envelope."""
        return cls(
            id=checkpoint_id or f"exec_ckpt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            graph_checkpoint_id=graph_checkpoint_id,
            conversation_checkpoint_id=conversation_checkpoint_id,
            filesystem_checkpoint_id=filesystem_checkpoint_id,
            triggering_tool_call=(dict(triggering_tool_call) if triggering_tool_call else None),
            approval_state=ApprovalState(approval_state),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the checkpoint envelope to primitive values."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "graph_checkpoint_id": self.graph_checkpoint_id,
            "conversation_checkpoint_id": self.conversation_checkpoint_id,
            "filesystem_checkpoint_id": self.filesystem_checkpoint_id,
            "triggering_tool_call": self.triggering_tool_call,
            "approval_state": self.approval_state.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionCheckpoint":
        """Deserialize a checkpoint envelope from primitive values."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if not isinstance(created_at, datetime):
            created_at = datetime.now(timezone.utc)

        triggering_tool_call = data.get("triggering_tool_call")
        return cls(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            graph_checkpoint_id=_optional_str(data.get("graph_checkpoint_id")),
            conversation_checkpoint_id=_optional_str(data.get("conversation_checkpoint_id")),
            filesystem_checkpoint_id=_optional_str(data.get("filesystem_checkpoint_id")),
            triggering_tool_call=(
                dict(triggering_tool_call) if isinstance(triggering_tool_call, Mapping) else None
            ),
            approval_state=ApprovalState(data.get("approval_state", ApprovalState.NOT_REQUIRED)),
            created_at=created_at,
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_trace_metadata(self) -> Dict[str, Any]:
        """Return compact metadata suitable for spans and graph events."""
        tool_call = self.triggering_tool_call or {}
        return {
            "execution_checkpoint_id": self.id,
            "session_id": self.session_id,
            "graph_checkpoint_id": self.graph_checkpoint_id,
            "conversation_checkpoint_id": self.conversation_checkpoint_id,
            "filesystem_checkpoint_id": self.filesystem_checkpoint_id,
            "approval_state": self.approval_state.value,
            "triggering_tool_call_id": tool_call.get("id"),
            "triggering_tool_name": tool_call.get("name"),
            "has_triggering_tool_arguments": bool(tool_call.get("arguments")),
            "metadata": dict(self.metadata),
        }


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def execution_checkpoint_trace_metadata(value: Any) -> Optional[Dict[str, Any]]:
    """Return trace-safe metadata from an execution checkpoint-like value."""
    if value is None:
        return None

    if isinstance(value, ExecutionCheckpoint):
        return value.to_trace_metadata()
    if isinstance(value, Mapping):
        if "execution_checkpoint_id" in value:
            return dict(value)
        if "id" in value and "session_id" in value:
            try:
                return ExecutionCheckpoint.from_dict(value).to_trace_metadata()
            except (TypeError, ValueError, KeyError):
                return dict(value)
        return dict(value)
    if hasattr(value, "to_trace_metadata"):
        metadata = value.to_trace_metadata()
        if isinstance(metadata, Mapping):
            return dict(metadata)
    return None


def normalize_execution_checkpoint_context(
    context: Optional[Mapping[str, Any]],
    *,
    approval_state: ApprovalState | str | None = None,
    graph_checkpoint_id: str | None = None,
) -> Dict[str, Any]:
    """Normalize execution checkpoint values in a trace/audit context.

    The original ``execution_checkpoint`` key is kept serializable when it
    contains an ``ExecutionCheckpoint`` object, while
    ``execution_checkpoint_metadata`` always carries the compact trace shape.
    """
    normalized = dict(context or {})
    checkpoint = normalized.get("execution_checkpoint")
    if checkpoint is None:
        checkpoint = normalized.get("execution_checkpoint_metadata")

    metadata = execution_checkpoint_trace_metadata(checkpoint)
    if not metadata:
        return normalized

    if approval_state is not None:
        metadata["approval_state"] = ApprovalState(approval_state).value
    if graph_checkpoint_id is not None:
        metadata["graph_checkpoint_id"] = graph_checkpoint_id

    normalized["execution_checkpoint_metadata"] = metadata
    if isinstance(checkpoint, ExecutionCheckpoint):
        serialized_checkpoint = checkpoint.to_dict()
        if approval_state is not None:
            serialized_checkpoint["approval_state"] = ApprovalState(approval_state).value
        if graph_checkpoint_id is not None:
            serialized_checkpoint["graph_checkpoint_id"] = graph_checkpoint_id
        normalized["execution_checkpoint"] = serialized_checkpoint
    elif isinstance(normalized.get("execution_checkpoint"), Mapping) and (
        approval_state is not None or graph_checkpoint_id is not None
    ):
        serialized_checkpoint = dict(normalized["execution_checkpoint"])
        if approval_state is not None:
            serialized_checkpoint["approval_state"] = ApprovalState(approval_state).value
        if graph_checkpoint_id is not None:
            serialized_checkpoint["graph_checkpoint_id"] = graph_checkpoint_id
        normalized["execution_checkpoint"] = serialized_checkpoint
    return normalized


__all__ = [
    "ApprovalState",
    "ExecutionCheckpoint",
    "execution_checkpoint_trace_metadata",
    "normalize_execution_checkpoint_context",
]
