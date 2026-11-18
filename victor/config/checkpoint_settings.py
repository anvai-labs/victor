"""State checkpointing for time-travel debugging."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CheckpointSettings(BaseModel):
    """State checkpointing for time-travel debugging.

    The ``workflow_checkpoint_*`` group is FEP-0047/FEP-0050: durable-by-default
    graph checkpointing at the composition roots, project-scoped by default, and
    opt-out-able per deployment.
    """

    checkpoint_enabled: bool = True
    checkpoint_auto_interval: int = 5
    checkpoint_max_per_session: int = 50
    checkpoint_compression_enabled: bool = True
    checkpoint_compression_threshold: int = 1024
    # FEP-0047: durable-by-default workflow (graph) checkpoints.
    workflow_checkpoints_enabled: bool = True
    workflow_checkpoint_max_per_thread: int = Field(
        default=20, ge=1, description="Per-thread retention applied on every save"
    )
    workflow_checkpoint_retention_days: int = Field(
        default=7, ge=0, description="TTL sweep for wf:: rows (0 disables the sweep)"
    )
    workflow_checkpoint_db_path: Optional[str] = Field(
        default=None,
        description=(
            "Explicit contention/portability override; None uses the canonical "
            "project .victor/project.db"
        ),
    )
