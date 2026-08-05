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

"""RAG-specific mode configurations using central registry.

This module registers RAG-specific operational modes with the central
ModeConfigRegistry and exports a registry-based provider for protocol
compatibility.
"""

from __future__ import annotations

from typing import Dict

from victor.framework.extensions import (
    ModeConfigRegistry,
    RegistryBasedModeConfigProvider,
)
from victor_contracts.verticals.mode_config import ModeDefinition

# =============================================================================
# RAG-Specific Modes (Registered with Central Registry)
# =============================================================================

# Vertical-specific modes that extend/override defaults
_RAG_MODES: Dict[str, ModeDefinition] = {
    "bulk_ingest": ModeDefinition(
        name="bulk_ingest",
        tool_budget=50,
        max_iterations=100,
        temperature=0.5,
        description="Bulk document ingestion with high tool budget",
        exploration_multiplier=2.0,
    ),
    "deep_search": ModeDefinition(
        name="deep_search",
        tool_budget=25,
        max_iterations=50,
        temperature=0.6,
        description="Deep search across knowledge base",
        exploration_multiplier=1.5,
    ),
    "synthesis": ModeDefinition(
        name="synthesis",
        tool_budget=15,
        max_iterations=30,
        temperature=0.7,
        description="Document synthesis and summarization",
        exploration_multiplier=1.0,
    ),
}

# RAG-specific task type budgets
_RAG_TASK_BUDGETS: Dict[str, int] = {
    "query": 5,
    "search": 8,
    "ingest": 10,
    "synthesis": 6,
    "maintenance": 15,
    "index": 12,
}


# =============================================================================
# Registration (Called at module load)
# =============================================================================


def _register_rag_modes() -> None:
    """Register RAG-specific modes with the central registry."""
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(
        name="rag",
        modes=_RAG_MODES,
        task_budgets=_RAG_TASK_BUDGETS,
    )


# Register on module import
_register_rag_modes()


# =============================================================================
# Provider (Protocol Compatible)
# =============================================================================


class RAGModeConfigProvider(RegistryBasedModeConfigProvider):
    """Mode configuration provider for RAG vertical.

    Uses the central ModeConfigRegistry for mode lookups,
    ensuring consistency with default modes.
    """

    def __init__(self):
        """Initialize with RAG vertical name."""
        super().__init__(vertical="rag")


# =============================================================================
# Convenience Functions
# =============================================================================


def get_rag_mode_config(mode_name: str) -> ModeDefinition:
    """Get mode configuration for RAG vertical.

    Args:
        mode_name: Name of the mode

    Returns:
        ModeDefinition for the requested mode
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_mode("rag", mode_name)


def get_rag_task_budget(task_type: str) -> int:
    """Get tool budget for a RAG task type.

    Args:
        task_type: Type of task

    Returns:
        Recommended tool budget
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_tool_budget("rag", task_type)


__all__ = [
    "RAGModeConfigProvider",
    "get_rag_mode_config",
    "get_rag_task_budget",
]
