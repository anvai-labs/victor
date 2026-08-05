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

"""RL integration for RAG vertical.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.

This module provides:
- RAGRLConfig: Configuration for active learners, task type mappings, and quality thresholds
- RAGRLHooks: Hooks for tracking search quality and synthesis quality
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor_contracts import LearnerType
from victor_contracts.rl import BaseRLConfig
from victor_contracts.constants import ToolNames


@dataclass
class RAGRLConfig(BaseRLConfig):
    """RL configuration for RAG vertical.

    Inherits common RL configuration from BaseRLConfig and extends
    with RAG-specific task types and quality thresholds.

    Configures which learners are active and how they should behave
    for RAG-specific tasks like search, synthesis, and ingestion.
    """

    # RAG uses different active learners (QUALITY_WEIGHTS instead of CONTINUATION_PATIENCE)
    active_learners: List[LearnerType] = field(
        default_factory=lambda: [
            LearnerType.TOOL_SELECTOR,
            LearnerType.GROUNDING_THRESHOLD,
            LearnerType.QUALITY_WEIGHTS,
        ]
    )

    # Uses canonical ToolNames constants and RAG-specific tool names
    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "search": [
                "rag_search",
                "rag_query",
                ToolNames.READ,
            ],
            "ingest": [
                "rag_ingest",
                ToolNames.READ,
                ToolNames.LS,
                ToolNames.WEB_FETCH,
            ],
            "synthesis": [
                "rag_query",
                "rag_search",
            ],
            "management": [
                "rag_list",
                "rag_delete",
                "rag_stats",
            ],
            "exploration": [
                "rag_search",
                "rag_list",
                "rag_stats",
                ToolNames.READ,
            ],
        }
    )

    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "search": 0.80,  # Search relevance threshold
            "synthesis": 0.85,  # Answer quality threshold (higher for factual)
            "ingest": 0.75,  # Ingestion success threshold
            "management": 0.70,  # Management operation threshold
            "exploration": 0.75,  # Exploration quality threshold
        }
    )

    # RAG-specific: lower default patience for RAG tasks (3 instead of 4)
    default_patience: Dict[str, int] = field(
        default_factory=lambda: {
            "anthropic": 3,
            "openai": 3,
            "ollama": 5,
            "google": 3,
        }
    )

    # Methods get_tools_for_task, get_quality_threshold, get_patience,
    # is_learner_active, get_rl_config, __repr__ all inherited


class RAGRLHooks:
    """RL recording hooks for RAG middleware.

    Provides methods to get tool recommendations, patience settings,
    and quality thresholds based on task context.
    """

    def __init__(self, config: Optional[RAGRLConfig] = None):
        """Initialize with optional custom config.

        Args:
            config: Custom RAGRLConfig, or None to use default
        """
        self._config = config or RAGRLConfig()

    @property
    def config(self) -> RAGRLConfig:
        """Get the RL configuration."""
        return self._config

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[str]:
        """Get tool recommendations for a task type.

        Args:
            task_type: Type of task
            available_tools: Optional filter for available tools

        Returns:
            List of recommended tool names
        """
        config_tools = self._config.get_tools_for_task(task_type)
        if available_tools:
            return [t for t in config_tools if t in available_tools]
        return config_tools

    def get_patience_recommendation(self, provider: str, model: str) -> int:
        """Get patience recommendation for provider/model.

        Args:
            provider: LLM provider name
            model: Model name (currently unused, for future model-specific tuning)

        Returns:
            Recommended patience value
        """
        return self._config.get_patience(provider)

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for task type.

        Args:
            task_type: Type of task

        Returns:
            Quality threshold (0.0-1.0)
        """
        return self._config.get_quality_threshold(task_type)

    def __repr__(self) -> str:
        return f"RAGRLHooks(config={self._config})"


# Module-level singletons
_default_config: RAGRLConfig | None = None
_hooks_instance: RAGRLHooks | None = None


def get_default_config() -> RAGRLConfig:
    """Get the default RAG RL configuration singleton.

    Returns:
        RAGRLConfig singleton instance
    """
    global _default_config
    if _default_config is None:
        _default_config = RAGRLConfig()
    return _default_config


def get_rag_rl_hooks() -> RAGRLHooks:
    """Get the RAG RL hooks singleton.

    Returns:
        RAGRLHooks singleton instance
    """
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = RAGRLHooks()
    return _hooks_instance


__all__ = [
    "RAGRLConfig",
    "RAGRLHooks",
    "get_default_config",
    "get_rag_rl_hooks",
]
