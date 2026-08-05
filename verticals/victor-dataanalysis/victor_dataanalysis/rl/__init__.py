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

"""RL integration for Data Analysis vertical.

Uses canonical tool names from ToolNames to ensure consistent naming
across RL Q-values, workflow patterns, and vertical configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor_contracts import LearnerType
from victor_contracts.rl import BaseRLConfig
from victor_contracts.constants import ToolNames


@dataclass
class DataAnalysisRLConfig(BaseRLConfig):
    """RL configuration for Data Analysis vertical.

    Inherits common RL configuration from BaseRLConfig and extends
    with data analysis-specific task types and quality thresholds.

    Configures which RL learners to use and their parameters
    for data analysis tasks.
    """

    # active_learners inherited from BaseRLConfig
    # default_patience inherited from BaseRLConfig

    # Uses canonical ToolNames constants for consistency
    task_type_mappings: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "eda": [ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE, ToolNames.LS],
            "cleaning": [ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE],
            "visualization": [ToolNames.SHELL, ToolNames.WRITE],
            "statistics": [ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE],
            "ml": [ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE],
            "profiling": [ToolNames.READ, ToolNames.SHELL, ToolNames.LS],
            "reporting": [ToolNames.WRITE, ToolNames.READ, ToolNames.SHELL],
        }
    )

    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "eda": 0.75,  # Exploratory, lower bar
            "cleaning": 0.85,  # Data quality matters
            "visualization": 0.80,
            "statistics": 0.90,  # High bar for statistical correctness
            "ml": 0.85,
            "profiling": 0.75,
            "reporting": 0.80,
        }
    )

    # default_patience inherited from BaseRLConfig
    # Methods get_tools_for_task, get_quality_threshold, get_patience,
    # is_learner_active, get_rl_config, __repr__ all inherited

    # Data analysis often needs larger outputs for code
    preferred_output_length: Dict[str, str] = field(
        default_factory=lambda: {
            "eda": "medium",
            "cleaning": "medium",
            "visualization": "long",  # Charts need more code
            "statistics": "medium",
            "ml": "long",  # ML pipelines are verbose
            "profiling": "short",
            "reporting": "long",
        }
    )

    # DataAnalysis-specific method
    def get_preferred_output_length(self, task_type: str) -> str:
        """Get preferred output length for task type."""
        return self.preferred_output_length.get(task_type.lower(), "medium")


class DataAnalysisRLHooks:
    """RL recording hooks for Data Analysis middleware.

    Provides hooks for recording RL training data during data analysis tasks.
    """

    def __init__(self, config: Optional[DataAnalysisRLConfig] = None):
        self._config = config or DataAnalysisRLConfig()

    @property
    def config(self) -> DataAnalysisRLConfig:
        return self._config

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[str]:
        """Get tool recommendations for a task type."""
        config_tools = self._config.get_tools_for_task(task_type)
        if available_tools:
            return [t for t in config_tools if t in available_tools]
        return config_tools

    def get_patience_recommendation(self, provider: str, model: str) -> int:
        """Get patience recommendation for provider/model."""
        return self._config.get_patience(provider)

    def get_quality_threshold(self, task_type: str) -> float:
        """Get quality threshold for task type."""
        return self._config.get_quality_threshold(task_type)

    def should_include_code(self, task_type: str) -> bool:
        """Check if code should be included in output."""
        # Data analysis almost always needs code
        return True

    def get_preferred_libraries(self, task_type: str) -> List[str]:
        """Get preferred Python libraries for task type."""
        libraries = {
            "eda": ["pandas", "numpy", "seaborn", "matplotlib"],
            "cleaning": ["pandas", "numpy"],
            "visualization": ["matplotlib", "seaborn", "plotly"],
            "statistics": ["scipy", "statsmodels", "pandas"],
            "ml": ["scikit-learn", "pandas", "numpy"],
            "profiling": ["pandas", "ydata-profiling"],
            "reporting": ["pandas", "matplotlib"],
        }
        return libraries.get(task_type.lower(), ["pandas", "numpy"])

    def __repr__(self) -> str:
        return f"DataAnalysisRLHooks(config={self._config})"


# Module-level singletons
_default_config: DataAnalysisRLConfig | None = None
_hooks_instance: DataAnalysisRLHooks | None = None


def get_default_config() -> DataAnalysisRLConfig:
    """Get default data analysis RL configuration."""
    global _default_config
    if _default_config is None:
        _default_config = DataAnalysisRLConfig()
    return _default_config


def get_data_analysis_rl_hooks() -> DataAnalysisRLHooks:
    """Get data analysis RL hooks instance."""
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = DataAnalysisRLHooks()
    return _hooks_instance


__all__ = [
    "DataAnalysisRLConfig",
    "DataAnalysisRLHooks",
    "get_default_config",
    "get_data_analysis_rl_hooks",
]
