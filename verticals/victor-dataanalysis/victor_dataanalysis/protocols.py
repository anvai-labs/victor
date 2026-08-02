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

"""Victor extension protocol implementations for victor-dataanalysis.

This module provides protocol implementations that can be discovered via
the Victor extension entry point system, enabling the data analysis vertical to
register capabilities with the framework without direct dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Import victor-contracts protocols (NO runtime dependency on victor-ai!)
from victor_contracts.verticals.protocols import (
    ToolProvider,
    ToolSelectionStrategy,
    SafetyProvider,
    PromptProvider,
    WorkflowProvider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Provider
# =============================================================================


class DataAnalysisToolProvider(ToolProvider):
    """Tool provider for Data Analysis vertical.

    Provides the list of tools available to the Data Analysis assistant.
    """

    def get_tools(self) -> List[str]:
        """Return list of tool names for Data Analysis vertical."""
        return [
            # Core filesystem tools
            "read",
            "write",
            "grep",
            "ls",
            # Data loading tools
            "csv_read",
            "excel_read",
            "json_read",
            "database_query",
            "api_fetch",
            # Data processing tools
            "data_clean",
            "data_transform",
            "data_aggregate",
            "data_merge",
            # Statistical tools
            "statistics_describe",
            "statistics_correlation",
            "statistics_regression",
            "statistics_test",
            # Visualization tools
            "plot_line",
            "plot_scatter",
            "plot_bar",
            "plot_histogram",
            "plot_heatmap",
            # ML tools
            "ml_train",
            "ml_predict",
            "ml_evaluate",
            # Data export tools
            "data_export_csv",
            "data_export_json",
            "report_generate",
        ]


class DataAnalysisToolSelectionStrategy(ToolSelectionStrategy):
    """Stage-aware tool selection for Data Analysis tasks."""

    def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]:
        """Return optimized tools for given stage and task type."""
        stage_tools: Dict[str, List[str]] = {
            "load": ["csv_read", "excel_read", "json_read", "database_query"],
            "clean": ["data_clean", "read", "write"],
            "explore": ["statistics_describe", "plot_histogram", "plot_scatter"],
            "analyze": ["statistics_correlation", "statistics_regression", "ml_train"],
            "visualize": ["plot_line", "plot_bar", "plot_heatmap"],
            "report": ["report_generate", "data_export_csv", "write"],
        }

        return stage_tools.get(stage, ["read", "csv_read", "statistics_describe"])


# =============================================================================
# Safety Provider
# =============================================================================


class DataAnalysisSafetyProvider(SafetyProvider):
    """Safety provider for Data Analysis vertical.

    Provides data analysis-specific safety patterns.
    """

    def __init__(self):
        self._dangerous_patterns = [
            # Database dangerous commands
            {"pattern": "DELETE FROM", "description": "Delete database records"},
            {"pattern": "DROP TABLE", "description": "Drop database table"},
            {"pattern": "TRUNCATE", "description": "Truncate table"},
            # File operations
            {"pattern": "data_export --force", "description": "Force overwrite export"},
        ]

    def get_extensions(self) -> List[Any]:
        """Return safety extensions for Data Analysis."""
        return []

    def get_bash_patterns(self) -> List[Any]:
        """Return bash command patterns to monitor."""
        return self._dangerous_patterns

    def get_file_patterns(self) -> List[Any]:
        """Return file operation patterns to monitor."""
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Return tool-specific restrictions."""
        return {
            "database_query": ["DELETE", "DROP", "TRUNCATE"],
            "data_export_csv": ["--force"],
        }


# =============================================================================
# Prompt Provider
# =============================================================================


class DataAnalysisPromptProvider(PromptProvider):
    """Prompt provider for Data Analysis vertical.

    Provides system prompt sections for data analysis tasks.
    """

    def get_system_prompt_sections(self) -> Dict[str, str]:
        """Return system prompt sections."""
        return {
            "role": (
                "You are a Data Analysis assistant specializing in data exploration, "
                "statistical analysis, and visualization."
            ),
            "expertise": (
                "You have expertise in pandas, numpy, statistical methods, machine "
                "learning, and data visualization."
            ),
            "methodology": (
                "Follow systematic data analysis: load data, clean data, explore "
                "patterns, analyze statistically, visualize results."
            ),
            "best_practices": (
                "Always verify data quality before analysis. Use appropriate "
                "statistical tests. Create clear visualizations."
            ),
            "communication": (
                "Explain findings clearly with supporting statistics and " "visualizations."
            ),
        }

    def get_task_type_hints(self) -> Dict[str, Any]:
        """Return task type hints for Data Analysis."""
        return {
            "eda": {
                "hint": "[EDA] Exploratory Data Analysis: load, clean, explore, visualize.",
                "tool_budget": 15,
            },
            "statistics": {
                "hint": "[STATISTICS] Perform statistical analysis and hypothesis testing.",
                "tool_budget": 12,
            },
            "ml": {
                "hint": "[ML] Train, evaluate, and use machine learning models.",
                "tool_budget": 20,
            },
            "visualization": {
                "hint": "[VISUALIZE] Create visualizations to communicate insights.",
                "tool_budget": 10,
            },
        }

    def get_prompt_contributors(self) -> List[Any]:
        """Return prompt contributors for Data Analysis."""
        return []


# =============================================================================
# Workflow Provider
# =============================================================================


class DataAnalysisWorkflowProvider(WorkflowProvider):
    """Workflow provider for Data Analysis vertical.

    Provides data analysis-specific workflow definitions.
    """

    def get_workflows(self) -> Dict[str, Any]:
        """Return workflow specifications."""
        return {
            "exploratory_analysis": {
                "name": "Exploratory Data Analysis",
                "description": "Load, clean, explore, and visualize data",
                "stages": ["load", "clean", "explore", "visualize"],
            },
            "statistical_analysis": {
                "name": "Statistical Analysis",
                "description": "Perform statistical tests and modeling",
                "stages": ["load", "clean", "analyze", "report"],
            },
            "ml_pipeline": {
                "name": "Machine Learning Pipeline",
                "description": "Train and evaluate ML models",
                "stages": ["load", "clean", "analyze", "report"],
            },
        }

    def get_workflow(self, name: str) -> Optional[Any]:
        """Get a specific workflow by name."""
        return self.get_workflows().get(name)

    def list_workflows(self) -> List[str]:
        """List available workflow names."""
        return list(self.get_workflows().keys())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DataAnalysisToolProvider",
    "DataAnalysisToolSelectionStrategy",
    "DataAnalysisSafetyProvider",
    "DataAnalysisPromptProvider",
    "DataAnalysisWorkflowProvider",
]
