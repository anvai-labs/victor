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
from victor_contracts.verticals import (
    make_pattern_safety_provider,
    make_prompt_provider,
    make_stage_tool_strategy,
    make_tool_provider,
    make_workflow_provider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Provider
# =============================================================================


DATA_ANALYSIS_TOOLS = [
    "read",
    "write",
    "grep",
    "ls",
    "csv_read",
    "excel_read",
    "json_read",
    "database_query",
    "api_fetch",
    "data_clean",
    "data_transform",
    "data_aggregate",
    "data_merge",
    "statistics_describe",
    "statistics_correlation",
    "statistics_regression",
    "statistics_test",
    "plot_line",
    "plot_scatter",
    "plot_bar",
    "plot_histogram",
    "plot_heatmap",
    "ml_train",
    "ml_predict",
    "ml_evaluate",
    "data_export_csv",
    "data_export_json",
    "report_generate",
]

DATA_ANALYSIS_STAGE_TOOLS = {
    "load": ["csv_read", "excel_read", "json_read", "database_query"],
    "clean": ["data_clean", "read", "write"],
    "explore": ["statistics_describe", "plot_histogram", "plot_scatter"],
    "analyze": ["statistics_correlation", "statistics_regression", "ml_train"],
    "visualize": ["plot_line", "plot_bar", "plot_heatmap"],
    "report": ["report_generate", "data_export_csv", "write"],
}

DATA_ANALYSIS_DANGEROUS_PATTERNS = [
    {"pattern": "DELETE FROM", "description": "Delete database records"},
    {"pattern": "DROP TABLE", "description": "Drop database table"},
    {"pattern": "TRUNCATE", "description": "Truncate table"},
    {"pattern": "data_export --force", "description": "Force overwrite export"},
]

DATA_ANALYSIS_TOOL_RESTRICTIONS = {
    "database_query": ["DELETE", "DROP", "TRUNCATE"],
    "data_export_csv": ["--force"],
}

DATA_ANALYSIS_PROMPT_SECTIONS = {
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
    "communication": ("Explain findings clearly with supporting statistics and " "visualizations."),
}

DATA_ANALYSIS_TASK_TYPE_HINTS = {
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

DATA_ANALYSIS_WORKFLOWS = {
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

DataAnalysisToolProvider = make_tool_provider(
    "DataAnalysisToolProvider", DATA_ANALYSIS_TOOLS, "Tool provider for the data-analysis vertical."
)
DataAnalysisToolSelectionStrategy = make_stage_tool_strategy(
    "DataAnalysisToolSelectionStrategy",
    DATA_ANALYSIS_STAGE_TOOLS,
    ["read", "csv_read", "statistics_describe"],
    "Stage-aware tool selection for data-analysis tasks.",
)
DataAnalysisSafetyProvider = make_pattern_safety_provider(
    "DataAnalysisSafetyProvider",
    DATA_ANALYSIS_DANGEROUS_PATTERNS,
    DATA_ANALYSIS_TOOL_RESTRICTIONS,
    doc="Safety provider for data-analysis operations.",
)
DataAnalysisPromptProvider = make_prompt_provider(
    "DataAnalysisPromptProvider",
    DATA_ANALYSIS_PROMPT_SECTIONS,
    DATA_ANALYSIS_TASK_TYPE_HINTS,
    "Prompt provider for data-analysis tasks.",
)
DataAnalysisWorkflowProvider = make_workflow_provider(
    "DataAnalysisWorkflowProvider",
    DATA_ANALYSIS_WORKFLOWS,
    "Workflow provider for the data-analysis vertical.",
)


__all__ = [
    "DataAnalysisToolProvider",
    "DataAnalysisToolSelectionStrategy",
    "DataAnalysisSafetyProvider",
    "DataAnalysisPromptProvider",
    "DataAnalysisWorkflowProvider",
]
