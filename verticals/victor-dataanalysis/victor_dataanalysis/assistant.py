"""Data Analysis Assistant - Complete vertical for data exploration and insights.

Competitive positioning: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
"""

from typing import ClassVar, Dict, List

from victor_contracts.capabilities import FileOperationsCapability
from victor_contracts.verticals.protocols.base import VerticalBase
from victor_contracts.core.types import StageDefinition
from victor_contracts.constants.tool_names import ToolNames

# Contract-owned capabilities are imported directly; no framework runtime import is needed.


class DataAnalysisAssistant(VerticalBase):
    """Data analysis assistant for exploration, visualization, and insights.

    Competitive with: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
    """

    name = "dataanalysis"
    description = "Data exploration, statistical analysis, visualization, and ML insights"
    version = "1.0.0"

    # Phase 3: Framework file operations capability (lazy-loaded)
    _file_ops: ClassVar[FileOperationsCapability | None] = None

    @classmethod
    def _get_file_ops(cls) -> FileOperationsCapability:
        if cls._file_ops is None:
            cls._file_ops = FileOperationsCapability()
        return cls._file_ops

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for data analysis tasks.

        Phase 3: Uses framework FileOperationsCapability for common file operations
        to reduce code duplication and maintain consistency across verticals.

        Uses SDK-owned canonical tool identifiers.
        """
        # Start with framework file operations (read, write, edit, grep)
        tools = cls._get_file_ops().get_tool_list()

        # Add data analysis-specific tools
        tools.extend(
            [
                # Directory listing for data file exploration
                ToolNames.LS,  # list_directory → ls
                # Python/Shell execution for analysis
                ToolNames.SHELL,  # bash → shell (for running Python scripts)
                # Code generation and search
                ToolNames.CODE_SEARCH,  # Semantic code search
                ToolNames.OVERVIEW,  # codebase_overview → overview
                ToolNames.GRAPH,  # Code graph analysis (PageRank, dependencies)
                # Web for datasets and documentation
                ToolNames.WEB_SEARCH,  # Web search (internet search)
                ToolNames.WEB_FETCH,  # Fetch URL content
            ]
        )

        return tools

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for data analysis tasks."""
        return cls._get_system_prompt()

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get Data Analysis-specific stage definitions.

        Uses SDK-owned canonical tool identifiers.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the data and analysis goals",
                tools={ToolNames.READ, ToolNames.LS, ToolNames.OVERVIEW},
                keywords=["what", "data", "analyze", "understand", "explore"],
                next_stages={"DATA_LOADING", "EXPLORATION"},
            ),
            "DATA_LOADING": StageDefinition(
                name="DATA_LOADING",
                description="Loading and validating data files",
                tools={ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE},
                keywords=["load", "import", "read", "open", "fetch"],
                next_stages={"EXPLORATION", "CLEANING"},
            ),
            "EXPLORATION": StageDefinition(
                name="EXPLORATION",
                description="Exploratory data analysis and profiling",
                tools={ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE},
                keywords=["explore", "profile", "describe", "summary", "statistics"],
                next_stages={"CLEANING", "ANALYSIS"},
            ),
            "CLEANING": StageDefinition(
                name="CLEANING",
                description="Data cleaning and transformation",
                tools={ToolNames.SHELL, ToolNames.WRITE, ToolNames.EDIT},
                keywords=["clean", "transform", "fix", "handle", "remove"],
                next_stages={"ANALYSIS", "EXPLORATION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Statistical analysis and modeling",
                tools={ToolNames.SHELL, ToolNames.WRITE, ToolNames.READ},
                keywords=["analyze", "model", "correlate", "test", "predict"],
                next_stages={"VISUALIZATION", "REPORTING"},
            ),
            "VISUALIZATION": StageDefinition(
                name="VISUALIZATION",
                description="Creating charts and visualizations",
                tools={ToolNames.SHELL, ToolNames.WRITE},
                keywords=["plot", "chart", "visualize", "graph", "figure"],
                next_stages={"REPORTING", "ANALYSIS"},
            ),
            "REPORTING": StageDefinition(
                name="REPORTING",
                description="Generating insights and reports",
                tools={ToolNames.WRITE, ToolNames.EDIT, ToolNames.READ},
                keywords=["report", "summarize", "document", "present"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Finalizing analysis deliverables",
                tools={ToolNames.WRITE, ToolNames.READ},
                keywords=["done", "complete", "finish", "final"],
                next_stages=set(),
            ),
        }

    @classmethod
    def _get_system_prompt(cls) -> str:
        return (
            "You are a data analysis assistant specializing in exploration, "
            "statistics, and visualization."
            """

## Core Capabilities

1. **Data Loading**: CSV, Excel, JSON, Parquet, SQL databases
2. **Exploration**: Profiling, summary statistics, distribution analysis
3. **Cleaning**: Missing values, outliers, type conversion, normalization
4. **Analysis**: Correlation, regression, hypothesis testing, clustering
5. **Visualization**: matplotlib, seaborn, plotly for charts and dashboards
6. **ML**: scikit-learn for classification, regression, clustering

## Analysis Workflow

1. **LOAD**: Read data, check structure, identify types
2. **EXPLORE**: Summary stats, distributions, missing values
3. **CLEAN**: Handle nulls, outliers, type issues
4. **ANALYZE**: Apply statistical methods, test hypotheses
5. **VISUALIZE**: Create informative charts
6. **REPORT**: Summarize insights with evidence

## Code Standards

- Always use pandas for data manipulation
- Include comments explaining methodology
- Handle missing data explicitly
- Use descriptive variable names
- Save intermediate results for reproducibility

## Output Format

When presenting analysis:
1. Start with data overview (shape, types, missing)
2. Show key statistics with context
3. Include visualizations with captions
4. State insights with supporting evidence
5. Note limitations and assumptions
6. Provide reproducible code

## Privacy and Ethics

- Never expose personally identifiable information (PII)
- Anonymize sensitive columns before analysis
- Note potential biases in data
- Be transparent about limitations
"""
        )

    @classmethod
    def get_capability_provider(cls):
        """Get the capability provider for DataAnalysisAssistant.

        Returns:
            DataAnalysisCapabilityProvider instance
        """
        from victor_dataanalysis.capabilities import DataAnalysisCapabilityProvider

        return DataAnalysisCapabilityProvider()

    # =========================================================================
    # New Framework Integrations (Workflows, RL, Teams)
    # =========================================================================
