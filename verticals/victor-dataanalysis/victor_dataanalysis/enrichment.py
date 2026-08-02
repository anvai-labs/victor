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

"""Data Analysis vertical enrichment strategy.

Provides prompt enrichment using data context such as:
- Database schema information
- Data profile summaries (types, distributions, missing values)
- Query patterns from similar analyses
- Statistical method recommendations

Example:
    from victor_dataanalysis.enrichment import DataAnalysisEnrichmentStrategy

    # Create strategy
    strategy = DataAnalysisEnrichmentStrategy()

    # Register with enrichment service
    enrichment_service.register_strategy("data_analysis", strategy)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Awaitable

from victor_contracts.enrichment_runtime import (
    ContextEnrichment,
    EnrichmentContext,
    EnrichmentPriority,
    EnrichmentType,
    FilePatternMatcher,
    DATA_PATTERNS,
    KeywordClassifier,
    ANALYSIS_TYPES,
)

logger = logging.getLogger(__name__)

# Use framework pattern matcher with DATA_PATTERNS for file detection
_file_pattern_matcher = FilePatternMatcher(DATA_PATTERNS)

# Use framework keyword classifier with ANALYSIS_TYPES for analysis type detection
_analysis_classifier = KeywordClassifier(ANALYSIS_TYPES)


def _detect_analysis_type(prompt: str) -> List[str]:
    """Detect the type of analysis requested from the prompt.

    Uses the framework's KeywordClassifier with ANALYSIS_TYPES for consistent
    keyword-based classification across verticals.

    Args:
        prompt: The prompt text to analyze

    Returns:
        List of detected analysis types
    """
    return _analysis_classifier.classify(prompt)


def _extract_data_references(prompt: str) -> Dict[str, List[str]]:
    """Extract data references from the prompt.

    Args:
        prompt: The prompt text

    Returns:
        Dict with file_mentions, column_mentions, table_mentions
    """
    references: Dict[str, List[str]] = {
        "files": [],
        "columns": [],
        "tables": [],
    }

    # File patterns (.csv, .xlsx, .parquet, .json, .sql)
    file_pattern = r"\b[\w/.-]+\.(?:csv|xlsx|xls|parquet|json|sql|db)\b"
    references["files"] = re.findall(file_pattern, prompt, re.IGNORECASE)

    # Column name patterns (snake_case or quoted)
    col_pattern = r"`([a-z_][a-z0-9_]*)`"
    references["columns"] = re.findall(col_pattern, prompt)

    # Table name patterns (FROM table, table.column)
    table_pattern = r"(?:FROM|JOIN|INTO)\s+(\w+)"
    references["tables"] = re.findall(table_pattern, prompt, re.IGNORECASE)

    return references


class DataAnalysisEnrichmentStrategy:
    """Enrichment strategy for the Data Analysis vertical.

    Provides schema context, statistical method guidance, and
    query patterns for data analysis tasks.

    Attributes:
        schema_lookup_fn: Async function to look up schema info
        max_columns: Maximum columns to include in schema context
    """

    def __init__(
        self,
        schema_lookup_fn: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
        max_columns: int = 20,
    ):
        """Initialize the Data Analysis enrichment strategy.

        Args:
            schema_lookup_fn: Optional async function for schema lookups
            max_columns: Max columns to include in schema (default: 20)
        """
        self._schema_lookup_fn = schema_lookup_fn
        self._max_columns = max_columns

    def set_schema_lookup_fn(
        self,
        fn: Callable[[str], Awaitable[Dict[str, Any]]],
    ) -> None:
        """Set the schema lookup function.

        Args:
            fn: Async function that takes a table/file name and returns schema
        """
        self._schema_lookup_fn = fn

    async def get_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        """Get enrichments for a data analysis prompt.

        Detects analysis type and provides relevant guidance,
        schema context, and method recommendations. Uses framework utilities
        for pattern matching and keyword classification.

        Args:
            prompt: The prompt to enrich
            context: Enrichment context with file/tool mentions

        Returns:
            List of context enrichments
        """
        enrichments: List[ContextEnrichment] = []

        # Detect analysis types using framework KeywordClassifier
        analysis_types = _detect_analysis_type(prompt)

        # Extract data references from prompt text
        data_refs = _extract_data_references(prompt)

        # Also categorize any files from context using framework FilePatternMatcher
        data_file_categories: Dict[str, List[str]] = {}
        if context.file_mentions:
            data_file_categories = _file_pattern_matcher.match(context.file_mentions)

        try:
            # Collect all data sources for schema lookup
            all_data_files = data_refs["files"] + data_refs["tables"]
            # Add categorized data files from context
            for category, files in data_file_categories.items():
                all_data_files.extend(files)

            # Add schema context if we have data references
            if all_data_files:
                schema_enrichment = await self._enrich_from_schema(
                    list(set(all_data_files))  # Deduplicate
                )
                if schema_enrichment:
                    enrichments.append(schema_enrichment)

            # Add method guidance based on analysis type
            for analysis_type in analysis_types:
                method_enrichment = self._build_method_guidance(analysis_type)
                if method_enrichment:
                    enrichments.append(method_enrichment)

            # Add query patterns from tool history
            if context.tool_history:
                history_enrichment = self._enrich_from_tool_history(context.tool_history)
                if history_enrichment:
                    enrichments.append(history_enrichment)

        except Exception as e:
            logger.warning("Error during data analysis enrichment: %s", e)

        logger.debug(
            "Data Analysis enrichment produced %d enrichments for task_type=%s",
            len(enrichments),
            context.task_type,
        )

        return enrichments

    async def _enrich_from_schema(
        self,
        data_sources: List[str],
    ) -> Optional[ContextEnrichment]:
        """Enrich with schema information.

        Args:
            data_sources: List of table/file names to look up

        Returns:
            Enrichment with schema context, or None
        """
        if not self._schema_lookup_fn or not data_sources:
            return None

        schemas = []

        for source in data_sources[:3]:  # Max 3 sources
            try:
                schema = await self._schema_lookup_fn(source)
                if schema:
                    schemas.append((source, schema))
            except Exception as e:
                logger.debug("Schema lookup error for %s: %s", source, e)

        if not schemas:
            return None

        content_parts = ["Data schema context:"]

        for source, schema in schemas:
            content_parts.append(f"\n**{source}**:")

            columns = schema.get("columns", [])
            for col in columns[: self._max_columns]:
                col_name = col.get("name", "unknown")
                col_type = col.get("type", "unknown")
                nullable = " (nullable)" if col.get("nullable") else ""
                content_parts.append(f"- `{col_name}`: {col_type}{nullable}")

            if len(columns) > self._max_columns:
                content_parts.append(f"  ... and {len(columns) - self._max_columns} more columns")

        return ContextEnrichment(
            type=EnrichmentType.SCHEMA,
            content="\n".join(content_parts),
            priority=EnrichmentPriority.HIGH,
            source="schema_lookup",
            metadata={"sources_count": len(schemas)},
        )

    def _build_method_guidance(
        self,
        analysis_type: str,
    ) -> Optional[ContextEnrichment]:
        """Build method guidance for an analysis type.

        Args:
            analysis_type: Type of analysis detected

        Returns:
            Enrichment with method recommendations
        """
        guidance = {
            "correlation": """Correlation Analysis Guidelines:
- Use Pearson for linear relationships (continuous data)
- Use Spearman for monotonic relationships or ordinal data
- Check for outliers that may skew results
- Consider partial correlation to control for confounders
- Correlation matrix: df.corr() with seaborn heatmap""",
            "regression": """Regression Analysis Guidelines:
- Check assumptions: linearity, normality of residuals, homoscedasticity
- Use train/test split or cross-validation
- Check for multicollinearity (VIF > 5 is concerning)
- Consider regularization (Ridge/Lasso) for many features
- Report R², RMSE, and residual plots""",
            "clustering": """Clustering Guidelines:
- Scale features first (StandardScaler or MinMaxScaler)
- Use elbow method or silhouette score to determine k
- Consider different algorithms: K-means (spherical), DBSCAN (density), hierarchical
- Use PCA/t-SNE for visualization of high-dimensional clusters
- Profile clusters with descriptive statistics""",
            "classification": """Classification Guidelines:
- Handle class imbalance (oversampling, undersampling, SMOTE)
- Use stratified train/test split
- Consider multiple metrics: accuracy, precision, recall, F1, AUC-ROC
- Use confusion matrix for detailed error analysis
- Cross-validation for robust evaluation""",
            "time_series": """Time Series Guidelines:
- Ensure datetime index is sorted and has consistent frequency
- Check for stationarity (ADF test)
- Decompose into trend, seasonal, and residual components
- Handle missing values with interpolation
- Consider ARIMA, Prophet, or exponential smoothing""",
            "statistical_test": """Statistical Testing Guidelines:
- State null and alternative hypotheses clearly
- Check assumptions for chosen test
- Report test statistic, p-value, and effect size
- Use appropriate test: t-test (means), chi-square (categorical), ANOVA (multiple groups)
- Consider multiple testing correction if many comparisons""",
            "visualization": """Visualization Guidelines:
- Choose appropriate chart type for data and message
- Use clear labels, titles, and legends
- Consider colorblind-friendly palettes (viridis, cividis)
- Add context with annotations where helpful
- Save in high resolution: plt.savefig('fig.png', dpi=300)""",
            "profiling": """Data Profiling Guidelines:
- Check shape and dtypes: df.info()
- Summary statistics: df.describe()
- Missing values: df.isnull().sum()
- Unique values: df.nunique()
- Distribution checks: histograms, value_counts()""",
        }

        content = guidance.get(analysis_type)
        if not content:
            return None

        return ContextEnrichment(
            type=EnrichmentType.PROJECT_CONTEXT,
            content=content,
            priority=EnrichmentPriority.NORMAL,
            source=f"method_guidance_{analysis_type}",
            metadata={"analysis_type": analysis_type},
        )

    def _enrich_from_tool_history(
        self,
        tool_history: List[Dict[str, Any]],
    ) -> Optional[ContextEnrichment]:
        """Enrich from previous Python/SQL tool results.

        Args:
            tool_history: List of recent tool calls

        Returns:
            Enrichment with successful query patterns, or None
        """
        successful_queries = []

        for call in tool_history[-15:]:  # Last 15 calls
            tool_name = call.get("tool", "")

            if tool_name in ("python", "execute_python", "sql_query"):
                result = call.get("result", {})
                if isinstance(result, dict) and result.get("success"):
                    code = call.get("arguments", {}).get("code", "")
                    query = call.get("arguments", {}).get("query", "")
                    content = code or query

                    # Filter for data analysis patterns
                    if content and any(
                        kw in content.lower()
                        for kw in [
                            "pandas",
                            "numpy",
                            "df.",
                            "plt.",
                            "seaborn",
                            "select",
                            "from",
                            "where",
                            "group by",
                        ]
                    ):
                        successful_queries.append(content[:200])

        if not successful_queries:
            return None

        content_parts = ["Successful queries/code from this session:"]
        for query in successful_queries[-3:]:  # Last 3 queries
            content_parts.append(f"```\n{query}\n```")

        return ContextEnrichment(
            type=EnrichmentType.TOOL_HISTORY,
            content="\n".join(content_parts),
            priority=EnrichmentPriority.LOW,
            source="query_history",
            metadata={"queries_count": len(successful_queries)},
        )

    def get_priority(self) -> int:
        """Get strategy priority.

        Returns:
            Priority value (50 = normal)
        """
        return 50

    def get_token_allocation(self) -> float:
        """Get token budget allocation.

        Returns:
            Fraction of token budget (0.35 = 35%)
        """
        return 0.35


__all__ = [
    "DataAnalysisEnrichmentStrategy",
]
