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

"""Regression tests for the data analysis enrichment module.

The module imports DATA_PATTERNS and ANALYSIS_TYPES through the
victor_contracts.enrichment_runtime bridge; these tests guard against the
bridge dropping symbols the vertical depends on (import-break regression).
"""

from victor_contracts.enrichment_runtime import (
    ANALYSIS_TYPES,
    DATA_PATTERNS,
    EnrichmentContext,
)

from victor_dataanalysis.enrichment import (
    DataAnalysisEnrichmentStrategy,
    _detect_analysis_type,
    _extract_data_references,
)


def test_bridge_exposes_data_patterns_and_analysis_types() -> None:
    """The contracts bridge must resolve the framework enrichment tables."""
    assert isinstance(DATA_PATTERNS, dict) and DATA_PATTERNS
    assert isinstance(ANALYSIS_TYPES, dict) and ANALYSIS_TYPES
    assert all(isinstance(patterns, list) for patterns in DATA_PATTERNS.values())
    assert all(isinstance(keywords, list) for keywords in ANALYSIS_TYPES.values())


def test_detect_analysis_type_uses_analysis_types_keywords() -> None:
    detected = _detect_analysis_type("Run a correlation analysis on these columns")
    assert "correlation" in detected
    assert _detect_analysis_type("hello there") == []


def test_extract_data_references() -> None:
    refs = _extract_data_references("Load sales.csv and SELECT * FROM orders using `unit_price`")
    assert "sales.csv" in refs["files"]
    assert "orders" in refs["tables"]
    assert "unit_price" in refs["columns"]


async def test_strategy_get_enrichments_smoke() -> None:
    """The strategy runs end-to-end with the bridge-resolved tables."""
    strategy = DataAnalysisEnrichmentStrategy()
    context = EnrichmentContext(file_mentions=["data/sales.csv"])

    enrichments = await strategy.get_enrichments("Run a regression on sales.csv", context)

    assert isinstance(enrichments, list)
    for enrichment in enrichments:
        assert enrichment.content
