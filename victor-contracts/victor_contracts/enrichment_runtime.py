"""SDK host adapters for framework enrichment helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.enrichment import (
        ANALYSIS_TYPES,
        DATA_PATTERNS,
        DEVOPS_PATTERNS,
        INFRA_TYPES,
        ContextEnrichment,
        EnrichmentContext,
        EnrichmentPriority,
        EnrichmentType,
        FilePatternMatcher,
        KeywordClassifier,
        extract_dotted_paths,
        extract_identifiers,
    )
    from victor.framework.enrichment.file_patterns import CODE_PATTERNS

__all__ = [
    "ANALYSIS_TYPES",
    "ContextEnrichment",
    "DATA_PATTERNS",
    "DEVOPS_PATTERNS",
    "EnrichmentContext",
    "EnrichmentPriority",
    "EnrichmentType",
    "FilePatternMatcher",
    "INFRA_TYPES",
    "KeywordClassifier",
    "extract_dotted_paths",
    "extract_identifiers",
    "CODE_PATTERNS",
]

_LAZY_IMPORTS = {
    "ANALYSIS_TYPES": "victor.framework.enrichment",
    "CODE_PATTERNS": "victor.framework.enrichment.file_patterns",
    "ContextEnrichment": "victor.framework.enrichment",
    "DATA_PATTERNS": "victor.framework.enrichment",
    "DEVOPS_PATTERNS": "victor.framework.enrichment",
    "EnrichmentContext": "victor.framework.enrichment",
    "EnrichmentPriority": "victor.framework.enrichment",
    "EnrichmentType": "victor.framework.enrichment",
    "FilePatternMatcher": "victor.framework.enrichment",
    "INFRA_TYPES": "victor.framework.enrichment",
    "KeywordClassifier": "victor.framework.enrichment",
    "extract_dotted_paths": "victor.framework.enrichment",
    "extract_identifiers": "victor.framework.enrichment",
}


def __getattr__(name: str) -> Any:
    """Resolve enrichment helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'victor_contracts.enrichment_runtime' has no attribute {name!r}"
        )

    module = importlib.import_module(module_name)
    return getattr(module, name)
