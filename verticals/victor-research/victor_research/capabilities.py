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

"""Dynamic capability definitions for the Research vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the Research vertical with custom functionality.

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor_research.capabilities")

    # Or use directly
    from victor_research.capabilities import (
        get_research_capabilities,
        ResearchCapabilityProvider,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from victor_contracts.capabilities import (
    BaseCapabilityProvider,
    CapabilityEntry,
    CapabilityMetadata,
    CapabilityType,
    OrchestratorCapability,
    build_capability_loader,
    capability,
    load_capability_config,
    store_capability_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Config Helpers (P1: Framework CapabilityConfigService Pilot)
# =============================================================================


_SOURCE_VERIFICATION_DEFAULTS: Dict[str, Any] = {
    "min_credibility": 0.7,
    "min_source_count": 3,
    "require_diverse_sources": True,
    "validate_urls": True,
}
_CITATION_DEFAULTS: Dict[str, Any] = {
    "default_style": "apa",
    "require_urls": True,
    "include_authors": True,
    "include_dates": True,
}
_RESEARCH_QUALITY_DEFAULTS: Dict[str, Any] = {
    "min_coverage_score": 0.75,
    "min_source_diversity": 2,
    "check_recency": True,
    "max_source_age_days": 365,
}
_LITERATURE_DEFAULTS: Dict[str, Any] = {
    "min_relevance_score": 0.6,
    "weight_citation_count": True,
    "prefer_recent_papers": True,
    "recent_paper_years": 5,
}
_FACT_CHECKING_DEFAULTS: Dict[str, Any] = {
    "min_confidence_threshold": 0.5,
    "require_multiple_sources": True,
    "min_source_count_for_claim": 2,
    "track_supporting_refuting": True,
}

# =============================================================================
# Capability Handlers
# =============================================================================


def configure_source_verification(
    orchestrator: Any,
    *,
    min_credibility: float = 0.7,
    min_source_count: int = 3,
    require_diverse_sources: bool = True,
    validate_urls: bool = True,
) -> None:
    """Configure source verification rules for the orchestrator.

    This capability configures how the research assistant validates
    and assesses the credibility of sources.

    Args:
        orchestrator: Target orchestrator
        min_credibility: Minimum average credibility score (0-1)
        min_source_count: Minimum number of sources required
        require_diverse_sources: Require source diversity (domains, types)
        validate_urls: Validate URL accessibility and format
    """
    store_capability_config(
        orchestrator,
        "source_verification_config",
        {
            "min_credibility": min_credibility,
            "min_source_count": min_source_count,
            "require_diverse_sources": require_diverse_sources,
            "validate_urls": validate_urls,
        },
    )

    logger.info(
        f"Configured source verification: credibility>={min_credibility:.0%}, "
        f"min_sources={min_source_count}"
    )


def get_source_verification(orchestrator: Any) -> Dict[str, Any]:
    """Get current source verification configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Source verification configuration dict
    """
    return load_capability_config(
        orchestrator, "source_verification_config", _SOURCE_VERIFICATION_DEFAULTS
    )


def configure_citation_management(
    orchestrator: Any,
    *,
    default_style: str = "apa",
    require_urls: bool = True,
    include_authors: bool = True,
    include_dates: bool = True,
) -> None:
    """Configure citation and bibliography management.

    Args:
        orchestrator: Target orchestrator
        default_style: Citation style (apa, mla, chicago, harvard)
        require_urls: Require URLs for web sources
        include_authors: Include author names in citations
        include_dates: Include publication dates
    """
    store_capability_config(
        orchestrator,
        "citation_config",
        {
            "default_style": default_style,
            "require_urls": require_urls,
            "include_authors": include_authors,
            "include_dates": include_dates,
        },
    )

    logger.info(f"Configured citation management: style={default_style}")


def get_citation_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current citation configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Citation configuration dict
    """
    return load_capability_config(orchestrator, "citation_config", _CITATION_DEFAULTS)


def configure_research_quality(
    orchestrator: Any,
    *,
    min_coverage_score: float = 0.75,
    min_source_diversity: int = 2,
    check_recency: bool = True,
    max_source_age_days: Optional[int] = 365,
) -> None:
    """Configure research quality standards.

    Args:
        orchestrator: Target orchestrator
        min_coverage_score: Minimum coverage score (0-1)
        min_source_diversity: Minimum number of source types (web, academic, code)
        check_recency: Check source recency for time-sensitive topics
        max_source_age_days: Maximum acceptable source age in days
    """
    store_capability_config(
        orchestrator,
        "research_quality_config",
        {
            "min_coverage_score": min_coverage_score,
            "min_source_diversity": min_source_diversity,
            "check_recency": check_recency,
            "max_source_age_days": max_source_age_days,
        },
    )

    logger.info(
        f"Configured research quality: coverage>={min_coverage_score:.0%}, "
        f"diversity>={min_source_diversity}"
    )


def get_research_quality(orchestrator: Any) -> Dict[str, Any]:
    """Get current research quality configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Research quality configuration dict
    """
    return load_capability_config(
        orchestrator, "research_quality_config", _RESEARCH_QUALITY_DEFAULTS
    )


def configure_literature_analysis(
    orchestrator: Any,
    *,
    min_relevance_score: float = 0.6,
    weight_citation_count: bool = True,
    prefer_recent_papers: bool = True,
    recent_paper_years: int = 5,
) -> None:
    """Configure literature and academic paper analysis.

    Args:
        orchestrator: Target orchestrator
        min_relevance_score: Minimum paper relevance score (0-1)
        weight_citation_count: Consider citation count in relevance
        prefer_recent_papers: Prioritize recent publications
        recent_paper_years: Years considered "recent"
    """
    store_capability_config(
        orchestrator,
        "literature_config",
        {
            "min_relevance_score": min_relevance_score,
            "weight_citation_count": weight_citation_count,
            "prefer_recent_papers": prefer_recent_papers,
            "recent_paper_years": recent_paper_years,
        },
    )

    logger.info(
        f"Configured literature analysis: relevance>={min_relevance_score:.0%}, "
        f"recent_years={recent_paper_years}"
    )


def get_literature_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current literature analysis configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Literature configuration dict
    """
    return load_capability_config(orchestrator, "literature_config", _LITERATURE_DEFAULTS)


def configure_fact_checking(
    orchestrator: Any,
    *,
    min_confidence_threshold: float = 0.5,
    require_multiple_sources: bool = True,
    min_source_count_for_claim: int = 2,
    track_supporting_refuting: bool = True,
) -> None:
    """Configure fact-checking and claim verification.

    Args:
        orchestrator: Target orchestrator
        min_confidence_threshold: Minimum confidence for verdict (0-1)
        require_multiple_sources: Require multiple sources for claims
        min_source_count_for_claim: Minimum sources to verify a claim
        track_supporting_refuting: Track supporting vs refuting evidence
    """
    store_capability_config(
        orchestrator,
        "fact_checking_config",
        {
            "min_confidence_threshold": min_confidence_threshold,
            "require_multiple_sources": require_multiple_sources,
            "min_source_count_for_claim": min_source_count_for_claim,
            "track_supporting_refuting": track_supporting_refuting,
        },
    )

    logger.info(
        f"Configured fact checking: confidence>={min_confidence_threshold:.0%}, "
        f"min_sources={min_source_count_for_claim}"
    )


def get_fact_checking_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current fact-checking configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Fact-checking configuration dict
    """
    return load_capability_config(orchestrator, "fact_checking_config", _FACT_CHECKING_DEFAULTS)


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="research_source_verification",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Source credibility validation and verification settings",
)
def source_verification_capability(
    min_credibility: float = 0.7,
    min_source_count: int = 3,
    **kwargs: Any,
) -> Callable:
    """Source verification capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_source_verification(
            orchestrator,
            min_credibility=min_credibility,
            min_source_count=min_source_count,
            **kwargs,
        )

    return handler


@capability(
    name="research_citation",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Citation management and bibliography formatting",
    getter="get_citation_config",
)
def citation_capability(
    default_style: str = "apa",
    require_urls: bool = True,
    **kwargs: Any,
) -> Callable:
    """Citation management capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_citation_management(
            orchestrator,
            default_style=default_style,
            require_urls=require_urls,
            **kwargs,
        )

    return handler


@capability(
    name="research_quality",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Research quality standards and coverage requirements",
    getter="get_research_quality",
)
def research_quality_capability(
    min_coverage_score: float = 0.75,
    min_source_diversity: int = 2,
    **kwargs: Any,
) -> Callable:
    """Research quality capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_research_quality(
            orchestrator,
            min_coverage_score=min_coverage_score,
            min_source_diversity=min_source_diversity,
            **kwargs,
        )

    return handler


@capability(
    name="research_literature",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Literature analysis and academic paper evaluation",
    getter="get_literature_config",
)
def literature_capability(
    min_relevance_score: float = 0.6,
    prefer_recent_papers: bool = True,
    **kwargs: Any,
) -> Callable:
    """Literature analysis capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_literature_analysis(
            orchestrator,
            min_relevance_score=min_relevance_score,
            prefer_recent_papers=prefer_recent_papers,
            **kwargs,
        )

    return handler


@capability(
    name="research_fact_checking",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Fact-checking and claim verification configuration",
    getter="get_fact_checking_config",
)
def fact_checking_capability(
    min_confidence_threshold: float = 0.5,
    require_multiple_sources: bool = True,
    **kwargs: Any,
) -> Callable:
    """Fact-checking capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_fact_checking(
            orchestrator,
            min_confidence_threshold=min_confidence_threshold,
            require_multiple_sources=require_multiple_sources,
            **kwargs,
        )

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class ResearchCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for Research-specific capabilities.

    This class provides a structured way to access and apply
    Research capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

    Example:
        provider = ResearchCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_source_verification(orchestrator)
        provider.apply_citation_management(orchestrator, style="chicago")

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("citation_management")
        if cap:
            cap(orchestrator)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "source_verification": configure_source_verification,
            "citation_management": configure_citation_management,
            "research_quality": configure_research_quality,
            "literature_analysis": configure_literature_analysis,
            "fact_checking": configure_fact_checking,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "source_verification": CapabilityMetadata(
                name="source_verification",
                description="Source credibility validation and verification settings",
                version="1.0",
                tags=["safety", "verification", "credibility", "sources"],
            ),
            "citation_management": CapabilityMetadata(
                name="citation_management",
                description="Citation management and bibliography formatting",
                version="1.0",
                tags=["citation", "bibliography", "formatting"],
            ),
            "research_quality": CapabilityMetadata(
                name="research_quality",
                description="Research quality standards and coverage requirements",
                version="1.0",
                dependencies=["source_verification"],
                tags=["quality", "coverage", "standards"],
            ),
            "literature_analysis": CapabilityMetadata(
                name="literature_analysis",
                description="Literature analysis and academic paper evaluation",
                version="1.0",
                dependencies=["source_verification"],
                tags=["literature", "academic", "papers", "research"],
            ),
            "fact_checking": CapabilityMetadata(
                name="fact_checking",
                description="Fact-checking and claim verification configuration",
                version="1.0",
                dependencies=["source_verification"],
                tags=["fact-check", "verification", "evidence", "safety"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to handler functions.
        """
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        return self._metadata.copy()

    def apply_source_verification(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply source verification capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Source verification options
        """
        configure_source_verification(orchestrator, **kwargs)
        self._applied.add("source_verification")

    def apply_citation_management(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply citation management capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Citation options
        """
        configure_citation_management(orchestrator, **kwargs)
        self._applied.add("citation_management")

    def apply_research_quality(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply research quality capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Research quality options
        """
        configure_research_quality(orchestrator, **kwargs)
        self._applied.add("research_quality")

    def apply_literature_analysis(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply literature analysis capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Literature analysis options
        """
        configure_literature_analysis(orchestrator, **kwargs)
        self._applied.add("literature_analysis")

    def apply_fact_checking(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply fact-checking capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Fact-checking options
        """
        configure_fact_checking(orchestrator, **kwargs)
        self._applied.add("fact_checking")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all Research capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_source_verification(orchestrator)
        self.apply_citation_management(orchestrator)
        self.apply_research_quality(orchestrator)
        self.apply_literature_analysis(orchestrator)
        self.apply_fact_checking(orchestrator)

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="research_source_verification",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_source_verification",
            getter="get_source_verification",
            description="Source credibility validation and verification settings",
        ),
        handler=configure_source_verification,
        getter_handler=get_source_verification,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="research_citation",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_citation_management",
            getter="get_citation_config",
            description="Citation management and bibliography formatting",
        ),
        handler=configure_citation_management,
        getter_handler=get_citation_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="research_quality",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_research_quality",
            getter="get_research_quality",
            description="Research quality standards and coverage requirements",
        ),
        handler=configure_research_quality,
        getter_handler=get_research_quality,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="research_literature",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_literature_analysis",
            getter="get_literature_config",
            description="Literature analysis and academic paper evaluation",
        ),
        handler=configure_literature_analysis,
        getter_handler=get_literature_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="research_fact_checking",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_fact_checking",
            getter="get_fact_checking_config",
            description="Fact-checking and claim verification configuration",
        ),
        handler=configure_fact_checking,
        getter_handler=get_fact_checking_config,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_research_capabilities() -> List[CapabilityEntry]:
    """Get all Research capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_research_capability_loader(
    loader_factory: Optional[Callable[[], Any]] = None,
) -> Any:
    """Create and populate a host capability loader for the Research vertical.

    Args:
        loader_factory: Host-side factory that returns a compatible capability loader.

    Returns:
        Host capability loader populated with Research capability entries.
    """
    if loader_factory is None:
        raise RuntimeError(
            "create_research_capability_loader() requires a host loader_factory, such as "
            "victor.framework.CapabilityLoader."
        )

    return build_capability_loader(
        CAPABILITIES,
        loader_factory=loader_factory,
        source_module="victor_research.capabilities",
    )


__all__ = [
    # Handlers
    "configure_source_verification",
    "configure_citation_management",
    "configure_research_quality",
    "configure_literature_analysis",
    "configure_fact_checking",
    # Getters
    "get_source_verification",
    "get_citation_config",
    "get_research_quality",
    "get_literature_config",
    "get_fact_checking_config",
    # Provider class and base types
    "ResearchCapabilityProvider",
    "CapabilityMetadata",  # Re-exported from framework for convenience
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_research_capabilities",
    "create_research_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]


# =============================================================================
# SOLID: Centralized Config Storage
# =============================================================================


def get_capability_configs() -> Dict[str, Any]:
    """Get Research capability configurations for centralized storage.

    Returns default Research configuration for VerticalContext storage.
    This replaces direct orchestrator.source_verification_config assignment.

    Returns:
        Dict with default Research capability configurations
    """
    return {
        "source_verification_config": {
            "min_sources": 3,
            "require_https": True,
            "exclude_domains": [],
            "trusted_domains": [],
        },
        "citation_config": {
            "style": "apa",
            "include_urls": True,
            "max_citations": 10,
        },
        "research_quality_config": {
            "min_coverage_score": 0.7,
            "require_multiple_sources": True,
            "enable_bias_detection": True,
        },
        "literature_config": {
            "prefer_recent": True,
            "max_paper_age_years": 5,
            "include_preprints": False,
        },
        "fact_checking_config": {
            "evidence_threshold": 0.8,
            "require_primary_sources": True,
            "enable_contradiction_detection": True,
        },
    }
