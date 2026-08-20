"""Evolved content resolution for prompt construction.

Bridges the gap between OptimizationInjector and prompt builders,
providing a unified interface for fetching evolved content with
fallback to static defaults.

Research basis:
- arXiv:2601.06007 — System-prompt-only caching is optimal (41-80% cost reduction)
- arXiv:2410.14826 — System prompt optimization yields ~10% gains (SPRIG)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedContent:
    """Resolved content for a prompt section.

    Attributes:
        section_name: Canonical section name (e.g., "ASI_TOOL_EFFECTIVENESS_GUIDANCE")
        text: Resolved text content (evolved or static)
        source: Content source ("evolved", "static", or "custom")
        metadata: Additional metadata (provider, model, candidate_hash, strategy, etc.)
    """

    section_name: str
    text: str
    source: str
    metadata: Dict[str, Any]

    def is_evolved(self) -> bool:
        """Check if content is from evolved source."""
        return self.source == "evolved"

    def is_static(self) -> bool:
        """Check if content is from static fallback."""
        return self.source == "static"


class EvolvedContentResolver:
    """Resolves evolved content for prompt sections.

    Queries OptimizationInjector for evolved versions of prompt sections,
    falling back to static defaults when no evolved version is available
    or when confidence is too low.

    Thread-safe: Uses internal cache for performance.
    """

    def __init__(
        self,
        optimization_injector: Optional[Any] = None,
    ) -> None:
        """Initialize resolver.

        Args:
            optimization_injector: Optional OptimizationInjector instance.
                If None, will always use static fallback.
        """
        self._injector = optimization_injector
        self._cache: Dict[Tuple[str, str, str, str], ResolvedContent] = {}
        logger.debug(
            "EvolvedContentResolver initialized (injector=%s)",
            "yes" if optimization_injector else "no",
        )

    def resolve_section(
        self,
        section_name: str,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
        fallback_text: str = "",
    ) -> ResolvedContent:
        """Resolve a section to evolved or static content.

        Checks cache first, then queries injector for evolved content,
        falling back to static text if no evolved version available.

        Args:
            section_name: Canonical section name (e.g., "ASI_TOOL_EFFECTIVENESS_GUIDANCE")
            provider: Provider name for evolution lookup
            model: Model name for evolution lookup
            task_type: Task type for context
            fallback_text: Static fallback if no evolved version

        Returns:
            ResolvedContent with text and metadata
        """
        # A/B hygiene: with VICTOR_FROZEN_PROMPTS set, always serve the
        # static fallback. Benchmark comparison arms must run on identical
        # prompts — an auto-evolved candidate landing between arms silently
        # confounds the comparison (observed: gen-3 -> gen-4 between the
        # sqlite and proxima arms of the backend A/B).
        import os

        if os.environ.get("VICTOR_FROZEN_PROMPTS"):
            return ResolvedContent(
                text=fallback_text,
                source="static",
                section_name=section_name,
                metadata={"frozen": True},
            )

        cache_key = self._cache_key(section_name, provider, model, task_type)

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"Cache hit for section: {section_name}")
            return self._cache[cache_key]

        # Try to get evolved content
        if self._injector:
            evolved = self._try_get_evolved(section_name, provider, model, task_type)
            if evolved:
                self._cache[cache_key] = evolved
                logger.info(
                    f"Resolved section '{section_name}' from evolved source "
                    f"(provider={provider or 'default'}, model={model or 'default'})"
                )
                return evolved

        # Fall back to static
        resolved = ResolvedContent(
            section_name=section_name, text=fallback_text, source="static", metadata={}
        )
        self._cache[cache_key] = resolved
        logger.debug(f"Resolved section '{section_name}' from static fallback")
        return resolved

    def resolve_multiple(
        self,
        section_names: List[str],
        provider: str = "",
        model: str = "",
        task_type: str = "default",
        fallback_map: Optional[Dict[str, str]] = None,
    ) -> List[ResolvedContent]:
        """Resolve multiple sections efficiently.

        Resolves all sections in a single call, sharing cache lookups
        and injector queries where possible.

        Args:
            section_names: List of canonical section names
            provider: Provider name for evolution lookup
            model: Model name for evolution lookup
            task_type: Task type for context
            fallback_map: Optional map of section_name -> fallback_text

        Returns:
            List of ResolvedContent in same order as section_names
        """
        fallback_map = fallback_map or {}
        results = []

        for name in section_names:
            fallback = fallback_map.get(name, "")
            resolved = self.resolve_section(name, provider, model, task_type, fallback)
            results.append(resolved)

        evolved_count = sum(1 for r in results if r.is_evolved())
        logger.debug(
            f"Resolved {len(results)} sections: {evolved_count} evolved, "
            f"{len(results) - evolved_count} static"
        )

        return results

    def clear_cache(self) -> None:
        """Clear resolution cache.

        Call this when provider/model changes or session ends.
        """
        cleared = len(self._cache)
        self._cache.clear()
        logger.debug(f"Cleared {cleared} cached section resolutions")

    @staticmethod
    def _cache_key(
        section_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Tuple[str, str, str, str]:
        """Build a provider/model-aware cache key for resolved sections."""
        return (
            str(section_name or "").strip(),
            str(provider or "").strip(),
            str(model or "").strip(),
            str(task_type or "default").strip() or "default",
        )

    def _try_get_evolved(
        self,
        section_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[ResolvedContent]:
        """Try to get evolved content from injector.

        Returns None if no evolved version available or confidence too low.

        Args:
            section_name: Canonical section name
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            ResolvedContent if evolved version found, None otherwise
        """
        if not self._injector:
            return None

        try:
            if hasattr(self._injector, "get_section_payload"):
                payload = self._injector.get_section_payload(
                    section_name,
                    provider=provider,
                    model=model,
                    task_type=task_type,
                )
                if isinstance(payload, dict):
                    text = payload.get("text", "")
                    if text:
                        return ResolvedContent(
                            section_name=section_name,
                            text=text,
                            source="evolved",
                            metadata={
                                "provider": payload.get("provider", provider),
                                "prompt_candidate_hash": payload.get("prompt_candidate_hash"),
                                "strategy_name": payload.get("strategy_name"),
                                "strategy_chain": payload.get("strategy_chain"),
                            },
                        )

            # Get evolved section payloads
            payloads = self._injector.get_evolved_section_payloads(
                provider=provider,
                model=model,
                task_type=task_type,
            )

            # Find matching section
            for payload in payloads:
                if payload.get("prompt_section_name") == section_name:
                    text = payload.get("text", "")
                    if not text:
                        continue

                    return ResolvedContent(
                        section_name=section_name,
                        text=text,
                        source="evolved",
                        metadata={
                            "provider": payload.get("provider", provider),
                            "prompt_candidate_hash": payload.get("prompt_candidate_hash"),
                            "strategy_name": payload.get("strategy_name"),
                            "strategy_chain": payload.get("strategy_chain"),
                        },
                    )

        except Exception as e:
            # If evolution fails, silently fall back to static
            logger.warning(
                f"Failed to get evolved content for '{section_name}': {e}. "
                "Falling back to static."
            )

        return None
