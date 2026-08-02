# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned parallel exploration runtime for the agentic loop.

Runs read-only tools (code_search, ls) in parallel before the main
agent loop starts. Provides the agent with pre-loaded context about
the project structure and relevant source files.

Uses direct tool calls — no SubAgent overhead. Tools are called via
asyncio.gather() for true parallelism.

Preferred imports for orchestration-facing callers:
    from victor.agent.coordinators import (
        ExplorationCoordinator,
        ExplorationStatePassedCoordinator,
    )

Usage:
    coordinator = ExplorationCoordinator()
    findings = await coordinator.explore_parallel(
        "Fix the bug in separability_matrix",
        project_root=Path("/path/to/repo"),
    )

Use ``ExplorationCoordinator`` for direct read-only discovery and
``ExplorationStatePassedCoordinator`` when you need snapshot/transition style
state-passed orchestration.

The legacy `victor.agent.coordinators.exploration_coordinator` module now
re-exports this implementation for compatibility.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Aggregated results from parallel exploration."""

    file_paths: List[str] = field(default_factory=list)
    summary: str = ""
    duration_seconds: float = 0.0
    tool_calls: int = 0


class ExplorationCoordinator:
    """Coordinates parallel file exploration during READING stage.

    Runs code_search + ls in parallel to give the main agent a head start
    on understanding the codebase. No SubAgent orchestrators — just direct
    async tool calls for minimal overhead.
    """

    async def explore_parallel(
        self,
        task_description: str,
        project_root: Path,
        max_results: int = 5,
        provider: str = "ollama",
        model: Optional[str] = None,
        complexity: str = "action",
    ) -> ExplorationResult:
        """Run parallel exploration and return aggregated findings.

        Uses ResourceBudget to determine parallelism based on provider,
        hardware, and task complexity.

        Args:
            task_description: The user's task/issue description
            project_root: Root directory of the project to explore
            max_results: Max search results to return
            provider: LLM provider name (affects parallelism)
            model: Optional model name
            complexity: Task complexity level

        Returns:
            ExplorationResult with file paths, summary, and metrics
        """
        from victor.agent.budget.resource_calculator import calculate_exploration_budget

        start = time.time()

        # Calculate resource-aware budget
        budget = calculate_exploration_budget(complexity=complexity, provider=provider, model=model)

        if budget.max_parallel_agents == 0:
            logger.debug("Resource budget: no exploration for this complexity")
            return ExplorationResult(duration_seconds=time.time() - start)

        # Extract key terms from task for targeted search
        key_terms = self._extract_search_terms(task_description)
        if not key_terms:
            return ExplorationResult(duration_seconds=time.time() - start)

        logger.info(
            "Parallel exploration: %d agents, timeout=%ds, searching %s in %s",
            budget.max_parallel_agents,
            budget.exploration_timeout,
            key_terms[:2],
            project_root.name,
        )

        # Run searches in parallel — limited by resource budget
        tasks = []
        for term in key_terms[: budget.max_parallel_agents]:
            tasks.append(self._search_codebase(term, project_root))

        # Also get project structure
        tasks.append(self._list_project_structure(project_root))

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=budget.exploration_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Parallel exploration timed out after %ds", budget.exploration_timeout)
            return ExplorationResult(duration_seconds=time.time() - start)
        except Exception as e:
            logger.debug("Parallel exploration failed: %s", e)
            return ExplorationResult(duration_seconds=time.time() - start)

        # Aggregate results
        file_paths: List[str] = []
        summaries: List[str] = []
        tool_count = len(tasks)

        for r in results:
            if isinstance(r, Exception):
                continue
            if isinstance(r, dict):
                # code_search result
                files = r.get("files", [])
                if isinstance(files, list):
                    for f in files[:max_results]:
                        path = f if isinstance(f, str) else str(f)
                        if path not in file_paths:
                            file_paths.append(path)
                summary = r.get("summary", "")
                if summary:
                    summaries.append(summary)
            elif isinstance(r, str):
                # ls result
                summaries.append(f"Project structure:\n{r[:500]}")

        result = ExplorationResult(
            file_paths=file_paths[:10],
            summary="\n\n".join(summaries[:3]),
            duration_seconds=time.time() - start,
            tool_calls=tool_count,
        )

        logger.info(
            "Parallel exploration: %d files found, %d tool calls, %.1fs",
            len(result.file_paths),
            result.tool_calls,
            result.duration_seconds,
        )

        return result

    def _extract_search_terms(self, task: str) -> List[str]:
        """Extract meaningful search terms from task description."""
        terms = []

        # Extract quoted strings (backtick or double-quote)
        quoted = re.findall(r'`([^`]+)`|"([^"]+)"', task)
        for q in quoted:
            term = q[0] or q[1]
            if len(term) > 2:
                terms.append(term)

        # Extract true CamelCase identifiers (require internal uppercase, e.g. ActionIntent)
        # or snake_case identifiers (e.g. task_analyzer). Single-capitalized words like
        # "Review" or "Analyze" are NOT code identifiers and are excluded.
        camel = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+\b", task)  # ≥2 words joined
        snake = re.findall(r"\b[a-z]+_[a-z_]+\b", task)
        for ident in camel + snake:
            if len(ident) > 3 and ident not in terms:
                terms.append(ident)

        # Fallback for conceptual/English queries (e.g. "review the codebase for UX issues"):
        # extract meaningful content words when no code identifiers were found.
        if not terms:
            stop_words = {
                "a",
                "an",
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "is",
                "are",
                "was",
                "be",
                "it",
                "its",
                "this",
                "that",
                "all",
                "any",
                "how",
                "what",
                "why",
                "when",
                "where",
                "who",
                "can",
                "do",
                "not",
                "if",
                "as",
                "up",
                "out",
                "so",
                "i",
                "we",
                "you",
                "he",
                "she",
                "they",
                "then",
                "than",
                "into",
                "over",
                "also",
                "should",
                "could",
                "would",
                "will",
                "may",
                "must",
                "have",
                "has",
                "had",
                "been",
                "being",
                "let",
                "get",
                "find",
                "identify",
                "review",
                "analyze",
                "check",
                "look",
                "list",
                "show",
            }
            words = re.findall(r"\b[a-zA-Z]{4,}\b", task.lower())
            seen: set = set()
            for word in words:
                if word not in stop_words and word not in seen:
                    seen.add(word)
                    terms.append(word)
                    if len(terms) >= 5:
                        break

        return terms[:5]

    async def _search_codebase(self, query: str, project_root: Path) -> Dict[str, Any]:
        """Run code_search for a single query via dynamic vertical loading."""
        try:
            from victor.core.utils.capability_loader import load_code_search_module

            module = load_code_search_module()
            code_search = module.code_search

            result = await code_search(
                query=query,
                path=str(project_root),
                k=5,
            )
            return result if isinstance(result, dict) else {"summary": str(result)}
        except ImportError:
            logger.debug("victor-coding not available, skipping code search.")
            return {}
        except Exception as e:
            logger.debug("Search for '%s' failed: %s", query, e)
            return {}

    async def _list_project_structure(self, project_root: Path) -> str:
        """Get project directory listing."""
        try:
            from victor.tools.filesystem import ls

            result = await ls(path=str(project_root), depth=1)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.debug("ls failed: %s", e)
            return ""
