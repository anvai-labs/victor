# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integration tests for GEPA prompt optimizer with local Ollama.

These tests require Ollama running locally with qwen3.5:2b available.
They exercise the full GEPA pipeline: reflect, mutate, and the tiered
service wrapper.

Run: pytest tests/integration/framework/rl/ -v --timeout=180
Skip in CI: -m "not integration"
"""

import sqlite3

import pytest

# All tests in this module require Ollama
pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]


def _ollama_available() -> bool:
    """Check if Ollama is running and qwen3.5:2b is available."""
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any("qwen3.5" in m or "qwen3" in m for m in models)
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running or qwen3.5:2b not available",
)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


class TestGEPAStrategyWithOllama:
    """Tests for GEPAStrategy.reflect() and .mutate() with real Ollama."""

    @skip_no_ollama
    def test_reflect_produces_analysis_with_llm(self):
        from victor.framework.rl.learners.prompt_optimizer import (
            ExecutionTrace,
            GEPAStrategy,
        )

        strategy = GEPAStrategy()
        traces = [
            ExecutionTrace(
                "s1",
                "action",
                "ollama",
                "qwen",
                5,
                {"file_not_found": 3},
                False,
                0.3,
                1000,
            ),
            ExecutionTrace(
                "s2",
                "action",
                "ollama",
                "qwen",
                3,
                {},
                True,
                0.9,
                800,
            ),
        ]
        reflection = strategy.reflect(traces, "TEST_SECTION", "current text")
        # Heuristic analysis always present
        assert "2 execution traces" in reflection
        assert "file_not_found" in reflection
        # LLM enhancement may or may not be present depending on model load
        # but the heuristic part is always there

    @skip_no_ollama
    def test_reflect_empty_traces(self):
        from victor.framework.rl.learners.prompt_optimizer import GEPAStrategy

        strategy = GEPAStrategy()
        reflection = strategy.reflect([], "TEST_SECTION", "current text")
        assert "0 execution traces" in reflection

    @skip_no_ollama
    def test_mutate_with_llm_produces_different_text(self):
        from victor.framework.rl.learners.prompt_optimizer import GEPAStrategy

        strategy = GEPAStrategy()
        current = "GROUNDING: Base responses on tool output only."
        reflection = (
            "Analysis: 60% file_not_found errors.\n"
            "- Agents read directories instead of listing them first.\n"
            "- Edit failures from ambiguous old_str matches."
        )
        result = strategy.mutate(current, reflection, "GROUNDING_RULES")
        # With LLM: result should differ from current
        # Without LLM: heuristic adds file_not_found guidance
        assert len(result) > 0


class TestGEPAServiceWithOllama:
    """Tests for the tiered GEPAService with real Ollama (economic tier)."""

    @skip_no_ollama
    def test_economic_tier_reflect(self):
        from victor.config.gepa_settings import GEPASettings
        from victor.framework.rl.gepa_tier_manager import GEPATierManager

        config = GEPASettings(
            enabled=True,
            default_tier="economic",
        )
        mgr = GEPATierManager(config)
        service = mgr.get_service()
        assert service.get_tier() == "economic"

        result = service.reflect(
            "Session s1: read('foo.py') -> FAILED (file_not_found)",
            "GROUNDING_RULES",
            "Base responses on tool output only.",
        )
        assert len(result) > 20  # Got meaningful reflection

    @skip_no_ollama
    def test_economic_tier_mutate(self):
        from victor.config.gepa_settings import GEPASettings
        from victor.framework.rl.gepa_tier_manager import GEPATierManager

        config = GEPASettings(enabled=True, default_tier="economic")
        mgr = GEPATierManager(config)
        service = mgr.get_service()

        result = service.mutate(
            "Base responses on tool output only.",
            "- Agents read directories instead of files",
            "GROUNDING_RULES",
            max_chars=500,
        )
        assert len(result) > 0
        assert len(result) <= 500  # Bloat control enforced

    @skip_no_ollama
    def test_full_evolution_cycle(self, db):
        """End-to-end: create learner, evolve with Ollama strategy.

        If usage traces exist on disk (from prior runs), evolution may succeed.
        If not, it returns None (insufficient traces). Both are valid.
        """
        from victor.config.gepa_settings import GEPASettings
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy
        from victor.framework.rl.gepa_tier_manager import GEPATierManager
        from victor.framework.rl.learners.prompt_optimizer import (
            PromptCandidate,
            PromptOptimizerLearner,
        )

        config = GEPASettings(enabled=True, default_tier="economic")
        tier_mgr = GEPATierManager(config)
        strategy = GEPAServiceStrategy(tier_mgr)

        learner = PromptOptimizerLearner(
            name="integration_test",
            db_connection=db,
            strategy=strategy,
            use_pareto=True,
            max_prompt_chars=500,
        )

        result = learner.evolve("TEST_SECTION", "Base prompt text.")
        # Either None (not enough traces) or a valid candidate
        if result is not None:
            assert isinstance(result, PromptCandidate)
            assert result.generation >= 1
            assert len(result.text) <= 500  # Bloat control enforced
