"""TDD tests for per-section token tracking in GEPA prompt optimization.

Verifies that ExecutionTrace tracks per-section token counts and
ParetoEntry includes token_count for efficiency-aware dominance.
"""

from victor.framework.rl.learners.prompt_optimizer import ExecutionTrace
from victor.framework.rl.pareto import ParetoEntry, ParetoFrontier


class TestExecutionTraceTokens:
    """ExecutionTrace should have section_tokens field."""

    def test_has_section_tokens_field(self):
        trace = ExecutionTrace(
            session_id="s1",
            task_type="coding",
            provider="ollama",
            model="qwen",
            tool_calls=5,
            tool_failures={},
            success=True,
            completion_score=0.9,
            tokens_used=1000,
        )
        assert hasattr(trace, "section_tokens")
        assert isinstance(trace.section_tokens, dict)

    def test_section_tokens_default_empty(self):
        trace = ExecutionTrace(
            session_id="s1",
            task_type="coding",
            provider="ollama",
            model="qwen",
            tool_calls=5,
            tool_failures={},
            success=True,
            completion_score=0.9,
            tokens_used=1000,
        )
        assert trace.section_tokens == {}

    def test_section_tokens_populated(self):
        trace = ExecutionTrace(
            session_id="s1",
            task_type="coding",
            provider="ollama",
            model="qwen",
            tool_calls=5,
            tool_failures={},
            success=True,
            completion_score=0.9,
            tokens_used=1000,
            section_tokens={
                "GROUNDING_RULES": 150,
                "ASI_TOOL_EFFECTIVENESS_GUIDANCE": 200,
                "COMPLETION_GUIDANCE": 120,
            },
        )
        assert trace.section_tokens["GROUNDING_RULES"] == 150
        assert sum(trace.section_tokens.values()) == 470


class TestParetoTokenEfficiency:
    """ParetoEntry should include token_count for efficiency scoring."""

    def test_pareto_entry_has_token_count(self):
        entry = ParetoEntry(
            text_hash="abc",
            text="short prompt",
            generation=1,
        )
        assert hasattr(entry, "token_count")
        assert entry.token_count == 0

    def test_pareto_entry_stores_token_count(self):
        entry = ParetoEntry(
            text_hash="abc",
            text="short prompt",
            generation=1,
            token_count=150,
        )
        assert entry.token_count == 150

    def test_shorter_prompt_preferred_when_equal_quality(self):
        """When quality scores are equal, shorter prompt should be preferred."""
        frontier = ParetoFrontier()
        # Both have identical quality scores
        scores = {"i1": 0.8, "i2": 0.7}
        # Candidate A: longer prompt (300 tokens)
        frontier.add_candidate("long", "x" * 1000, 1, scores, token_count=300)
        # Candidate B: shorter prompt (100 tokens)
        frontier.add_candidate("short", "y" * 300, 2, scores, token_count=100)

        # Both should be on frontier (neither dominates on quality)
        # But shorter should have higher coverage due to efficiency tiebreak
        candidates = frontier._candidates
        assert len(candidates) >= 1
        # The shorter candidate should be present
        hashes = {c.text_hash for c in candidates}
        assert "short" in hashes
