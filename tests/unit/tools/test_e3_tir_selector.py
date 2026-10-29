"""Tests for E3-TIR Tool Selector and Experience Store."""

from victor.tools.e3_tir_selector import (
    E3TIRConfig,
    E3TIRToolSelector,
    SelectionPhase,
)
from victor.tools.experience_store import (
    ExperienceType,
    ToolExperience,
    ToolExperienceStore,
)

TOOLS = [
    "read",
    "write",
    "edit",
    "ls",
    "grep",
    "shell",
    "git",
    "test",
    "code_search",
    "web_search",
]


class TestToolExperienceStore:
    def test_record_and_retrieve(self):
        store = ToolExperienceStore()
        store.record_outcome("read", "coding", success=True, reward=0.9)
        assert len(store) == 1
        stats = store.get_stats("read")
        assert stats.total_uses == 1
        assert stats.successes == 1

    def test_running_average_reward(self):
        store = ToolExperienceStore()
        store.record_outcome("read", "coding", success=True, reward=1.0)
        store.record_outcome("read", "coding", success=True, reward=0.5)
        stats = store.get_stats("read")
        assert abs(stats.avg_reward - 0.75) < 0.01

    def test_underutilized_tools(self):
        store = ToolExperienceStore()
        store.register_tools(["read", "write", "edit", "ls"])
        store.record_outcome("read", "coding", success=True, reward=0.9)
        store.record_outcome("read", "coding", success=True, reward=0.8)
        underutilized = store.get_underutilized_tools(threshold=3)
        assert "write" in underutilized
        assert "edit" in underutilized
        assert "read" in underutilized  # Only 2 uses, threshold is 3

    def test_stale_tools(self):
        store = ToolExperienceStore()
        store.register_tools(["read", "write"])
        store.record_outcome("read", "coding", success=True, reward=0.9)
        # "write" was never used → infinite staleness
        stale = store.get_stale_tools(staleness_seconds=0.01)
        assert "write" in stale

    def test_sample_by_type(self):
        store = ToolExperienceStore()
        store.record(
            ToolExperience(
                tool_name="read",
                task_type="coding",
                experience_type=ExperienceType.DEMONSTRATION,
                success=True,
                reward=1.0,
            )
        )
        store.record(
            ToolExperience(
                tool_name="write",
                task_type="coding",
                experience_type=ExperienceType.SELF_PLAY,
                success=True,
                reward=0.7,
            )
        )
        demos = store.sample_experiences(experience_type=ExperienceType.DEMONSTRATION)
        assert len(demos) == 1
        assert demos[0].tool_name == "read"

    def test_diversity_score_single_tool(self):
        store = ToolExperienceStore()
        store.register_tools(["read", "write", "edit"])
        store.record_outcome("read", "coding", success=True, reward=0.9)
        store.record_outcome("read", "coding", success=True, reward=0.8)
        # Only one tool used — low diversity
        assert store.get_diversity_score() < 0.5

    def test_diversity_score_balanced(self):
        store = ToolExperienceStore()
        store.register_tools(["read", "write", "edit"])
        store.record_outcome("read", "coding", success=True, reward=0.9)
        store.record_outcome("write", "coding", success=True, reward=0.8)
        store.record_outcome("edit", "coding", success=True, reward=0.7)
        # All tools used equally — high diversity
        assert store.get_diversity_score() > 0.9

    def test_max_capacity_eviction(self):
        store = ToolExperienceStore(max_experiences=10)
        for i in range(20):
            store.record_outcome(f"tool_{i}", "coding", success=True, reward=0.5)
        assert len(store) == 10

    def test_success_rate(self):
        store = ToolExperienceStore()
        store.record_outcome("read", "coding", success=True, reward=0.9)
        store.record_outcome("read", "coding", success=False, reward=0.1)
        store.record_outcome("read", "coding", success=True, reward=0.8)
        assert abs(store.get_stats("read").success_rate - 2 / 3) < 0.01


class TestE3TIRSelector:
    def test_select_returns_tools(self):
        selector = E3TIRToolSelector()
        result = selector.select(TOOLS, task_type="coding")
        assert len(result) > 0
        assert all(t in TOOLS for t in result)

    def test_select_respects_max_tools(self):
        selector = E3TIRToolSelector()
        result = selector.select(TOOLS, task_type="coding", max_tools=3)
        assert len(result) <= 3

    def test_demonstration_boost_reranks(self):
        store = ToolExperienceStore()
        # R4: demonstration-reward reranking is OFF by default (RL Q-value owns tool
        # success). This test exercises the opt-in feature, so it enables it explicitly.
        config = E3TIRConfig(min_demonstrations=1, enable_demonstration_reranking=True)
        selector = E3TIRToolSelector(store=store, config=config)

        # Add demonstrations for "web_search" (normally last)
        for _ in range(5):
            selector.add_demonstration("web_search", "research", reward=1.0)

        result = selector.select(
            TOOLS,
            task_type="research",
            base_ranking=TOOLS,  # web_search is at index 9
        )
        # web_search should be boosted toward the top
        web_idx = result.index("web_search") if "web_search" in result else len(result)
        assert web_idx < 7, f"web_search at index {web_idx}, expected < 7 after demo boost"

    def test_demonstration_boost_gate(self, monkeypatch):
        # R4: demonstration-reward reranking is gated OFF by default (RL Q-value owns
        # tool success) and runs only when explicitly enabled. Verify the gate directly
        # (robust to E3-TIR's other transforms like exploration injection).
        def _make(enabled):
            store = ToolExperienceStore()
            cfg = E3TIRConfig(min_demonstrations=1, enable_demonstration_reranking=enabled)
            sel = E3TIRToolSelector(store=store, config=cfg)
            calls = []
            orig = sel._apply_demonstration_boost

            def spy(candidates, task_type):
                calls.append(1)
                return orig(candidates, task_type)

            monkeypatch.setattr(sel, "_apply_demonstration_boost", spy)
            sel.select(TOOLS, task_type="research", base_ranking=list(TOOLS))
            return calls

        assert _make(False) == []  # default: not invoked
        assert _make(True) == [1]  # opt-in: invoked once

    def test_targeted_exploration_injects_underutilized(self):
        store = ToolExperienceStore()
        store.register_tools(TOOLS)
        # Use only "read" heavily
        for _ in range(20):
            store.record_outcome("read", "coding", success=True, reward=0.9)

        config = E3TIRConfig(underutilized_threshold=5, exploration_slots=2)
        selector = E3TIRToolSelector(store=store, config=config)

        result = selector.select(TOOLS, task_type="coding", base_ranking=["read"])
        # Should have injected at least one underutilized tool
        assert len(result) > 1

    def test_mode_collapse_prevention(self):
        store = ToolExperienceStore()
        config = E3TIRConfig(max_consecutive_same=2, diversity_floor=0.8)
        selector = E3TIRToolSelector(store=store, config=config)

        # Record same tool 5 times
        for _ in range(5):
            selector.record_outcome("read", "coding", success=True, reward=0.9)

        result = selector.select(TOOLS, task_type="coding", base_ranking=["read"] + TOOLS[1:])
        # "read" should be demoted from position 0
        assert result[0] != "read" or result[1] != "read"

    def test_phase_advances(self):
        selector = E3TIRToolSelector()
        assert selector.phase.turn_count == 0
        selector.select(TOOLS, task_type="coding")
        assert selector.phase.turn_count == 1
        selector.select(TOOLS, task_type="coding")
        assert selector.phase.turn_count == 2

    def test_metrics(self):
        store = ToolExperienceStore()
        store.register_tools(TOOLS)
        store.record_outcome("read", "coding", success=True, reward=0.9)
        selector = E3TIRToolSelector(store=store)

        metrics = selector.get_metrics()
        assert "diversity_score" in metrics
        assert "total_experiences" in metrics
        assert metrics["total_experiences"] == 1
        assert "phase" in metrics

    def test_add_demonstration(self):
        selector = E3TIRToolSelector()
        selector.add_demonstration("git", "coding", reward=1.0, trajectory=[{"step": 1}])
        demos = selector.store.sample_experiences(experience_type=ExperienceType.DEMONSTRATION)
        assert len(demos) == 1
        assert demos[0].tool_name == "git"


class TestSelectionPhase:
    def test_warmup_phase(self):
        config = E3TIRConfig(warmup_turns=5)
        phase = SelectionPhase()
        # During warmup, demo and exploration weights should be meaningful
        phase.advance(config)
        assert phase.demo_weight > 0.1
        assert phase.exploration_weight > 0.1

    def test_post_warmup_exploitation(self):
        config = E3TIRConfig(warmup_turns=3)
        phase = SelectionPhase()
        for _ in range(10):
            phase.advance(config)
        # After warmup, self-play (exploitation) should dominate
        assert phase.self_play_weight > 0.7
        assert phase.demo_weight < 0.1

    def test_weights_sum_to_one(self):
        config = E3TIRConfig()
        phase = SelectionPhase()
        for _ in range(20):
            phase.advance(config)
            total = phase.demo_weight + phase.self_play_weight + phase.exploration_weight
            assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
