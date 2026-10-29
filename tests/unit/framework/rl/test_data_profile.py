"""Tests for data-aware instruction generation (MIPROv2-inspired)."""

from unittest.mock import MagicMock


class TestDataAwareProfile:
    def _make_trace(self, task="coding", provider="ollama", score=0.7, tools=None):
        t = MagicMock()
        t.task_type = task
        t.provider = provider
        t.completion_score = score
        t.tool_calls = 5
        details = []
        for tool in tools or ["read", "edit"]:
            d = MagicMock()
            d.tool_name = tool
            details.append(d)
        t.tool_call_details = details
        return t

    def test_function_exists(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        assert hasattr(GEPAServiceStrategy, "_build_data_profile")

    def test_profile_contains_statistics(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [self._make_trace(score=0.8), self._make_trace(score=0.6)]
        profile = GEPAServiceStrategy._build_data_profile(traces)
        assert "Total sessions: 2" in profile
        assert "Avg completion score" in profile

    def test_profile_shows_task_types(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [self._make_trace(task="coding"), self._make_trace(task="analysis")]
        profile = GEPAServiceStrategy._build_data_profile(traces)
        assert "coding" in profile

    def test_profile_shows_top_tools(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [self._make_trace(tools=["read", "edit", "code_search"])]
        profile = GEPAServiceStrategy._build_data_profile(traces)
        assert "read" in profile

    def test_profile_canonicalizes_legacy_tool_aliases(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [self._make_trace(tools=["read_file", "list_directory", "execute_bash"])]
        profile = GEPAServiceStrategy._build_data_profile(traces)
        assert "read(" in profile
        assert "ls(" in profile
        assert "shell(" in profile
        assert "read_file" not in profile
        assert "list_directory" not in profile
        assert "execute_bash" not in profile

    def test_empty_traces_returns_empty(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        assert GEPAServiceStrategy._build_data_profile([]) == ""

    def test_profile_has_section_header(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [self._make_trace()]
        profile = GEPAServiceStrategy._build_data_profile(traces)
        assert "DATA DISTRIBUTION PROFILE" in profile
