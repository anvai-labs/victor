"""Unit tests for ToolService KV-prefix strategy methods (Item 3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from victor.agent.services.tool_service import ToolService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, *, schema_level: str | None = "full", priority=None) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.schema_level = schema_level
    tool.to_schema = MagicMock(return_value={"name": name, "description": "x" * 100})
    if priority is not None:
        tool.priority = priority
    else:
        del tool.priority  # ensure hasattr returns False
    return tool


def _make_service() -> ToolService:
    """Construct a minimal ToolService with required dependencies mocked."""
    settings = MagicMock()
    settings.tools.budget = 50
    registry = MagicMock()
    registry.get_all_tools.return_value = []
    service = ToolService.__new__(ToolService)
    service._settings = settings
    service._registry = registry
    service._logger = MagicMock()
    service._usage_stats = {}
    return service


# ---------------------------------------------------------------------------
# sort_tools_for_kv_stability
# ---------------------------------------------------------------------------


class TestSortToolsForKvStability:
    def test_sorts_full_before_compact_before_stub(self):
        svc = _make_service()
        tools = [
            _make_tool("z_stub", schema_level="stub"),
            _make_tool("a_full", schema_level="full"),
            _make_tool("b_compact", schema_level="compact"),
        ]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=True)
        assert [t.name for t in result] == ["a_full", "b_compact", "z_stub"]

    def test_alphabetical_within_same_level(self):
        svc = _make_service()
        tools = [
            _make_tool("c_full", schema_level="full"),
            _make_tool("a_full", schema_level="full"),
        ]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=True)
        assert result[0].name == "a_full"

    def test_ranked_core_tools_precede_alphabetical_order(self):
        svc = _make_service()
        tools = [
            _make_tool("shell", schema_level="full"),
            _make_tool("code_search", schema_level="full"),
            _make_tool("read", schema_level="full"),
        ]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=True)
        # Priority order per STABLE_TOOL_ORDER: read < shell < code_search.
        # (shell is more fundamental than code_search in the core read/edit loop.)
        assert [t.name for t in result] == ["read", "shell", "code_search"]

    def test_returns_none_when_tools_is_none(self):
        svc = _make_service()
        assert svc.sort_tools_for_kv_stability(None) is None

    def test_passthrough_when_kv_disabled(self):
        svc = _make_service()
        tools = [_make_tool("z"), _make_tool("a")]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=False)
        assert result is tools  # same object, no copy

    def test_unknown_schema_level_treated_as_stub(self):
        svc = _make_service()
        tools = [
            _make_tool("b_full", schema_level="full"),
            _make_tool("a_unknown", schema_level=None),
        ]
        result = svc.sort_tools_for_kv_stability(tools)
        assert result[0].name == "b_full"
        assert result[1].name == "a_unknown"


# ---------------------------------------------------------------------------
# estimate_tool_tokens
# ---------------------------------------------------------------------------


class TestEstimateToolTokens:
    def test_returns_positive_integer(self):
        svc = _make_service()
        tool = _make_tool("my_tool")
        with (patch("victor.config.tool_tiers.get_tool_tier", return_value="full"),):
            result = svc.estimate_tool_tokens(tool)
        assert isinstance(result, int)
        assert result > 0

    def test_fallback_on_schema_error(self):
        svc = _make_service()
        tool = MagicMock()
        tool.name = "bad_tool"
        tool.to_schema = MagicMock(side_effect=RuntimeError("boom"))
        with patch("victor.config.tool_tiers.get_tool_tier", return_value="full"):
            result = svc.estimate_tool_tokens(tool)
        # fallback: len(name) + 50
        assert result == len("bad_tool") + 50

    def test_uses_provider_category_when_given(self):
        svc = _make_service()
        tool = _make_tool("my_tool")
        with (
            patch(
                "victor.config.tool_tiers.get_provider_tool_tier",
                return_value="compact",
            ) as mock_pt,
            patch("victor.config.tool_tiers.get_tool_tier") as mock_gt,
        ):
            svc.estimate_tool_tokens(tool, provider_category="small")
        mock_pt.assert_called_once_with("my_tool", "small")
        mock_gt.assert_not_called()


# ---------------------------------------------------------------------------
# apply_kv_tool_strategy — session_stable
# ---------------------------------------------------------------------------


class TestApplyKvToolStrategySessionStable:
    def test_additive_appends_new_tools(self):
        # P4: session_stable is grow-only/additive — a newly-selected tool joins the
        # cached set instead of being locked out (the mid-session intent-shift fix).
        svc = _make_service()
        cached = [_make_tool("cached")]
        fresh = [_make_tool("fresh")]
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            fresh,
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=cached,
            kv_tool_strategy="session_stable",
        )
        assert [t.name for t in result] == ["cached", "fresh"]

    def test_returns_cached_object_when_no_new_tools(self):
        # A turn that selects nothing new returns the cached set unchanged -> KV cache hit.
        svc = _make_service()
        cached = [_make_tool("a"), _make_tool("b")]
        fresh = [_make_tool("a")]  # subset of cached
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            fresh,
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=cached,
            kv_tool_strategy="session_stable",
        )
        assert result is cached

    def test_additive_alias_behaves_like_session_stable(self):
        svc = _make_service()
        cached = [_make_tool("a")]
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            [_make_tool("b")],
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=cached,
            kv_tool_strategy="additive",
        )
        assert [t.name for t in result] == ["a", "b"]

    def test_returns_tools_on_first_call_no_cache(self):
        svc = _make_service()
        tools = [_make_tool("t")]
        provider = MagicMock(supports_prompt_caching=MagicMock(return_value=False))
        provider.context_window = MagicMock(return_value=8192)
        result = svc.apply_kv_tool_strategy(
            tools,
            kv_optimization_enabled=True,
            provider=provider,
            model="m",
            session_semantic_tools=None,
            kv_tool_strategy="session_stable",
        )
        assert result is tools

    def test_passthrough_when_kv_disabled(self):
        svc = _make_service()
        tools = [_make_tool("t")]
        provider = MagicMock()
        result = svc.apply_kv_tool_strategy(
            tools,
            kv_optimization_enabled=False,
            provider=provider,
            model="m",
            kv_tool_strategy="session_stable",
        )
        assert result is tools


# ---------------------------------------------------------------------------
# apply_context_aware_strategy — compact providers
# ---------------------------------------------------------------------------


class TestApplyContextAwareStrategy:
    def _provider(self, *, supports_caching: bool = False, context_window: int = 8192):
        p = MagicMock()
        p.supports_prompt_caching = MagicMock(return_value=supports_caching)
        p.context_window = MagicMock(return_value=context_window)
        p.name = "test_provider"
        return p

    def test_session_lock_for_caching_provider(self):
        svc = _make_service()
        tools = [_make_tool(f"t{i}") for i in range(5)]
        provider = self._provider(supports_caching=True, context_window=100000)
        with (
            patch("victor.config.tool_tiers.get_provider_category", return_value="cloud"),
            patch.object(svc, "estimate_tool_tokens", return_value=50),
        ):
            result = svc.apply_context_aware_strategy(tools, provider=provider, model="gpt")
        assert result == tools  # session-lock: returns all tools unchanged

    def test_semantic_selection_for_small_context(self):
        svc = _make_service()
        tools = [_make_tool(f"t{i}") for i in range(10)]
        provider = self._provider(supports_caching=False, context_window=4096)
        with (
            patch("victor.config.tool_tiers.get_provider_category", return_value="small"),
            patch.object(svc, "estimate_tool_tokens", return_value=100),
            patch.object(svc, "semantic_select_tools", return_value=tools[:3]) as mock_ss,
        ):
            result = svc.apply_context_aware_strategy(tools, provider=provider, model="small-m")
        mock_ss.assert_called_once()
        assert result == tools[:3]

    def test_context_aware_large_no_cache_provider_merges_additively(self):
        svc = _make_service()
        cached = [_make_tool("read")]
        fresh = [_make_tool("git")]
        provider = self._provider(supports_caching=False, context_window=128000)
        with (
            patch("victor.config.tool_tiers.get_provider_category", return_value="large"),
            patch.object(svc, "estimate_tool_tokens", return_value=50),
        ):
            result = svc.apply_context_aware_strategy(
                fresh,
                provider=provider,
                model="glm-5.2",
                session_semantic_tools=cached,
            )
        assert [t.name for t in result] == ["read", "git"]

    def test_context_aware_runs_even_when_kv_flag_disabled(self):
        svc = _make_service()
        cached = [_make_tool("read")]
        provider = self._provider(supports_caching=False, context_window=128000)
        with (
            patch("victor.config.tool_tiers.get_provider_category", return_value="large"),
            patch.object(svc, "estimate_tool_tokens", return_value=50),
        ):
            result = svc.apply_kv_tool_strategy(
                [_make_tool("refs")],
                kv_optimization_enabled=False,
                provider=provider,
                model="glm-5.2",
                session_semantic_tools=cached,
                kv_tool_strategy="context_aware",
            )
        assert [t.name for t in result] == ["read", "refs"]


# ---------------------------------------------------------------------------
# semantic_select_tools
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# STABLE_TOOL_ORDER coverage (coding-agent priority ranking)
# ---------------------------------------------------------------------------


class TestStableToolOrderCoverage:
    """Regression: STABLE_TOOL_ORDER must rank the full coding-agent tool set.

    Previously only 14/54 tools were ranked, so the remaining 40 fell through to
    alphabetical ordering (unknown_rank tie-break). This made the serialized
    tool schema order alphabetical instead of priority-based, hurting both model
    salience and the intended read/edit/shell-first ordering.
    """

    CORE_LOOP_TOOLS = [
        "read",
        "edit",
        "write",
        "shell",
        "ls",
        "code_search",
    ]

    GRAPH_FOLDED_TOOLS = {
        "graph_analytics",
        "graph_dependencies",
        "graph_neighbors",
        "graph_path",
        "graph_patterns",
        "graph_query",
        "graph_search",
        "graph_semantic",
        "graph_semantic_search",
        "impact_analysis",
    }

    # Every tool observed in a real coding-agent session before graph aliases
    # were folded into graph(mode=...).
    FULL_SESSION_TOOLS = {
        "graph",
        "graph_analytics",
        "graph_dependencies",
        "graph_neighbors",
        "graph_path",
        "graph_patterns",
        "graph_query",
        "graph_search",
        "graph_semantic",
        "lsp",
        "cicd",
        "docker",
        "git",
        "pr",
        "db",
        "test",
        "find",
        "ls",
        "project_overview",
        "read",
        "write",
        "deps",
        "shell",
        "workflow",
        "notebook_edit",
        "extract",
        "inline",
        "organize_imports",
        "rename",
        "jira",
        "batch",
        "code_search",
        "http",
        "docs",
        "docs_coverage",
        "scaffold",
        "cache",
        "web_fetch",
        "web_search",
        "sandbox",
        "mcp",
        "scan",
        "graph_semantic_search",
        "impact_analysis",
        "refs",
        "symbol",
        "edit",
        "metrics",
        "analysis_checkpoint",
        "patch",
        "merge",
        "audit",
        "pipeline",
        "iac",
    }

    DEFAULT_SESSION_TOOLS = FULL_SESSION_TOOLS - GRAPH_FOLDED_TOOLS

    def test_core_loop_tools_ranked_before_others(self):
        """read/edit/write/shell/ls/code_search must precede unranked tools."""
        svc = _make_service()
        tools = [
            _make_tool("zzz_unranked", schema_level="full"),
            _make_tool("shell", schema_level="full"),
            _make_tool("read", schema_level="full"),
            _make_tool("edit", schema_level="full"),
        ]
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=True)
        names = [t.name for t in result]
        # All core-loop tools must come before the unranked tail tool.
        assert names.index("read") < names.index("zzz_unranked")
        assert names.index("edit") < names.index("zzz_unranked")
        assert names.index("shell") < names.index("zzz_unranked")

    def test_core_loop_order_is_read_edit_write_shell_ls_code_search(self):
        """The exact priority order of the core coding loop must be preserved."""
        svc = _make_service()
        tools = [_make_tool(n, schema_level="full") for n in self.CORE_LOOP_TOOLS]
        # Shuffle so input order does not influence result.
        import random

        random.shuffle(tools)
        result = svc.sort_tools_for_kv_stability(tools, kv_optimization_enabled=True)
        assert [t.name for t in result] == self.CORE_LOOP_TOOLS

    def test_full_session_tools_are_all_ranked(self):
        """No coding-agent session tool should fall through to unknown_rank.

        This is the core regression guard: every tool in a real session must
        appear in STABLE_TOOL_ORDER so ordering is priority-based, not
        alphabetical.
        """
        ranked = set(ToolService.STABLE_TOOL_ORDER)
        unranked = self.DEFAULT_SESSION_TOOLS - ranked
        assert not unranked, (
            f"These session tools are unranked and fall back to alphabetical "
            f"ordering: {sorted(unranked)}"
        )

    def test_folded_graph_aliases_are_not_ranked_as_default_tools(self):
        """Graph subtools are intentionally folded into graph(mode=...)."""
        ranked = set(ToolService.STABLE_TOOL_ORDER)
        assert self.GRAPH_FOLDED_TOOLS.isdisjoint(ranked)
        assert "graph" in ranked


# ---------------------------------------------------------------------------
# semantic_select_tools
# ---------------------------------------------------------------------------


class TestSemanticSelectTools:
    def test_respects_token_budget(self):
        svc = _make_service()
        tools = [_make_tool(f"t{i}") for i in range(5)]
        with patch.object(svc, "estimate_tool_tokens", return_value=200):
            result = svc.semantic_select_tools(tools, max_tokens=500)
        # 200*2 = 400 ≤ 500, but 200*3 = 600 > 500 → first 2
        assert len(result) == 2

    def test_empty_tools_returns_empty(self):
        svc = _make_service()
        assert svc.semantic_select_tools([], max_tokens=1000) == []
