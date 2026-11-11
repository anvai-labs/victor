"""CallerRunContext (consolidation PR 4, seam-duplication audit Cluster 2).

One structured dataclass replaces the hand-built loop-config dicts at the
buffered (`turn_execution_runtime._execute_via_agentic_loop`) and streaming
(`chat_stream_executor.run_unified`) consumption sites, and the scattered
budget-key extraction from `runtime_context_overrides` (the ledger funding
seam).
"""

from victor.core.caller_run_context import CallerRunContext


class TestFromOverridesAndContext:
    def test_full_construction_carries_every_declared_field(self):
        overrides = {"tool_budget": 134, "iteration_budget": 64}
        context = {
            "criteria": {"findings": 30},
            "complete_on_key": "corpus_closed",
            "completion_tool_names": ("admit_proposal",),
            "executor": "standard",
            "completion_strategy": "rubric",
            "unparsed_key": {"nested": True},
        }

        ctx = CallerRunContext.from_overrides_and_context(overrides, context, None)

        assert ctx.tool_budget == 134
        assert ctx.iteration_budget == 64
        assert ctx.criteria == {"findings": 30}
        assert ctx.complete_on_key == "corpus_closed"
        assert ctx.completion_tool_names == ("admit_proposal",)
        assert ctx.executor == "standard"
        assert ctx.completion_strategy == "rubric"
        # Unparsed context keys pass through uninterpreted.
        assert ctx.extra == {"unparsed_key": {"nested": True}}

    def test_to_loop_config_carries_declarations_and_gate_resolution(self):
        ctx = CallerRunContext.from_overrides_and_context(
            {"tool_budget": 134, "iteration_budget": 64},
            {
                "criteria": {"findings": 30},
                "completion_tool_names": ("admit_proposal",),
                "executor": "standard",
                "completion_strategy": "rubric",
            },
            None,
        )

        config = ctx.to_loop_config()
        assert config["enable_criteria_gate"] is True
        assert config["completion_tool_names"] == ("admit_proposal",)
        assert config["executor"] == "standard"
        assert config["completion_strategy"] == "rubric"

    def test_to_runtime_overrides_carries_budget_keys_only(self):
        ctx = CallerRunContext.from_overrides_and_context(
            {"tool_budget": 134, "iteration_budget": 64},
            {"criteria": {"findings": 30}},
            None,
        )

        assert ctx.to_runtime_overrides() == {
            "tool_budget": 134,
            "iteration_budget": 64,
        }

    def test_non_dict_inputs_are_ignored(self):
        ctx = CallerRunContext.from_overrides_and_context(None, None, None)

        assert ctx.tool_budget is None
        assert ctx.iteration_budget is None
        assert ctx.criteria is None
        assert ctx.extra == {}

    def test_budget_values_are_int_coerced(self):
        ctx = CallerRunContext.from_overrides_and_context(
            {"tool_budget": "134", "iteration_budget": "64"}, None, None
        )

        assert ctx.tool_budget == 134
        assert ctx.iteration_budget == 64

    def test_settings_resolve_completion_tool_names_when_undeclared(self):
        settings = type("S", (), {})()
        settings.tools = type("T", (), {})()
        settings.tools.completion_tools = ("admit_proposal",)

        ctx = CallerRunContext.from_overrides_and_context(None, None, settings)

        assert ctx.completion_tool_names == ("admit_proposal",)
        assert "completion_tool_names" in ctx.to_loop_config()

    def test_declared_completion_tool_names_win_over_settings(self):
        settings = type("S", (), {})()
        settings.tools = type("T", (), {})()
        settings.tools.completion_tools = ("settings_tool",)

        ctx = CallerRunContext.from_overrides_and_context(
            None,
            {"completion_tool_names": ("declared_tool",)},
            settings,
        )

        assert ctx.completion_tool_names == ("declared_tool",)


class TestLoopConfigParity:
    def test_empty_construction_matches_buffered_hand_built_literal(self):
        """No declared overrides/context -> to_loop_config equals the buffered
        site's hand-built no-override dict byte-for-byte (dict equality against
        the literal that shipped before the consolidation)."""
        from victor.framework.agentic_loop_executor import AgenticLoopExecutor

        ctx = CallerRunContext()

        assert ctx.to_loop_config(
            executor=AgenticLoopExecutor.STANDARD,
            completion_strategy="enhanced",
            enable_effect_gate=False,
        ) == {
            "executor": AgenticLoopExecutor.STANDARD,
            "completion_strategy": "enhanced",
            "enable_effect_gate": False,
            "enable_criteria_gate": False,
        }
        assert ctx.to_runtime_overrides() == {}

    def test_empty_criteria_declaration_keeps_gate_off(self):
        ctx = CallerRunContext(criteria={})

        assert ctx.to_loop_config()["enable_criteria_gate"] is False

    def test_declared_fields_fill_defaults_the_site_did_not_resolve(self):
        ctx = CallerRunContext(executor="standard", completion_strategy="rubric")

        config = ctx.to_loop_config()
        assert config["executor"] == "standard"
        assert config["completion_strategy"] == "rubric"

    def test_site_resolved_defaults_win_over_declarations(self):
        """The build site's governance-checked executor and calibrated
        completion strategy are authoritative; declared fields only fill keys
        the site did not resolve."""
        ctx = CallerRunContext(executor="standard", completion_strategy="rubric")

        config = ctx.to_loop_config(executor="stategraph", completion_strategy="enhanced")
        assert config["executor"] == "stategraph"
        assert config["completion_strategy"] == "enhanced"


class TestRuntimeOverrides:
    def test_partial_budget_declaration_only_carries_set_keys(self):
        ctx = CallerRunContext(iteration_budget=32)

        assert ctx.to_runtime_overrides() == {"iteration_budget": 32}

    def test_zero_budget_is_carried(self):
        ctx = CallerRunContext(tool_budget=0)

        assert ctx.to_runtime_overrides() == {"tool_budget": 0}


class TestMergeThroughSharedHelper:
    def test_object_budgets_merge_with_derived_overrides_without_loss(self):
        """The object's budget keys and a derived override map merge through
        the shared PR-3 helper; caller-bound keys win and nothing is dropped."""
        from victor.core.runtime_context import merge_runtime_context_overrides

        ctx = CallerRunContext(tool_budget=134, iteration_budget=64)

        merged = merge_runtime_context_overrides(
            ctx.to_runtime_overrides(),
            {"tool_budget": 50, "provider_hint": "smart-router"},
        )

        assert merged == {
            "tool_budget": 134,
            "iteration_budget": 64,
            "provider_hint": "smart-router",
        }


class TestRunContextReconstruction:
    def test_to_run_context_round_trips_declarations_and_extra(self):
        ctx = CallerRunContext.from_overrides_and_context(
            {"tool_budget": 134},
            {
                "criteria": {"findings": 30},
                "complete_on_key": "corpus_closed",
                "unparsed": {"nested": True},
            },
            None,
        )

        assert ctx.to_run_context() == {
            "criteria": {"findings": 30},
            "complete_on_key": "corpus_closed",
            "unparsed": {"nested": True},
        }

    def test_empty_criteria_declaration_stays_present(self):
        ctx = CallerRunContext(criteria={})

        assert ctx.to_run_context() == {"criteria": {}}

    def test_empty_context_reconstructs_empty(self):
        assert CallerRunContext().to_run_context() == {}
