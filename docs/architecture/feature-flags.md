# Feature Flags (generated)

> **Generated file — do not edit by hand.** Regenerate with `python scripts/gen_feature_flag_doc.py`. A CI guard (`tests/unit/runtime/test_feature_flag_manifest_guard.py`) fails if this drifts from `victor.core.feature_flags.FeatureFlag`.
>
> A flag's **code default** is OFF iff it is in `is_opt_in_by_default()` (assuming no YAML/env override). This table is the single source of truth for flag defaults — cite it instead of restating defaults in prose, which drifts (see F-016 / TD-17).

Total flags: 26 · Opt-in (default OFF): 12 · Default ON: 14

| Flag | Code default | Opt-in |
|------|--------------|--------|
| `tool_strategy_v2` | OFF | yes |
| `use_calibrated_completion` | OFF | yes |
| `use_ccg` | ON | no |
| `use_composition_over_inheritance` | ON | no |
| `use_e3_tir_exploration` | OFF | yes |
| `use_edge_model` | ON | no |
| `use_graph_embeddings` | OFF | yes |
| `use_graph_enhanced_context` | ON | no |
| `use_graph_query_tool` | ON | no |
| `use_graph_rag` | ON | no |
| `use_learning_from_execution` | ON | no |
| `use_llm_decision_service` | ON | no |
| `use_multi_hop_retrieval` | ON | no |
| `use_policy_engine` | OFF | yes |
| `use_prime_memory_evolution` | OFF | yes |
| `use_prompt_completeness_guard` | OFF | yes |
| `use_prompt_dictionary_compression` | OFF | yes |
| `use_rich_formatting` | ON | no |
| `use_semantic_response_cache` | OFF | yes |
| `use_session_ledger` | ON | no |
| `use_smart_routing` | ON | no |
| `use_stage_transition_coordinator` | OFF | yes |
| `use_stategraph_agentic_loop` | OFF | yes |
| `use_strategy_based_tool_registration` | ON | no |
| `use_tiered_classification` | OFF | yes |
| `use_tool_result_dedup` | ON | no |
