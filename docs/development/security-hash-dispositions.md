# B324 disposition for security closeout

Audited all 52 B324 findings in the unconfigured Bandit 1.9.4 security closeout
baseline, covering 39 production files. Every call
is a non-security identifier, lookup key, content-deduplication fingerprint,
or deterministic statistical feature. No call authenticates a user, validates
an executable/download, verifies a signature, stores a password, or establishes
an integrity/authorization boundary.

The fix explicitly supplies `usedforsecurity=False` at each audited constructor
and removes B324 from the global `.bandit` and CI command skip lists. Algorithm,
digest bytes, truncation, persisted IDs, and cache paths are unchanged; no migration is needed.
Validation verified that removing the new keyword from each changed AST
reproduces the original AST exactly. This is classification and explicit API
intent, not a claim that MD5/SHA-1 became collision-resistant.

| File under `victor/` | Calls | Reviewed purpose |
| --- | ---: | --- |
| agent/conversation/assembler.py | 1 | Normalized conversation-message deduplication |
| agent/deferred_tool_loading.py | 1 | In-memory deferred result lookup ID |
| agent/output_aggregator.py | 1 | Tool-argument aggregation/deduplication |
| agent/output_deduplicator.py | 2 | Normalized output-block deduplication in two implementations |
| agent/prompt_normalizer.py | 1 | Redundant prompt-section comparison |
| agent/prompt_section_allocator.py | 1 | Prompt-selection decision cache key |
| agent/read_cache.py | 1 | Detect unchanged cached read content before replacing entry |
| agent/semantic_response_cache.py | 1 | Query/context response-cache lookup key |
| agent/services/decision_service.py | 1 | LLM decision-result cache key |
| agent/services/prompt_builder_runtime.py | 1 | Short KV-cache prefix diagnostic fingerprint |
| agent/services/turn_execution_runtime.py | 1 | Deterministic tool-call correlation/envelope ID |
| agent/session_id.py | 1 | Human-readable project prefix; independent random suffix identifies session |
| agent/session_state_manager.py | 1 | Previously failed tool-call signature matching to stop retries |
| agent/thinking_detector.py | 1 | Repeated reasoning/loop detection |
| agent/tool_pipeline.py | 2 | Tool-call ID and observability intent argument fingerprint |
| agent/turn_policy.py | 1 | Repeated-content stopping heuristic |
| agent/unified_classifier.py | 1 | Message-classification cache key |
| agent/unified_task_tracker.py | 1 | Context-aware repeated-tool signature |
| core/query_enhancement/pipeline.py | 1 | Query/context enhancement cache key |
| evaluation/agent_adapter.py | 1 | Same-error circuit-breaker grouping |
| evaluation/experiment_analyzer.py | 1 | Evaluation-memory record ID |
| evaluation/harness.py | 1 | Deterministic evaluation checkpoint filename |
| evaluation/rl_test_optimizer.py | 1 | File-path key in persisted learned test-dependency Q-values |
| evaluation/runtime_feedback.py | 1 | Scope-key filename for feedback artifacts; no content authentication |
| evaluation/swe_bench_loader.py | 6 | Benchmark workspace name and five matching repository-cache lookups |
| framework/framework_integration_registry_service.py | 1 | Skip duplicate equivalent registrations within a scope |
| framework/lsp_context.py | 1 | Symbol/diagnostic throttling fingerprint |
| framework/rl/experiment_coordinator.py | 1 | Stable experiment cohort assignment |
| framework/rl/learners/prompt_optimizer.py | 3 | Prompt-candidate and parent record identifiers, not authenticity proofs |
| framework/rl/pareto.py | 1 | Merged candidate ID matching optimizer registry |
| framework/rl/shared_encoder.py | 2 | Deterministic task/provider feature vectors |
| framework/search/codebase_embedding_bridge.py | 1 | Search chunk document ID from source coordinates |
| observability/debouncing/debouncer.py | 2 | Event/metadata deduplication keys |
| observability/emitters/error_emitter.py | 2 | Repeated-warning grouping and suppressed-count lookup |
| processing/native/deduplication.py | 2 | Python output-block and tool-signature deduplication fallbacks |
| providers/openai_compat.py | 1 | Serialized tool-schema stability diagnostic |
| storage/memory/unified.py | 1 | Default memory-result ID for result grouping |
| tools/query_cache.py | 1 | Compact representation of long query lookup keys |
| tools/verification/claim_verifier.py | 1 | Content fingerprint emitted as evidence metadata; not compared as proof of integrity |

The benchmark paths, feedback artifacts, and candidate registry remain trusted
local storage mechanisms; these hashes do not sandbox their files or authenticate
their contents. Existing independent integrity controls are outside these 52 calls.
