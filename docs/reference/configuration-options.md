# Configuration Options Reference

Complete reference for Victor configuration options from `victor/config/settings.py`.

## Nested Config Groups

Settings are organized into 7 typed nested config groups for better organization. Both flat and nested access are supported:

```python
from victor.config.settings import Settings

settings = Settings(default_provider="anthropic")

# Flat access (backward compatible)
settings.default_provider          # "anthropic"

# Nested access (new, preferred for typed grouping)
settings.provider.default_provider  # "anthropic"
settings.tools.tool_retry_enabled   # True
settings.resilience.circuit_breaker_failure_threshold  # 5
settings.security.write_approval_mode  # "risky_only"
settings.search.codebase_vector_store  # "lancedb"
settings.events.event_backend_type     # "in_memory"
settings.pipeline.runtime_intelligence_enabled  # True
```

| Config Group | Fields | Purpose |
|-------------|--------|---------|
| `ProviderSettings` | 13 | Provider connection and model defaults |
| `ToolSettings` | 21 | Tool execution, selection, retry, caching |
| `SearchSettings` | 18 | Codebase search and semantic configuration |
| `ResilienceSettings` | 17 | Circuit breaker, retry, rate limiting |
| `SecuritySettings` | 19 | Server security, sandboxing, approval |
| `EventSettings` | 24 | Event system backend and configuration |
| `PipelineSettings` | 30+ | Runtime intelligence, evaluation, recovery, and compatibility aliases |

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `profiles.yaml` | `~/.victor/profiles.yaml` | Model profiles and provider settings |
| `api_keys.yaml` | `~/.victor/api_keys.yaml` | API key storage (0600 permissions) |
| `init.md` | `.victor/init.md` | Project-specific context |
| `mcp.yaml` | `.victor/mcp.yaml` | MCP server configuration |

---

## Provider Settings

### Default Provider Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `default_provider` | `ollama` | Default LLM provider |
| `default_model` | `qwen3-coder:30b` | Default model identifier |
| `default_temperature` | `0.7` | Default sampling temperature (0.0-2.0) |
| `default_max_tokens` | `4096` | Default maximum output tokens |

### Local Provider URLs

| Setting | Default | Description |
|---------|---------|-------------|
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `lmstudio_base_urls` | `["http://127.0.0.1:1234"]` | LMStudio server URLs (tries in order) |
| `vllm_base_url` | `http://localhost:8000` | vLLM server URL |

### API Keys

| Setting | Env Variable | Description |
|---------|--------------|-------------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `openai_api_key` | `OPENAI_API_KEY` | OpenAI API key |
| `google_api_key` | `GOOGLE_API_KEY` | Google (Gemini) API key |
| `moonshot_api_key` | `MOONSHOT_API_KEY` | Moonshot AI (Kimi) API key |
| `deepseek_api_key` | `DEEPSEEK_API_KEY` | DeepSeek API key |

### Provider Resilience

| Setting | Default | Description |
|---------|---------|-------------|
| `resilience_enabled` | `True` | Enable circuit breaker and retry logic |
| `circuit_breaker_failure_threshold` | `5` | Failures before circuit opens |
| `circuit_breaker_success_threshold` | `2` | Successes before circuit closes |
| `circuit_breaker_timeout` | `60.0` | Seconds before half-open state |
| `circuit_breaker_half_open_max` | `3` | Max requests in half-open state |
| `retry_max_attempts` | `3` | Maximum retry attempts |
| `retry_base_delay` | `1.0` | Base delay in seconds |
| `retry_max_delay` | `60.0` | Maximum delay between retries |
| `retry_exponential_base` | `2.0` | Exponential backoff multiplier |

### Rate Limiting

| Setting | Default | Description |
|---------|---------|-------------|
| `rate_limiting_enabled` | `True` | Enable rate limiting |
| `rate_limit_requests_per_minute` | `50` | Requests per minute limit |
| `rate_limit_tokens_per_minute` | `50000` | Tokens per minute limit |
| `rate_limit_max_concurrent` | `5` | Maximum concurrent requests |
| `rate_limit_queue_size` | `100` | Maximum pending requests in queue |

---

## Tool Settings

All tool settings are under the `tools` config group (`settings.tools.*`).

### Tool Execution & Budgets

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_call_budget` | `2000` | Maximum tool calls per session (hard limit) |
| `tool_call_budget_warning_threshold` | `1800` | Warn when approaching budget (90%) |
| `fallback_max_tools` | `8` | Cap tool list broadcast to LLM per turn (1-20, CI-guarded) |
| `tool_budget_calibration_enabled` | `false` | Opt-in RL-driven tool-budget calibration. When `true`, the session tool budget is derived from aggregated `rl_tool_q` + `decision_outcome` signals via `BudgetCalibrator`. Default `false` = unchanged behavior; low-confidence recommendations always retain the baseline. |
| `tool_budget_calibration_min_confidence` | `0.7` | Minimum calibrator confidence required to apply a calibrated budget overlay (0.0-1.0). Below this threshold the baseline budget is returned unchanged. Default raised from `0.5` to `0.7` per FEP-0002 review (requires modestly more RL evidence before applying). |

When `tool_budget_calibration_enabled` is `true`, the calibration pipeline runs
at session bootstrap:

1. `BudgetSignalReader` loads aggregated RL signals (`rl_tool_q` Q-values and
   `decision_outcome` success/failure/reward aggregates) from the global Victor
   database (`~/.victor/victor.db`).
2. `BudgetCalibrator` maps those signals to a `BudgetRecommendation` with a
   calibrated `tool_call_budget` and a `confidence` score.
3. `apply_budget_calibration()` applies the overlay only if the toggle is on
   *and* the confidence meets `tool_budget_calibration_min_confidence`.

The overlay is applied immutably (a new `ToolSettings` instance); the baseline
is never mutated. Pipeline exceptions and cold-start (confidence `<= 0.0`)
always retain the baseline, so calibration can never break session bootstrap.

Task-based budgets override the default based on detected complexity:

| Task Type | Budget | Description |
|-----------|--------|-------------|
| simple | 20 | Quick queries, single file edits |
| medium | 50 | Multi-file analysis, refactoring |
| complex | 100 | Architecture changes, migrations |
| action | 200 | Building features, large refactors |
| analysis | 500 | Deep codebase exploration, audits |

### Tool Retry

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_retry_enabled` | `True` | Enable automatic retry for failed tools |
| `tool_retry_max_attempts` | `3` | Maximum retry attempts per tool |
| `tool_retry_base_delay` | `1.0` | Base delay in seconds (exponential backoff) |
| `tool_retry_max_delay` | `10.0` | Maximum delay between retries |

### Tool Result Caching

Three independent caching layers prevent redundant tool execution:

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_cache_enabled` | `True` | Enable persistent tool result caching |
| `tool_cache_ttl` | `600` | Persistent cache TTL in seconds (10 min) |
| `tool_cache_allowlist` | `["code_search", ...]` | Tools eligible for persistent caching |
| `generic_result_cache_enabled` | `False` | Enable generic result cache |
| `generic_result_cache_ttl` | `300` | Generic result cache TTL in seconds |

**Idempotent cache** (session-level, always active): LRU cache with 50 entries and 5-min TTL for read-only tools (`read`, `ls`, `code_search`, `grep`, `graph`, `glob`). Prevents re-reading same files within a session.

**Cross-turn dedup** (session-level, configurable):

| Setting | Default | Description |
|---------|---------|-------------|
| `cross_turn_dedup_enabled` | `True` | Cache results for "effectively idempotent" tools across turns |
| `cross_turn_dedup_ttl` | `300` | Cross-turn dedup cache TTL in seconds (5 min) |

Covers tools that produce identical results for identical args within a session window but aren't classified as strictly idempotent: `web_search`, `web_fetch`, `http_request`, `grep_search`, `plan_files`, `git`. Prevents re-executing the same search or git command 2-3 turns later.

### Tool Validation

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_validation_mode` | `lenient` | Validation mode: `strict`, `lenient`, `off` |

### Tool Deduplication (Batch-Level)

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_tool_deduplication` | `True` | Prevent duplicate tool calls within a single batch |
| `tool_deduplication_window_size` | `20` | Number of recent calls to track for dedup |

### Tool Selection (Semantic)

| Setting | Default | Description |
|---------|---------|-------------|
| `use_semantic_tool_selection` | `True` | Use embeddings for tool selection |
| `embedding_provider` | `sentence-transformers` | Embedding provider |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `tool_selection_cache_enabled` | `True` | Cache semantic selection results across turns |
| `tool_selection_cache_ttl` | `300` | Selection cache TTL in seconds (5 min) |

Adaptive selection adjusts thresholds by model size:

| Model Size | Similarity Threshold | Max Tools |
|-----------|---------------------|-----------|
| tiny (0.5B-3B) | 0.35 | 5 |
| small (7B-8B) | 0.25 | 7 |
| medium (13B-15B) | 0.20 | 10 |
| large (30B+) | 0.15 | 12 |
| cloud (Claude/GPT) | 0.18 | 10 |

### Tool Schema Broadcasting

Victor uses a three-tier schema system to reduce token cost when broadcasting tool definitions to the LLM. Tools are sent via the provider's native `tools` parameter at varying verbosity levels.

**Schema levels:**

| Level | ~Tokens | Max Desc | Max Param Desc | Optional Params | Usage |
|-------|---------|----------|----------------|-----------------|-------|
| FULL | 125 | 500 chars | 100 chars | Yes | Mandatory tools (read, write, shell) |
| COMPACT | 70 | 150 chars | 50 chars | Yes | Vertical core + stage-specific tools |
| STUB | 32 | 80 chars | 25 chars | No (required only) | Semantic pool + MCP tools |

**Token budget and schema promotion:**

| Setting | Default | Description |
|---------|---------|-------------|
| `max_tool_schema_tokens` | `4000` | Max tokens for all tool schemas combined. Enforced by demoting COMPACT→STUB then dropping tail STUBs. Set `0` to disable. |
| `schema_promotion_threshold` | `0.8` | Semantic similarity above which STUB tools are promoted to COMPACT for richer schemas |

With `fallback_max_tools=8`, typical token usage is ~650 (3 FULL + 3 COMPACT + 2 STUB). The 4000 budget is a safety net that only activates with large registries (30+ tools). The tiered system, KV cache ordering, and Anthropic cache boundaries all function regardless of this budget setting.

```yaml
# Example: conservative token budget for small context models
tools:
  max_tool_schema_tokens: 2500
  schema_promotion_threshold: 0.85
  fallback_max_tools: 6
```

### MCP Tool Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `max_mcp_tools_per_turn` | `12` | Max MCP tools broadcast when relevance filtering is active |

MCP tools are registered at STUB schema level by default. When `MCPToolProjector.project()` is called with a `user_message`, only tools whose name/description match the query are included, capped at this limit. Without a user message, all MCP tools are included (backward compatible).

### Parallel Tool Execution

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_parallel_execution` | `True` | Enable parallel tool execution |
| `max_concurrent_tools` | `5` | Maximum concurrent tool calls |
| `parallel_batch_size` | `10` | Batch size for parallel execution |
| `parallel_timeout_per_tool` | `60.0` | Timeout per tool in parallel batch (seconds) |

Writes to different files parallelize automatically via a per-file dependency graph. Same-file writes serialize. Reads that depend on a prior write to the same file wait for the write to complete. Writes with no extractable file path conservatively serialize against all prior writes.

---

## Workflow Settings

### Workflow Definition Cache

| Setting | Default | Description |
|---------|---------|-------------|
| `workflow_definition_cache_enabled` | `True` | Cache parsed YAML workflows |
| `workflow_definition_cache_ttl` | `3600` | Cache TTL in seconds |
| `workflow_definition_cache_max_entries` | `100` | Maximum cached entries |

### StateGraph

| Setting | Default | Description |
|---------|---------|-------------|
| `stategraph_copy_on_write_enabled` | `True` | Enable copy-on-write for state |

### Checkpoints

| Setting | Default | Description |
|---------|---------|-------------|
| `checkpoint_enabled` | `True` | Enable checkpoint system |
| `checkpoint_auto_interval` | `5` | Tool calls between auto-checkpoints |
| `checkpoint_max_per_session` | `50` | Maximum checkpoints per session |
| `checkpoint_compression_enabled` | `True` | Compress checkpoint data |
| `checkpoint_compression_threshold` | `1024` | Min bytes before compression |

### Human-in-the-Loop (HITL)

| Setting | Default | Description |
|---------|---------|-------------|
| `hitl_default_timeout` | `300.0` | Default timeout in seconds |
| `hitl_default_fallback` | `abort` | Action on timeout: `abort`, `continue`, `skip` |
| `hitl_auto_approve_low_risk` | `False` | Auto-approve LOW risk actions |
| `hitl_keyboard_shortcuts_enabled` | `True` | Enable y/n shortcuts in interactive CLI prompts |

---

## Safety Settings

### Write Approval

| Setting | Default | Description |
|---------|---------|-------------|
| `write_approval_mode` | `risky_only` | Approval mode: `off`, `risky_only`, `all_writes` |

### Headless/Automation Mode

| Setting | Default | Description |
|---------|---------|-------------|
| `headless_mode` | `False` | Run without prompts |
| `dry_run_mode` | `False` | Preview changes without applying |
| `auto_approve_safe` | `False` | Auto-approve read-only and LOW risk ops |
| `max_file_changes` | `None` | Limit file modifications per session |
| `one_shot_mode` | `False` | Exit after completing single request |

### Air-Gapped Mode

| Setting | Default | Description |
|---------|---------|-------------|
| `airgapped_mode` | `False` | Only allow local providers, no web tools |

### Security Scans

| Setting | Default | Description |
|---------|---------|-------------|
| `security_dependency_scan` | `False` | Enable dependency vulnerability scanning |
| `security_iac_scan` | `False` | Enable infrastructure-as-code scanning |

---

## Performance Settings

### Context Management

| Setting | Default | Description |
|---------|---------|-------------|
| `max_context_tokens` | `100000` | Maximum tokens in context window |
| `response_token_reserve` | `4096` | Tokens reserved for model response |

### Cache & KV Prefix Optimization

These settings control how Victor optimizes for provider-level prompt caching (API billing discounts) and KV prefix stability (latency savings). They live under `context`:

| Setting | Default | Description |
|---------|---------|-------------|
| `cache_optimization_enabled` | `True` | API prompt caching: lock all tools at session start for billing discount (90% on Anthropic, 50% on DeepSeek) |
| `kv_optimization_enabled` | `True` | KV prefix stability: freeze system prompt, sort tools deterministically for prefix matching |
| `kv_tool_strategy` | `per_turn` | Tool selection strategy for KV providers: `per_turn` (fresh selection, breaks prefix) or `session_stable` (lock after first query, stable prefix) |
| `tiered_schema_enabled` | `True` | Enable FULL/COMPACT/STUB schema levels for tiered broadcasting |
| `tool_approval_mode` | `auto` | HITL tool approval: `auto` (all approved), `dangerous` (MEDIUM+ require approval), `all` (every call) |

**How they interact:**

| Provider Type | API Cache | KV Cache | Tool Strategy |
|--------------|-----------|----------|---------------|
| Cloud (Anthropic, OpenAI, DeepSeek) | Yes — 90% discount | Yes — latency savings | Session-locked (full set cached) |
| Local (Ollama, LMStudio, vLLM) | No | Yes — KV prefix stability | Configurable (`per_turn` or `session_stable`) |
| Unknown/custom | No | No | Fresh selection each turn |

The system prompt is **frozen** after first build for providers that support caching (Tier A/B). Dynamic content (skills, reminders, credit guidance) is injected into the **user message** instead, keeping the system prompt byte-identical across turns.

```yaml
# Example: disable KV optimization for debugging
context:
  kv_optimization_enabled: false
  kv_tool_strategy: per_turn
```

### Context Compaction

| Setting | Default | Description |
|---------|---------|-------------|
| `context_compaction_strategy` | `tiered` | Strategy: `simple`, `tiered`, `semantic`, `hybrid` |
| `context_min_messages_to_keep` | `6` | Minimum messages after compaction |
| `context_tool_retention_weight` | `1.5` | Boost for tool result retention |
| `context_recency_weight` | `2.0` | Boost for recent messages |
| `context_semantic_threshold` | `0.3` | Min similarity for semantic retention |

### Exploration Limits

| Setting | Default | Description |
|---------|---------|-------------|
| `max_exploration_iterations` | `200` | Base iteration limit (multiplied by mode) |
| `max_exploration_iterations_action` | `500` | Limit for action tasks |
| `max_exploration_iterations_analysis` | `1000` | Limit for analysis tasks |
| `min_content_threshold` | `50` | Min chars for "substantial" output |
| `max_research_iterations` | `50` | Limit for web research |

### Recovery Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `recovery_empty_response_threshold` | `5` | Force after N empty responses |
| `recovery_blocked_consecutive_threshold` | `6` | Force after N consecutive blocks |
| `recovery_blocked_total_threshold` | `9` | Force after N total blocked attempts |
| `max_continuation_prompts_analysis` | `6` | Continuation prompts for analysis |
| `max_continuation_prompts_action` | `5` | Continuation prompts for actions |
| `max_continuation_prompts_default` | `3` | Default continuation prompts |

### Session Timeout

| Setting | Default | Description |
|---------|---------|-------------|
| `session_idle_timeout` | `180` | Seconds of inactivity before forcing completion |

### Streaming Metrics

| Setting | Default | Description |
|---------|---------|-------------|
| `streaming_metrics_enabled` | `True` | Enable streaming performance metrics |
| `streaming_metrics_history_size` | `1000` | Samples to retain |

---

## Codebase Search Settings

### Vector Store

| Setting | Default | Description |
|---------|---------|-------------|
| `codebase_vector_store` | `lancedb` | Vector store: `lancedb`, `proximadb` |
| `codebase_embedding_provider` | `sentence-transformers` | Embedding provider |
| `codebase_embedding_model` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `codebase_persist_directory` | `None` | Custom persistence path |
| `codebase_dimension` | `384` | Embedding dimension |
| `codebase_batch_size` | `32` | Batch size for embedding generation |

### Graph Store

| Setting | Default | Description |
|---------|---------|-------------|
| `codebase_graph_store` | `sqlite` | Graph backend |
| `codebase_graph_path` | `None` | Custom graph db path |
| `codebase_graph_writer_mode` | `off` | `off` for victor-owned graph writes, `compatibility` (deprecated) for victor-coding compatibility writes |

### Semantic Search Quality

| Setting | Default | Description |
|---------|---------|-------------|
| `semantic_similarity_threshold` | `0.25` | Min score for results |
| `semantic_query_expansion_enabled` | `True` | Expand queries with synonyms |
| `semantic_max_query_expansions` | `5` | Max query variations |

### Hybrid Search

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_hybrid_search` | `False` | Enable semantic + keyword search |
| `hybrid_search_semantic_weight` | `0.6` | Weight for semantic search |
| `hybrid_search_keyword_weight` | `0.4` | Weight for keyword search |

---

## Server Settings

### HTTP/WebSocket Server

| Setting | Default | Description |
|---------|---------|-------------|
| `server_api_key` | `None` | API key for HTTP/WebSocket auth |
| `server_session_secret` | `None` | HMAC secret for session tokens |
| `server_max_sessions` | `100` | Max simultaneous sessions |
| `server_max_message_bytes` | `32768` | Max WebSocket message size |
| `server_session_ttl_seconds` | `86400` | Session token TTL |

### Diagram Rendering

| Setting | Default | Description |
|---------|---------|-------------|
| `render_max_payload_bytes` | `20000` | Max diagram payload size |
| `render_timeout_seconds` | `10` | Render timeout |
| `render_max_concurrency` | `2` | Max concurrent renders |

### Code Execution Sandbox

| Setting | Default | Description |
|---------|---------|-------------|
| `code_executor_network_disabled` | `True` | Disable network in sandbox |
| `code_executor_memory_limit` | `512m` | Memory limit |
| `code_executor_cpu_shares` | `256` | CPU shares |

---

## UI Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `theme` | `monokai` | Color theme |
| `show_token_count` | `True` | Display token counts |
| `show_cost_metrics` | `False` | Show cost in metrics |
| `stream_responses` | `True` | Enable streaming output |
| `use_emojis` | `True` | Enable emoji indicators |

---

## Vertical Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `default_vertical` | `coding` | Default domain vertical |
| `auto_detect_vertical` | `False` | Auto-detect from project context |

---

## Conversation Memory

| Setting | Default | Description |
|---------|---------|-------------|
| `conversation_memory_enabled` | `True` | Enable SQLite persistence |
| `conversation_embeddings_enabled` | `True` | Enable LanceDB embeddings |

---

## Prompt Enrichment

| Setting | Default | Description |
|---------|---------|-------------|
| `prompt_enrichment_enabled` | `True` | Enable prompt enrichment |
| `prompt_enrichment_max_tokens` | `2000` | Max tokens to add |
| `prompt_enrichment_timeout_ms` | `500.0` | Timeout in milliseconds |
| `prompt_enrichment_cache_enabled` | `True` | Cache enrichments |
| `prompt_enrichment_cache_ttl` | `300` | Cache TTL in seconds |
| `prompt_enrichment_strategies` | `["knowledge_graph", ...]` | Enabled strategies |

---

## Runtime Intelligence And Evaluation

These settings live under `settings.pipeline.*`. The canonical enablement flag
is `runtime_intelligence_enabled`; several related compatibility aliases retain
the older "intelligent pipeline" wording for backward compatibility even though
the live runtime now flows through the service-owned chat runtime and
`StreamingChatExecutor`.

| Setting | Default | Description |
|---------|---------|-------------|
| `runtime_intelligence_enabled` | `True` | Master switch for runtime-intelligence features on the live chat/runtime path |
| `intelligent_quality_scoring` | `True` | Enable quality scoring |
| `intelligent_mode_learning` | `True` | Enable Q-learning for modes |
| `intelligent_prompt_optimization` | `True` | Enable prompt optimization |
| `intelligent_grounding_verification` | `True` | Enable hallucination detection |
| `intelligent_min_quality_threshold` | `0.5` | Minimum quality score |
| `intelligent_grounding_threshold` | `0.7` | Grounding confidence threshold |
| `intelligent_exploration_rate` | `0.3` | Q-learning exploration rate |
| `intelligent_learning_rate` | `0.1` | Q-learning alpha |
| `intelligent_discount_factor` | `0.9` | Q-learning gamma |

---

## Serialization

| Setting | Default | Description |
|---------|---------|-------------|
| `serialization_enabled` | `True` | Enable token-optimized serialization |
| `serialization_default_format` | `None` | Auto-select best format |
| `serialization_min_savings_threshold` | `0.15` | Min savings for alt format |
| `serialization_include_format_hint` | `True` | Include format description |
| `serialization_min_rows_for_tabular` | `3` | Min rows for tabular formats |
| `serialization_debug_mode` | `False` | Include data characteristics |

---

## Event System

| Setting | Default | Description |
|---------|---------|-------------|
| `event_backend_type` | `in_memory` | Backend: `in_memory`, `sqlite`, `redis`, etc. |
| `event_delivery_guarantee` | `at_most_once` | Delivery: `at_most_once`, `at_least_once`, `exactly_once` |
| `event_max_batch_size` | `100` | Max batch size |
| `event_flush_interval_ms` | `1000.0` | Flush interval in ms |
| `event_queue_maxsize` | `10000` | In-memory backend queue bound |
| `event_queue_overflow_policy` | `drop_newest` | Default overflow policy (`drop_newest`, `drop_oldest`, `block_with_timeout`) |
| `event_queue_overflow_block_timeout_ms` | `50.0` | Default timeout for `block_with_timeout` policy |
| `event_queue_overflow_topic_policies` | `{lifecycle.session.*: block_with_timeout, vertical.applied: block_with_timeout, error.*: block_with_timeout, core.events.emit_sync.metrics: drop_oldest, vertical.extensions.loader.metrics: drop_oldest}` | Per-topic policy overrides (supports `*` wildcard) |
| `event_queue_overflow_topic_block_timeout_ms` | `{lifecycle.session.*: 150.0, vertical.applied: 120.0, error.*: 200.0}` | Per-topic timeout overrides for `block_with_timeout` |

---

## Plugin System

| Setting | Default | Description |
|---------|---------|-------------|
| `plugin_enabled` | `True` | Enable plugin system |
| `plugin_packages` | `[]` | Python packages to load |
| `plugin_disabled` | `[]` | Plugin names to disable |
| `plugin_config` | `{}` | Per-plugin configuration |

---

## Observability

| Setting | Default | Description |
|---------|---------|-------------|
| `log_level` | `INFO` | Logging level |
| `log_file` | `None` | Log file path |
| `enable_observability_logging` | `False` | Write events to JSONL |
| `observability_log_path` | `None` | Custom JSONL path |
| `analytics_enabled` | `True` | Enable analytics |

---

## Profile Configuration (profiles.yaml)

Profiles in `~/.victor/profiles.yaml` can override many of these settings per-model:

```yaml
profiles:
  default:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.7
    max_tokens: 4096
    description: "Default local model"

    # Provider tuning (per-profile overrides)
    loop_repeat_threshold: 3
    max_continuation_prompts: 5
    quality_threshold: 0.6
    grounding_threshold: 0.7
    max_tool_calls_per_turn: 10
    tool_cache_enabled: true
    tool_deduplication_enabled: true
    session_idle_timeout: 300
    timeout: 600

    # Tool selection tuning
    tool_selection:
      model_size_tier: medium  # tiny, small, medium, large, cloud
      base_threshold: 0.35
      base_max_tools: 12

  anthropic:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.5
    max_tokens: 8192
    tool_selection:
      model_size_tier: cloud

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  openai:
    api_key: ${OPENAI_API_KEY}
```
