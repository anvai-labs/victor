# Victor Settings Reference

Comprehensive reference for all configurable settings. Settings are loaded from `~/.victor/profiles.yaml`, environment variables, or CLI flags.

## Quick Reference

| Setting | Default | Env Var | CLI Flag |
|---------|---------|---------|----------|
| provider/model defaults | via `~/.victor/profiles.yaml` profiles (`default` profile) | — | `--profile`, `--model` |
| `tool_call_budget` | `2000` | — | `--tool-budget` |
| `enable_planning` | `false` | — | `--planning/--no-planning` |
| `skill_auto_select_enabled` | `true` | — | `--auto-skill/--no-auto-skill` |
| `airgapped_mode` | `false` | `AIRGAPPED_MODE` | `--airgapped` |

## Provider & Model

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.7
    max_tokens: 4096

  cloud:
    provider: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.3
    max_tokens: 8192

  fast:
    provider: xai
    model: grok-3-mini-fast
    temperature: 0.5
```

**Supported providers**: `ollama`, `anthropic`, `openai`, `deepseek`, `xai`, `google`, `lmstudio`, `vllm`, `moonshot`

**Usage**:
```bash
victor chat --profile cloud         # Use cloud profile
victor chat --provider anthropic    # Override provider
victor chat --model gpt-4.1         # Override model
```

### Local Server URLs

```yaml
# Environment variables
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URLS=http://192.168.1.100:1234,http://localhost:1234
VLLM_BASE_URL=http://localhost:8000
```

## API Keys

Set via environment variables (never commit to config files):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AIza...
export DEEPSEEK_API_KEY=sk-...
export XAI_API_KEY=xai-...
```

Check configured keys:
```bash
victor keys check
```

## Tools & Budgets

| Setting | Default | Description |
|---------|---------|-------------|
| `tool_call_budget` | `2000` | Max tool calls per session |
| `fallback_max_tools` | `8` | Cap tool list when stage pruning removes all (1-20, CI-guarded) |
| `tool_retry_enabled` | `true` | Auto-retry failed tool executions |
| `tool_retry_max_attempts` | `3` | Max retry attempts per tool call |
| `tool_retry_base_delay` | `1.0` | Base delay (seconds) for exponential backoff |
| `tool_retry_max_delay` | `10.0` | Max delay between retries |

```bash
victor chat --tool-budget 50 "quick fix"   # Limit tool calls
```

## Tool Schema Broadcasting & Token Optimization

Controls how tool definitions are sent to the LLM and how tokens are managed. Victor uses a tiered schema system (FULL/COMPACT/STUB) with KV-cache-aware ordering.

### Schema Tiering

Tool schemas are broadcast at three verbosity levels to reduce token cost:

| Level | ~Tokens | Description Chars | Param Desc Chars | Optional Params |
|-------|---------|-------------------|------------------|-----------------|
| FULL | 125 | 500 | 100 | Yes |
| COMPACT | 70 | 150 | 50 | Yes |
| STUB | 32 | 80 | 25 | No (required only) |

Tools are assigned levels by `TieredToolConfig` tier membership:
- **Mandatory** tools (read, write, shell) → FULL
- **Vertical core** tools → COMPACT
- **Semantic pool** tools → STUB

### Schema Token Budget

| Setting | Default | Description |
|---------|---------|-------------|
| `max_tool_schema_tokens` | `4000` | Max tokens for all tool schemas combined. When exceeded, COMPACT tools demote to STUB, then tail STUBs are dropped. Set `0` to disable. |
| `schema_promotion_threshold` | `0.8` | Semantic similarity above which STUB tools promote to COMPACT for richer schemas |

With `fallback_max_tools=8` (default), typical usage is ~650 tokens — well under the 4000 budget. The budget is a safety net for large registries (30+ tools).

```yaml
# Profile: conservative token usage
tools:
  max_tool_schema_tokens: 2500
  schema_promotion_threshold: 0.85
```

### Cache Optimization

| Setting | Default | Description |
|---------|---------|-------------|
| `cache_optimization_enabled` | `true` | API prompt caching — lock tools for billing discount (90% on Anthropic) |
| `kv_optimization_enabled` | `true` | KV prefix stability — freeze system prompt, sort tools deterministically |
| `kv_tool_strategy` | `per_turn` | `per_turn` = fresh selection each turn; `session_stable` = lock after first query |
| `tiered_schema_enabled` | `true` | Enable FULL/COMPACT/STUB schema levels |

These settings live in `context` (not `tools`):

```yaml
context:
  cache_optimization_enabled: true
  kv_optimization_enabled: true
  kv_tool_strategy: per_turn
  tiered_schema_enabled: true
```

### MCP Tool Filtering

| Setting | Default | Description |
|---------|---------|-------------|
| `max_mcp_tools_per_turn` | `12` | Max MCP tools broadcast per turn when relevance filtering is active |

MCP tools default to STUB schema level. When relevance filtering is active, only MCP tools matching the query are broadcast (keyword overlap on name + description), capped at this limit.

## Tool Selection

| Setting | Default | Description |
|---------|---------|-------------|
| `use_semantic_tool_selection` | `true` | Embedding-based tool selection |
| `embedding_provider` | `sentence-transformers` | Embedding provider |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `tool_selection_cache_enabled` | `true` | Cache selection results across turns |
| `tool_selection_cache_ttl` | `300` | Selection cache TTL (seconds) |

## Tool Deduplication

Automatically removes duplicate tools across native, LangChain, and MCP sources with priority-based resolution.

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_tool_deduplication` | `true` | Enable cross-source tool deduplication |
| `deduplication_priority_order` | `["native", "langchain", "mcp", "plugin"]` | Priority order (highest to lowest) |
| `deduplication_whitelist` | `[]` | Tools to always allow (bypass deduplication) |
| `deduplication_blacklist` | `[]` | Tools to always skip (force deduplication) |
| `deduplication_strict_mode` | `false` | Fail on conflicts instead of logging and skipping |
| `deduplication_naming_enforcement` | `true` | Enforce naming conventions (lgc_*, mcp_*, plg_*) |
| `deduplication_semantic_threshold` | `0.85` | Threshold for semantic similarity detection (0.0-1.0) |

### Priority System

Tools are prioritized by source when conflicts are detected:

| Priority | Source | Prefix | Examples |
|----------|--------|--------|----------|
| 1 (highest) | Native | none | `read`, `write`, `edit`, `search` |
| 2 | LangChain | `lgc_` | `lgc_wikipedia`, `lgc_wolfram_alpha` |
| 3 | MCP | `mcp_` | `mcp_github_search`, `mcp_filesystem_read` |
| 4 (lowest) | Plugin | `plg_` | `plg_custom_tool` |

When two tools have the same normalized name (e.g., `search` vs `lgc_search`), the higher priority source is kept and the lower priority tool is automatically skipped during registration.

### Token Savings

Deduplication provides token savings through two mechanisms:

1. **Native tool preference**: No wrapper overhead, full schema control
2. **Adapter tool STUB schemas**: LangChain and MCP tools use STUB schemas (57% reduction vs FULL)

### Configuration Examples

```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    tools:
      enable_tool_deduplication: true
      deduplication_priority_order:
        - native
        - langchain
        - mcp
        - plugin
      deduplication_naming_enforcement: true

  # Disable deduplication (allow all tools)
  no_dedup:
    tools:
      enable_tool_deduplication: false

  # Custom whitelist (always allow specific tools)
  custom:
    tools:
      deduplication_whitelist:
        - special_tool
        - custom_search
```

### Usage

Tool deduplication is always on (native fast path with a pure-Python fallback;
there is no toggle). Check the active set:

```bash
victor tools list  # Shows the deduplicated tool set
```

### Adaptive Selection by Model Size

| Model Size | Threshold | Max Tools |
|-----------|-----------|-----------|
| tiny (0.5B-3B) | 0.35 | 5 |
| small (7B-8B) | 0.25 | 7 |
| medium (13B-15B) | 0.20 | 10 |
| large (30B+) | 0.15 | 12 |
| cloud (Claude/GPT) | 0.18 | 10 |

## Tool Execution Pipeline

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_tool_deduplication` | `true` | Prevent duplicate tool calls within a batch |
| `tool_deduplication_window_size` | `20` | Number of recent calls to track for dedup |
| `cross_turn_dedup_enabled` | `true` | Cache results for effectively-idempotent tools across turns |
| `cross_turn_dedup_ttl` | `300` | Cross-turn dedup cache TTL (seconds) |

Cross-turn dedup covers tools that produce identical results for identical args within a session window: `web_search`, `web_fetch`, `http_request`, `grep_search`, `plan_files`, `git`.

### Parallel Execution

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_parallel_execution` | `true` | Enable parallel tool execution |
| `max_concurrent_tools` | `5` | Max concurrent tool calls |
| `parallel_timeout_per_tool` | `60.0` | Timeout per tool in parallel batch |

Writes to different files parallelize automatically. Same-file writes serialize via dependency graph.

## Skills & Auto-Selection

Skills are composable expertise units that inject focused prompts + tools based on the user's message. Auto-selection uses embedding similarity to match messages to skills.

| Setting | Default | Description |
|---------|---------|-------------|
| `skill_auto_select_enabled` | `true` | Enable embedding-based skill matching |
| `skill_auto_select_high_threshold` | `0.65` | Above: use skill directly (high confidence) |
| `skill_auto_select_low_threshold` | `0.45` | Below: no skill injected |
| `skill_auto_select_use_edge_fallback` | `true` | Use edge LLM for ambiguous matches (0.45-0.65) |
| `skill_auto_select_log_selections` | `true` | Log which skill was selected |

```bash
victor chat "fix the test" --auto-skill     # Force skill selection on
victor chat "hello" --no-auto-skill         # Disable for this session
victor skill list                           # See available skills
victor skill create my_skill ...            # Create custom skill (YAML)
```

User skills are stored at `~/.victor/skills/*.yaml`. See `victor skill --help` for management commands.

## Prompt Optimization (GEPA)

Runtime evolution of system prompt sections using execution trace analysis.

| Setting | Default | Description |
|---------|---------|-------------|
| `prompt_optimization.enabled` | `false` | Master switch for all prompt optimization |
| `prompt_optimization.default_strategies` | `["gepa"]` | Strategies applied in order |
| `prompt_optimization.gepa.enabled` | `true` | Enable GEPA v2 (Pareto + tiered service) |
| `prompt_optimization.gepa.default_tier` | `balanced` | Default tier: economic/balanced/performance |

### GEPA Model Tiers

| Tier | Provider/Model | Use Case |
|------|---------------|----------|
| economic | `ollama/gemma4` | Post-convergence maintenance |
| balanced | `openai/gpt-4.1-mini` | Default — active optimization |
| performance | `anthropic/claude-sonnet-4-20250514` | Initial convergence, regressions |

Auto-switches based on convergence metrics. Three evolvable sections:
- `ASI_TOOL_EFFECTIVENESS_GUIDANCE`
- `GROUNDING_RULES`
- `COMPLETION_GUIDANCE`

## Decision Service (Tiered)

Routes different decision types to different model tiers. Fallback always goes DOWN in cost.

| Setting | Default | Description |
|---------|---------|-------------|
| `DecisionServiceSettings.enabled` | `true` | Enable tiered routing |
| `DecisionServiceSettings.edge` | `ollama/qwen3.5:2b` | Fast local micro-decisions |
| `DecisionServiceSettings.balanced` | `deepseek/deepseek-chat` | Mid-tier cloud decisions |
| `DecisionServiceSettings.performance` | `anthropic/claude-sonnet` | Frontier (opt-in only) |

### Default Routing

| Decision Type | Default Tier | Purpose |
|--------------|-------------|---------|
| `tool_selection` | edge | Select which tools to broadcast |
| `tool_necessity` | edge | Skip tools entirely for Q&A turns (saves ~2-4K tokens) |
| `skill_selection` | edge | Match user message to skills |
| `intent_classification` | edge | Classify model response intent |
| `task_completion` | edge | Detect when task is done |
| `error_classification` | edge | Classify error types for recovery |
| `stage_detection` | edge | Detect conversation stage transitions |
| `task_type_classification` | balanced | Classify task type with deliverables |
| `multi_skill_decomposition` | balanced | Decompose complex requests into skills |

**Tool Necessity gate**: Before tool selection runs, a fast heuristic checks if the message is pure Q&A (greeting, explanation request, clarification). If the heuristic is confident (keyword scan), tools are skipped immediately. For ambiguous cases, the edge model is consulted via `TOOL_NECESSITY` decision type. This saves ~2-4K tokens per conversational turn (~30-40% of messages).

Performance tier is never used by default — opt-in via custom `tier_routing` config.

## Planning

Structured task decomposition for complex multi-step requests.

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_planning` | `false` | Auto-detect and use planning |
| `planning_min_complexity` | `moderate` | Minimum complexity to trigger: simple/moderate/complex |

```bash
victor chat --planning "Analyze the codebase and refactor auth module"
victor chat --no-planning "quick question"
```

## Observability

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_observability_logging` | `false` | Write events to JSONL for dashboard |
| `observability_log_path` | `~/.victor/metrics/victor.jsonl` | Log file path |
| `log_level` | `INFO` | Logging level: DEBUG/INFO/WARN/ERROR |

```bash
victor chat --log-events "task"       # Enable JSONL logging
victor chat --log-level DEBUG         # Verbose logging
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_edge_model` | `true` | Enable edge model for micro-decisions |
| `use_semantic_tool_selection` | `true` | Embedding-based tool selection |
| `use_mcp_tools` | `false` | Enable MCP (Model Context Protocol) tools |
| `use_emojis` | `true` (not in CI) | Emoji indicators in output |
| `use_provider_pooling` | `false` | Provider connection pooling |

Feature flags are checked via `FeatureFlag.USE_EDGE_MODEL.is_enabled()`.

## Edge Model

| Setting | Default | Description |
|---------|---------|-------------|
| `EdgeModelConfig.enabled` | `true` | Enable edge model |
| `EdgeModelConfig.provider` | `ollama` | Edge model provider |
| `EdgeModelConfig.model` | `qwen3.5:2b` | Edge model name |
| `EdgeModelConfig.timeout_ms` | `4000` | Hard timeout per call |
| `EdgeModelConfig.max_tokens` | `50` | Max response tokens |
| `EdgeModelConfig.cache_ttl` | `120` | Cache TTL in seconds |

## MCP (Model Context Protocol)

| Setting | Default | Description |
|---------|---------|-------------|
| `use_mcp_tools` | `false` | Enable MCP support |
| `mcp_command` | `null` | MCP server command (e.g., `python mcp_server.py`) |
| `mcp_prefix` | `mcp` | Tool name prefix for MCP tools |

## Advanced Settings

These are internal settings that most users won't need to change.

| Module | Class | Key Settings |
|--------|-------|-------------|
| `automation_settings` | `AutomationSettings` | `one_shot_mode`, `max_follow_ups` |
| `checkpoint_settings` | `CheckpointSettings` | `enabled`, `auto_save_interval` |
| `compaction_settings` | `CompactionSettings` | `enabled`, `max_context_tokens` |
| `conversation_settings` | `ConversationSettings` | `max_history_tokens` |
| `context_settings` | `ContextSettings` | `max_context_files` |
| `event_settings` | `EventSettings` | `backend_type`, `buffer_size` |
| `exploration_settings` | `ExplorationSettings` | `max_parallel_searches` |
| `hitl_settings` | `HITLSettings` | `approval_timeout`, `auto_approve` |
| `resilience_settings` | `ResilienceSettings` | `circuit_breaker_threshold` |
| `sandbox_settings` | `SandboxSettings` | `enabled`, `timeout` |
| `search_settings` | `SearchSettings` | `max_results`, `semantic_threshold` |
| `security_settings` | `SecuritySettings` | `block_dangerous_commands` |

All settings modules live in `victor/config/*_settings.py`.

## Environment Variables

Victor reads these environment variables (prefix `VICTOR_` for most):

```bash
# API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Local servers
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URLS=http://localhost:1234

# Modes
AIRGAPPED_MODE=true        # Disable network tools
CI=true                    # Auto-disable emojis

# Logging
VICTOR_LOG_LEVEL=DEBUG
```

## Configuration Precedence

Settings are resolved in this order (highest priority first):

1. **CLI flags** (`--provider anthropic --model claude-sonnet`)
2. **Environment variables** (`VICTOR_DEFAULT_PROVIDER=anthropic`)
3. **Profile overrides** (`~/.victor/profiles.yaml` → selected profile)
4. **Settings defaults** (hardcoded in `victor/config/settings.py`)
