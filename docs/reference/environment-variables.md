# Environment Variables Reference

Complete reference for all environment variables recognized by Victor.

---

## API Keys

### LLM Provider API Keys

| Variable | Provider | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic | Claude models (claude-opus-4, claude-sonnet-4) |
| `OPENAI_API_KEY` | OpenAI | GPT models (gpt-4o, gpt-4-turbo) |
| `GOOGLE_API_KEY` | Google | Gemini models (gemini-2.0-flash, gemini-1.5-pro) |
| `XAI_API_KEY` | xAI | Grok models (grok-2, grok-beta) |
| `MOONSHOT_API_KEY` | Moonshot AI | Kimi K2 models |
| `DEEPSEEK_API_KEY` | DeepSeek | DeepSeek models (deepseek-chat, deepseek-reasoner) |
| `ZAI_API_KEY` | ZhipuAI | GLM models (glm-4.7, glm-4.6) |
| `GROQCLOUD_API_KEY` | Groq Cloud | Ultra-fast LPU inference |
| `GROQ_API_KEY` | Groq Cloud | Alternative for Groq |
| `CEREBRAS_API_KEY` | Cerebras | Fast inference |
| `MISTRAL_API_KEY` | Mistral AI | Mistral models |
| `TOGETHER_API_KEY` | Together AI | Together.ai platform |
| `OPENROUTER_API_KEY` | OpenRouter | Unified gateway |
| `FIREWORKS_API_KEY` | Fireworks AI | Fireworks platform |
| `HF_TOKEN` | Hugging Face | Hugging Face Inference API |
| `REPLICATE_API_TOKEN` | Replicate | Replicate platform |

### Enterprise Cloud Provider Keys

| Variable | Provider | Description |
|----------|----------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI | Path to service account JSON |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | Azure resource endpoint URL |
| `AWS_ACCESS_KEY_ID` | AWS Bedrock | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS Bedrock | AWS secret key |
| `AWS_SESSION_TOKEN` | AWS Bedrock | AWS session token (optional) |
| `AWS_REGION` | AWS Bedrock | AWS region |

### External Service API Keys

| Variable | Service | Description |
|----------|---------|-------------|
| `FINNHUB_API_KEY` | Finnhub | Stock data, analyst estimates |
| `FRED_API_KEY` | FRED | Federal Reserve Economic Data |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage | Stock/forex/crypto data |
| `POLYGON_API_KEY` | Polygon | Real-time market data |
| `TIINGO_API_KEY` | Tiingo | Stock/crypto/forex data |
| `IEX_API_KEY` | IEX Cloud | IEX Cloud market data |
| `QUANDL_API_KEY` | Quandl | Nasdaq Data Link (legacy) |
| `NASDAQ_API_KEY` | Nasdaq | Nasdaq Data Link |
| `NEWSAPI_API_KEY` | NewsAPI | News aggregation |
| `MARKETAUX_API_KEY` | Marketaux | Financial news |
| `SEC_API_KEY` | SEC | SEC EDGAR API (optional) |
| `OPENWEATHER_API_KEY` | OpenWeather | Weather data |
| `GEOCODING_API_KEY` | Geocoding | Geocoding services |

---

## Configuration Overrides

### Victor Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_SKIP_ENV_FILE` | `unset` | Set to `1` to skip loading `.env` file |
| `VICTOR_DIR_NAME` | `.victor` | Directory name for Victor config |
| `VICTOR_CONTEXT_FILE` | `init.md` | Project context filename |
| `VICTOR_EXTRA_READ_ROOTS` | `unset` | Extra directories `read()` may access, `:`-separated. Opt-in only |
| `VICTOR_CONVERSATION_DB` | `unset` | Pin the conversation store to an explicit path instead of deriving it from the cwd |
| `VICTOR_RUN_KIND` | `interactive` | Seed the run kind stamped on every usage event: `interactive`, `evaluation`, `delegate`, `headless` |
| `VICTOR_DISABLE_WORKSPACE_GUARD` | `unset` | Set to `1` to drop the `read()` workspace scope entirely (testing) |

#### Working across two repositories

`read()` is scoped to the current project plus its linked git worktrees.
`ls()` and `shell()` are **not** scoped — the guard narrows one tool, it is not
a sandbox. When a session legitimately spans two codebases — co-designing a wire
contract, or letting Victor inspect its own `~/.victor` state — declare the
second root:

```bash
VICTOR_EXTRA_READ_ROOTS=~/code/sandhi:~/.victor victor chat
```

Without it, an out-of-project path is still reachable via
`ls(path='/abs/dir')` or `shell(cmd='sed -n "1,200p" /abs/file', action='read')`.
Prefer `VICTOR_EXTRA_READ_ROOTS` when the whole session needs the second repo,
since it keeps `read()`'s pagination and in-file search working.

#### Conversations from a process that changes directory

`ConversationStore` defaults to `{cwd}/.victor/project.db`. Any process that
`chdir`s therefore writes its conversations somewhere unexpected — benchmark
evaluation runs each task in a temporary workspace, so their sessions were lost
with it. Set `VICTOR_CONVERSATION_DB` to pin the store:

```bash
VICTOR_CONVERSATION_DB=~/.victor/evaluations/sessions.db victor benchmark ...
```

Precedence is explicit `db_path` argument → this variable → project default.
`victor benchmark` sets it automatically for the duration of a run (and restores
whatever was there before), so this is only needed for custom embeddings.

#### Telling apart the kind of run that produced a trace

Every usage event carries a `run_kind` field beside `session_id`. Benchmark runs
tag themselves `evaluation` and delegate workers tag themselves `delegate`, so
consumers no longer have to guess from prompt text — the turn-budget notice
("WARNING: N turns remaining") is emitted by the shared agentic loop and appears
in both, which previously caused delegate work to be counted as benchmark runs.

`VICTOR_RUN_KIND` seeds the default for a whole process, which is what a
subprocess-based runner wants:

```bash
VICTOR_RUN_KIND=evaluation my-eval-runner.sh
```

An in-process scope always wins over the variable, and nesting is honoured — a
delegate spawned inside an evaluation is tagged `delegate` for its duration and
the evaluation tag resumes afterwards.

### Mode Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_HEADLESS_MODE` | `false` | Run without prompts |
| `VICTOR_DRY_RUN_MODE` | `false` | Preview changes without applying |
| `VICTOR_MAX_FILE_CHANGES` | `unset` | Limit file modifications per session |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_LOG_LEVEL` | `INFO` | Console log level (DEBUG, INFO, WARNING, ERROR) |
| `VICTOR_LOG_FILE_LEVEL` | `DEBUG` | File log level |
| `VICTOR_LOG_FILE` | `unset` | Log file path |
| `VICTOR_LOG_DISABLED` | `false` | Set to `true` to disable file logging |

### Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_TIMEOUT_HTTP_DEFAULT` | `30.0` | HTTP request timeout (seconds) |
| `VICTOR_TIMEOUT_BASH_DEFAULT` | `120` | Bash command timeout (seconds) |

---

## Local Provider URLs

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LMSTUDIO_BASE_URLS` | `http://127.0.0.1:1234` | LMStudio URLs (comma-separated) |
| `VLLM_BASE_URL` | `http://localhost:8000` | vLLM server URL |

### Provider Loading Controls

Victor lazily materializes provider implementations at first use (`ProviderRegistry.get/create`).
This keeps startup/import overhead lower, especially when optional local runtimes are not installed.

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_ENABLE_MLX_PROVIDER` | `unset` | Set to `0`/`false` to disable MLX provider aliases (`mlx`, `mlx-lm`, `applesilicon`) entirely |
| `VICTOR_MLX_SKIP_PREFLIGHT` | `unset` | Set to `1`/`true` to bypass MLX subprocess preflight (only for known-good MPS/Metal environments) |

**Example for LAN LMStudio server:**

```bash
export LMSTUDIO_BASE_URLS="http://192.168.1.20:1234,http://127.0.0.1:1234"
```

---

## MCP Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_MCP_CONFIG` | `unset` | Path to MCP configuration file |

---

## Code Execution Sandbox

| Variable | Default | Description |
|----------|---------|-------------|
| `VICTOR_CODE_EXECUTOR_MEM` | `512m` | Memory limit for code execution |
| `VICTOR_CODE_EXECUTOR_CPU_SHARES` | `256` | CPU shares for code execution |

---

## Workflow/Sandbox Internal Variables

These are set internally during workflow execution:

| Variable | Description |
|----------|-------------|
| `VICTOR_SANDBOXED` | Set to `1` when running in MCP sandbox |
| `VICTOR_NETWORK_DISABLED` | Set to `1` when network is disabled in sandbox |

---

## HITL (Human-in-the-Loop) Transports

### Email (SMTP)

| Variable | Default | Description |
|----------|---------|-------------|
| `SMTP_HOST` | `localhost` | SMTP server host |
| `SMTP_PORT` | `587` | SMTP server port |
| `SMTP_USER` | `unset` | SMTP username |
| `SMTP_PASSWORD` | `unset` | SMTP password |
| `SMTP_FROM` | `approvals@example.com` | From address |
| `APPROVAL_EMAILS` | `unset` | To addresses (comma-separated) |

### SMS (Twilio)

| Variable | Default | Description |
|----------|---------|-------------|
| `SMS_PROVIDER` | `twilio` | SMS provider |
| `TWILIO_ACCOUNT_SID` | `unset` | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | `unset` | Twilio auth token |
| `TWILIO_FROM_NUMBER` | `unset` | From phone number |
| `APPROVAL_PHONES` | `unset` | To phone numbers (comma-separated) |

### Slack

| Variable | Default | Description |
|----------|---------|-------------|
| `SLACK_WEBHOOK_URL` | `unset` | Slack webhook URL |
| `SLACK_BOT_TOKEN` | `unset` | Slack bot token |
| `SLACK_APPROVAL_CHANNEL` | `#approvals` | Approval channel |

### Microsoft Teams

| Variable | Default | Description |
|----------|---------|-------------|
| `TEAMS_WEBHOOK_URL` | `unset` | Teams webhook URL |
| `AZURE_TENANT_ID` | `unset` | Azure tenant ID |
| `AZURE_CLIENT_ID` | `unset` | Azure client ID |
| `AZURE_CLIENT_SECRET` | `unset` | Azure client secret |

### GitHub

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | `unset` | GitHub personal access token |
| `GITHUB_APP_ID` | `unset` | GitHub App ID |
| `GITHUB_OWNER` | `unset` | Repository owner |
| `GITHUB_REPO` | `unset` | Repository name |
| `GITHUB_API_URL` | `https://api.github.com` | GitHub API URL |

### GitLab

| Variable | Default | Description |
|----------|---------|-------------|
| `GITLAB_TOKEN` | `unset` | GitLab personal access token |
| `GITLAB_PROJECT_ID` | `unset` | GitLab project ID |
| `GITLAB_URL` | `https://gitlab.com` | GitLab base URL |

### Jira

| Variable | Default | Description |
|----------|---------|-------------|
| `JIRA_URL` | `unset` | Jira instance URL |
| `JIRA_EMAIL` | `unset` | Jira user email |
| `JIRA_API_TOKEN` | `unset` | Jira API token |
| `JIRA_PROJECT_KEY` | `unset` | Jira project key |

### PagerDuty

| Variable | Default | Description |
|----------|---------|-------------|
| `PAGERDUTY_API_KEY` | `unset` | PagerDuty API key |
| `PAGERDUTY_ROUTING_KEY` | `unset` | Events routing key |
| `PAGERDUTY_SERVICE_ID` | `unset` | Service ID |

### Terraform Cloud

| Variable | Default | Description |
|----------|---------|-------------|
| `TF_TOKEN` | `unset` | Terraform Cloud token |
| `TFC_ORGANIZATION` | `unset` | TFC organization |
| `TFC_WORKSPACE` | `unset` | TFC workspace |

---

## OpenTelemetry

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `unset` | OTLP exporter endpoint |
| `VICTOR_ENV` | `development` | Deployment environment tag |

---

## Tool-Specific Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORE_READONLY_TOOLS` | `unset` | Override curated read-only tool set |

---

## Quick Reference: Essential Variables

For a minimal setup, you typically need:

```bash
# Cloud provider (choose one)
export ANTHROPIC_API_KEY="sk-..."
# OR
export OPENAI_API_KEY="sk-..."
# OR
export GOOGLE_API_KEY="..."

# For local providers, ensure the server is running:
# Ollama: ollama serve
# LMStudio: Start the app and enable local server
# vLLM: python -m vllm.entrypoints.openai.api_server --model MODEL
```

---

## Setting Variables

### Temporary (Current Session)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
victor chat
```

### Permanent (Shell Profile)

Add to `~/.bashrc`, `~/.zshrc`, or equivalent:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Using .env File

Create `.env` in your project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
VICTOR_LOG_LEVEL=DEBUG
```

### Using Victor's Secure Key Storage

```bash
# Store in system keyring (recommended)
victor keys --set anthropic --keyring

# Store in ~/.victor/api_keys.yaml
victor keys --set anthropic
```

---

## Security Best Practices

1. **Use system keyring** for API keys when possible
2. **Never commit** `.env` files or `api_keys.yaml` to version control
3. **Add to .gitignore**:
   ```
   .env
   .victor/api_keys.yaml
   ```
4. **Use environment variables** in CI/CD pipelines
5. **Set restrictive permissions** on key files: `chmod 600 ~/.victor/api_keys.yaml`
