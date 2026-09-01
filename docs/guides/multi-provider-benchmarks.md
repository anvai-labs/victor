# Multi-Provider Benchmarks: One Framework, Any LLM

Victor runs identically across cloud and local LLM providers. This guide shows real benchmark results and how to reproduce them.

## Why It Matters

Most AI frameworks lock you into a single provider. Victor's provider abstraction means you can:

- **Switch providers in one line** — no code changes, just a different `--provider` flag
- **Compare cost vs quality** — run the same tasks across providers to find your sweet spot
- **Stay resilient** — automatic retry with exponential backoff on rate limits, circuit breaker protection

## Benchmark Results (March 2026)

We ran 5 tasks spanning code generation, research, security analysis, file operations, and workflow orchestration across 3 providers using their cheapest capable models:

| Provider | Model | Success | Avg Time | Avg Output | Input Cost |
|----------|-------|---------|----------|------------|-----------|
| **Anthropic** | Claude Haiku 4.5 | 5/5 (100%) | 16.9s | 9,122 chars | $0.80/1M tokens |
| **OpenAI** | GPT-4o-mini | 5/5 (100%) | 14.7s | 3,837 chars | $0.15/1M tokens |
| **DeepSeek** | DeepSeek-Chat V3 | 5/5 (100%) | 35.9s | 8,356 chars | $0.07/1M tokens |

### What We Tested

| Task | What It Does | Tests |
|------|-------------|-------|
| **Code Generation** | Generate a `DataProcessor` class with CSV I/O, type hints, error handling | Can the model write production-quality Python? |
| **Research Synthesis** | Synthesize microservices architecture research with examples | Can it reason across multiple concepts? |
| **File Operations** | Create directories, parse files, organize by type | Can it plan multi-step file operations? |
| **Security Audit** | Find SQL injection, XSS, auth flaws with OWASP classification | Can it do deep technical analysis? |
| **Workflow Design** | Design a sequential data pipeline: load, clean, transform, validate | Can it architect a multi-stage system? |

### Quality Highlights

**Anthropic Claude Haiku 4.5** — Richest output. The security audit included OWASP A03/CWE-89 classifications, exploit scenarios, and fixed code examples. Best for tasks requiring depth.

**DeepSeek Chat V3** — Matches Haiku's depth at 1/10th the cost. Excellent code generation with logging, type hints, and error recovery. Best for budget-conscious teams.

**OpenAI GPT-4o-mini** — Fastest responses, concise output. Good for high-volume tasks where speed matters more than exhaustive detail.

## How to Run Your Own Benchmarks

### 1. Install Victor

```bash
pip install victor-ai
```

### 2. Set Up API Keys

Victor uses your system keychain for secure key storage:

```bash
victor keys set anthropic    # Enter your Anthropic API key
victor keys set openai       # Enter your OpenAI API key
victor keys set deepseek     # Enter your DeepSeek API key
```

### 3. Run Benchmarks

```bash
# Anthropic (best quality)
python docs/benchmarking/run_benchmark.py \
  --provider anthropic --model claude-haiku-4-5-20251001 \
  --verbose --output

# OpenAI (fastest)
python docs/benchmarking/run_benchmark.py \
  --provider openai --model gpt-4o-mini \
  --verbose --output

# DeepSeek (cheapest)
python docs/benchmarking/run_benchmark.py \
  --provider deepseek --model deepseek-chat \
  --verbose --output

# Run specific tasks only
python docs/benchmarking/run_benchmark.py \
  --provider anthropic --task C1 --task A1 \
  --verbose
```

### 4. Compare Results

Results are saved to `docs/benchmarking/results/victor/` as JSON files. Use the analysis script:

```bash
python docs/benchmarking/analyze_results.py
```

## Built-In Resilience

All providers automatically get:

- **Rate-limit retry** — exponential backoff with jitter on 429 errors
- **Retry-After respect** — honors server-specified wait times
- **Circuit breaker** — stops hammering a failing provider
- **Configurable policy** — `max_retries=0` to raise immediately, `max_retries=3` (default) for automatic retry

This is handled at the `BaseProvider` level — no per-provider configuration needed.

## Adding Your Own Provider

Victor supports 25 providers out of the box. To benchmark with a different one:

```bash
# Any provider Victor supports works
python docs/benchmarking/run_benchmark.py \
  --provider ollama --model llama3.2 \
  --verbose

# Or use Groq for fast inference
python docs/benchmarking/run_benchmark.py \
  --provider groq --model llama-3.1-70b-versatile \
  --verbose
```

## Full Results

See the [detailed benchmark report](../benchmarking/results/BENCHMARK_RESULTS_2026-03-16.md) for per-task breakdowns and output quality analysis.
