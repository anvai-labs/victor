# Provider Guide

Complete guide to using LLM providers in Victor. Victor supports 24 providers, from local inference to cloud APIs, with seamless mid-conversation switching.

## What is a Provider?

A **provider** in Victor is an abstraction layer that connects to an LLM service. All providers implement a common interface (`BaseProvider`) with:

- **`chat()`**: Send messages and receive responses
- **`stream()`**: Stream responses in real-time
- **`supports_tools()`**: Check if tool/function calling is available
- **`supports_streaming()`**: Check if streaming is supported

This abstraction enables Victor's signature feature: **switching providers mid-conversation without losing context**.

## Provider Categories

| Category | Providers | Best For |
|----------|-----------|----------|
| **Local** | Ollama, LM Studio, vLLM, llama.cpp | Privacy, no API costs, offline use |
| **Cloud** | Anthropic, OpenAI, Google, xAI, DeepSeek, Mistral, Groq, Cerebras, Together, Fireworks, OpenRouter, Moonshot | Best quality, fastest inference, latest models |
| **Enterprise** | Azure OpenAI, AWS Bedrock, Vertex AI | Compliance, security, enterprise integrations |
| **Platforms** | Hugging Face, Replicate | Model variety, research models |

## Provider Decision Matrix

A quick decision view of the most-used providers. Capability columns reflect each
adapter's own feature detection (`supports_tools()` / `supports_streaming()` /
`supports_prompt_caching()` / `cache_cost_model()` on `BaseProvider` in
`victor/providers/`), not marketing claims.

| Provider | Local/Cloud | Tool calling | Streaming | Prompt caching | Cost tier |
|----------|-------------|--------------|-----------|----------------|-----------|
| **Anthropic** | Cloud | Native | Yes | Yes — 90% read discount, 1.25x write, 5m TTL | $$$ |
| **OpenAI** | Cloud | Native | Yes | Yes — automatic prefix caching | $$$ |
| **Google Gemini** | Cloud | Native | Yes | Yes — context caching (32K-token minimum) | $$ (free experimental tier) |
| **Azure OpenAI** | Cloud | Native | Yes | Yes — prompt caching | $$$ (enterprise) |
| **AWS Bedrock** | Cloud | Native | Yes | Yes — 90% read discount (Claude models) | $$$ (enterprise) |
| **DeepSeek** | Cloud | Native (`deepseek-chat` only) | Yes | Per provider spec¹ | $ |
| **Groq** | Cloud | Native | Yes | Per provider spec¹ | $ (free tier) |
| **Mistral** | Cloud | Native | Yes | Per provider spec¹ | $$ (free tier) |
| **Ollama** | Local | Model-dependent (auto-detected) | Yes | Local KV-prefix reuse (nothing billed) | Free (your hardware) |
| **LM Studio** | Local | Native (llama.cpp) | Yes | Local KV-prefix reuse (nothing billed) | Free (your hardware) |
| **vLLM** | Local | Native (`--enable-auto-tool-choice`) | Yes | Local KV-prefix reuse (nothing billed) | Free (your hardware) |

¹ OpenAI-compatible adapters (DeepSeek, Groq, Mistral, xAI, Together, …) read their
capabilities from a validated per-provider spec rather than hard-coding them
(`victor/providers/sandhi_openai_compat_policy.py`).

Cost tiers are indicative ($ = budget, $$$ = premium); check current provider pricing.
This is a subset — Victor ships **24 provider adapters**. For the full list run
`victor providers`, or see the [Provider Reference](../reference/providers/) and the
detailed per-provider sections below.

---

## Quick Start

### Using a Local Provider (Ollama)

```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:7b

# Victor automatically detects Ollama
victor chat "Hello, world!"
```

### Using a Cloud Provider (Anthropic)

```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Use Claude
victor chat --provider anthropic --model claude-sonnet-4-20250514 "Hello!"
```

### Switching Providers Mid-Conversation

```bash
# Start with Claude for planning
victor chat --provider anthropic "Design a REST API for a blog"

# Switch to GPT-4 for implementation (context preserved)
/provider openai --model gpt-4o

# Finish with local model for privacy
/provider ollama --model qwen2.5-coder:7b
```

---

## Local Providers

**No API key required.** Run models locally with full privacy and control.

### Ollama (Recommended for Beginners)

The easiest way to run models locally. Supports 100+ models with automatic model management.

**Installation**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Start Victor
victor chat "Hello!"
```

**Popular Models**:
| Model | Size | Use Case |
|-------|------|----------|
| `qwen2.5-coder:7b` | 4.5GB | Best for coding on most laptops |
| `qwen3:32b` | 19GB | Higher quality, needs 24GB+ RAM |
| `llama3.2:3b` | 2GB | Fast, lightweight |
| `mistral:7b` | 4.1GB | General purpose |
| `deepseek-coder:6.7b` | 4GB | Code-focused |

**Configuration**:
```bash
# Set custom Ollama host (if running on another machine)
export OLLAMA_ENDPOINTS="http://192.168.1.100:11434,http://localhost:11434"

# Victor will try each endpoint in order
victor chat --provider ollama --model qwen2.5-coder:7b
```

**Tool Calling Support**:
- Most Ollama models support tool calling via the native API
- Models with `-tools` suffix (e.g., `qwen3-coder-tools`) have enhanced tool support
- Victor automatically detects tool support and falls back to text parsing if needed

**Profile Example**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
```

---

### LM Studio

GUI-based model management with OpenAI-compatible API. Great for Windows users.

**Installation**:
1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install and launch LM Studio
3. Download a model (e.g., Qwen 2.5 Coder)
4. Start the local server (Developer tab)

**Usage**:
```bash
# Set LM Studio host
export VICTOR_LM_STUDIO_HOST=127.0.0.1:1234

# Use with Victor
victor chat --provider lmstudio --model qwen2.5-coder-7b
```

**Features**:
- Native tool calling support (llama.cpp 0.3.6+ enables tools for all models)
- Thinking tag extraction for Qwen3/DeepSeek-R1 models
- Automatic endpoint discovery with health probing
- 300s timeout for local model inference

**Profile Example**:
```yaml
profiles:
  lmstudio:
    provider: lmstudio
    base_url: http://127.0.0.1:1234/v1
    model: qwen2.5-coder-7b
```

---

### vLLM

High-throughput inference server with PagedAttention. Best for production deployments.

**Installation**:
```bash
# Install vLLM
pip install vllm

# Start server with tool calling enabled
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**Usage**:
```bash
export VICTOR_VLLM_HOST=127.0.0.1:8000
victor chat --provider vllm --model Qwen/Qwen2.5-Coder-7B-Instruct
```

**Recommended Models** (with VRAM requirements):
| Model | FP16 | Q8 |
|-------|------|-----|
| Qwen/Qwen2.5-Coder-7B-Instruct | 14GB | 7GB |
| Qwen/Qwen2.5-Coder-14B-Instruct | 28GB | 14GB |
| deepseek-ai/DeepSeek-Coder-V2-Lite | 32GB | 16GB |
| mistralai/Codestral-22B-v0.1 | 44GB | 22GB |

**Advantages**:
- Up to 20x faster than Ollama
- Automatic request batching
- Production-ready
- OpenAI-compatible API

---

### llama.cpp

Lightweight C++ inference. Runs on CPU with optional GPU acceleration.

**Installation**:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Start server
./server --model /path/to/model.gguf --port 8080
```

**Usage**:
```bash
export VICTOR_LLAMACPP_HOST=127.0.0.1:8080
victor chat --provider llama-cpp
```

**Advantages**:
- Smallest footprint (<5MB binary)
- Runs on CPU
- Broad model support (any GGUF)
- Low memory consumption

---

## Cloud Providers

**API key required.** Access to the most powerful models with fast inference.

### Anthropic Claude

Best overall for complex reasoning, coding, and long context.

**Models**:
| Model | Context | Best For |
|-------|---------|----------|
| `claude-opus-4-5-20251101` | 200K | Most capable, complex tasks |
| `claude-sonnet-4-20250514` | 200K | Excellent balance of quality and speed |
| `claude-3-5-sonnet-20241022` | 200K | Fast, efficient for everyday tasks |
| `claude-3-5-haiku-20241022` | 200K | Fastest, most affordable |

**Setup**:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
victor chat --provider anthropic --model claude-sonnet-4-20250514
```

**Features**:
- Excellent tool calling (50+ tools supported)
- Parallel tool calls
- Streaming tool calls
- Vision support (images, charts, diagrams)
- 200K token context window

**Profile Example**:
```yaml
profiles:
  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    max_tokens: 8192
```

---

### OpenAI GPT

Industry standard with excellent function calling and reasoning models.

**Models**:
| Model | Context | Best For |
|-------|---------|----------|
| `gpt-4o` | 128K | Multimodal, fast |
| `gpt-4o-mini` | 128K | Cost-effective |
| `o1-preview` | 128K | Advanced reasoning |
| `o1-mini` | 128K | Fast reasoning |

**Setup**:
```bash
export OPENAI_API_KEY=sk-proj-...
victor chat --provider openai --model gpt-4o
```

**Features**:
- Native function calling
- Parallel tool calls
- Vision support
- O-series reasoning models (no tool support for o1/o3)

**Note**: O-series models (o1, o3) use `max_completion_tokens` instead of `max_tokens` and do not support temperature or tools.

---

### Google Gemini

Largest context windows (up to 2M tokens) and free experimental models.

**Models**:
| Model | Context | Best For |
|-------|---------|----------|
| `gemini-2.0-flash-exp` | 1M | Free, fast |
| `gemini-2.0-flash-thinking-exp` | 1M | Reasoning (free) |
| `gemini-1.5-pro` | 2M | Largest context |
| `gemini-1.5-flash` | 2M | Fast, large context |

**Setup**:
```bash
export GOOGLE_API_KEY=...
victor chat --provider google --model gemini-2.0-flash-exp
```

**Features**:
- Massive context window (up to 2M tokens)
- Free experimental models
- Function calling
- Multimodal (images, video, audio)

**Safety Settings**:
```python
# For code generation without safety blocks
provider = GoogleProvider(api_key=key, safety_level="block_none")
```

---

### DeepSeek

Very affordable with strong code generation. Chinese language support.

**Models**:
| Model | Context | Tool Support |
|-------|---------|--------------|
| `deepseek-chat` | 128K | Yes |
| `deepseek-reasoner` | 128K | No (thinking mode) |

**Setup**:
```bash
export DEEPSEEK_API_KEY=...
victor chat --provider deepseek --model deepseek-chat
```

**Features**:
- 10-30x cheaper than OpenAI
- Native tool calling (deepseek-chat only)
- Reasoning traces (deepseek-reasoner)
- 128K context window

**Note**: `deepseek-reasoner` does NOT support function calling; use for reasoning tasks only.

---

### Groq

Ultra-fast inference using custom LPU hardware. Free tier available.

**Models**:
| Model | Context | Speed |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | 128K | 100+ tok/s |
| `llama-3.1-8b-instant` | 128K | 300+ tok/s |
| `qwen/qwen3-32b` | 128K | 150+ tok/s |

**Setup**:
```bash
export GROQ_API_KEY=...
victor chat --provider groqcloud --model llama-3.3-70b-versatile
```

**Features**:
- Fastest inference (300+ tokens/sec)
- Free developer tier
- Native tool calling
- 128K+ context windows

**Note**: Groq has strict ~4MB payload limits due to LPU architecture. Victor automatically truncates payloads if needed.

---

### Mistral

European provider with generous free tier and excellent tool calling.

**Models**:
| Model | Context | Best For |
|-------|---------|----------|
| `mistral-large-latest` | 128K | Most capable |
| `mistral-small-latest` | 32K | Balanced |
| `codestral-latest` | 32K | Code generation |

**Setup**:
```bash
export MISTRAL_API_KEY=...
victor chat --provider mistral --model mistral-large-latest
```

**Free Tier**:
- 500,000 tokens/minute
- 1B tokens/month per model
- Requires phone verification

---

### xAI Grok

Real-time knowledge with Grok models.

**Setup**:
```bash
export XAI_API_KEY=...
victor chat --provider xai --model grok-beta
```

**Features**:
- Real-time knowledge
- Function calling
- 128K context

---

### Cerebras

Ultra-fast inference using Wafer Scale Engine hardware.

**Setup**:
```bash
export CEREBRAS_API_KEY=...
victor chat --provider cerebras
```

**Features**:
- 500+ tokens/sec inference
- Free tier (30 req/min)
- Llama, Mistral, Qwen models

---

### Together AI

Wide model selection with fast inference.

**Setup**:
```bash
export TOGETHER_API_KEY=...
victor chat --provider together --model meta-llama/Llama-3.2-3B-Instruct-Turbo
```

**Features**:
- 100+ open models
- Up to 150 tokens/sec
- Fine-tuned models available

---

### Fireworks AI

Fast inference with function calling support.

**Setup**:
```bash
export FIREWORKS_API_KEY=...
victor chat --provider fireworks
```

---

### OpenRouter

Unified gateway to multiple providers with price comparison.

**Setup**:
```bash
export OPENROUTER_API_KEY=...
victor chat --provider openrouter --model anthropic/claude-sonnet-4
```

**Features**:
- Single API for all providers
- Price comparison
- Fallback routing

---

### Moonshot (Kimi)

Trillion-parameter Kimi K2 model with 256K context.

**Setup**:
```bash
export MOONSHOT_API_KEY=...
victor chat --provider moonshot --model kimi-k2-thinking
```

**Features**:
- 256K context window
- Thinking/reasoning mode
- Native tool calling

---

## Enterprise Providers

For enterprise deployments with compliance and security requirements.

### Azure OpenAI

Enterprise-grade OpenAI access with Azure security.

**Setup**:
```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

victor chat --provider azure --model gpt-4o
```

**Features**:
- Azure AD integration
- RBAC, Private Link, VNET
- Regional data residency
- Content filtering
- Phi models (Microsoft SLMs)

**Profile Example**:
```yaml
profiles:
  azure:
    provider: azure
    api_key_env: AZURE_OPENAI_API_KEY
    api_base_env: AZURE_OPENAI_ENDPOINT
    api_version: "2024-08-01-preview"
    deployment: gpt-4o
```

---

### AWS Bedrock

Access Claude, Llama, Mistral, and Titan models on AWS infrastructure.

**Setup**:
```bash
# Configure AWS credentials
aws configure

victor chat --provider bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0
```

**Available Models**:
- Anthropic Claude 3.5/3 (Sonnet, Haiku, Opus)
- Meta Llama 3.2/3.1 (up to 405B)
- Mistral Large, Small, Mixtral
- Amazon Titan
- Cohere Command R/R+

**Features**:
- AWS IAM authentication
- VPC endpoints
- SOC2, HIPAA compliance
- Provisioned throughput

---

### Google Vertex AI

Enterprise Gemini access on Google Cloud.

**Setup**:
```bash
gcloud auth application-default login

victor chat --provider vertex --model gemini-2.0-flash-exp
```

**Features**:
- Google Cloud authentication
- VPC Service Controls
- Enterprise support

---

## Open Model Platforms

### Hugging Face

Access to 100,000+ models on Hugging Face Hub.

**Setup**:
```bash
export HF_API_KEY=...  # Optional for public models
victor chat --provider huggingface --model Qwen/Qwen2.5-Coder-7B-Instruct
```

**Features**:
- Largest model collection
- Free serverless inference
- Research and community models

---

### Replicate

Easy model hosting with pay-per-use pricing.

**Setup**:
```bash
export REPLICATE_API_TOKEN=...
victor chat --provider replicate --model meta/llama-3.2-3b-instruct
```

---

## Provider Features

### Provider Switching

Switch models mid-conversation without losing context:

```bash
# Start with Claude
victor chat --provider anthropic "Design a REST API"

# Mid-conversation: switch to GPT-4
/provider openai --model gpt-4o

# Continue: switch to local model
/provider ollama --model qwen2.5-coder:7b
```

**Benefits**:
- Leverage different model strengths
- Cost optimization (cheaper models for simple tasks)
- Privacy (local models for sensitive data)
- Redundancy (fallback if provider is down)

---

### Fallback Configuration

Configure automatic provider fallback:

```yaml
# ~/.victor/profiles.yaml
profiles:
  resilient:
    provider: anthropic
    model: claude-sonnet-4-20250514
    fallback:
      - provider: openai
        model: gpt-4o
      - provider: ollama
        model: qwen2.5-coder:7b
```

---

### Cost Tracking

Victor tracks token usage across providers:

```bash
# View session costs
/usage

# In profile
profiles:
  cost-aware:
    provider: anthropic
    track_costs: true
```

---

### Rate Limiting

Victor implements automatic rate limiting and circuit breaker patterns:

- **Circuit Breaker**: Protects against cascading failures
- **Automatic Retries**: Configurable retry with exponential backoff
- **Rate Limit Handling**: Automatic backoff on 429 errors

---

## Provider Comparison Table

### Tool Calling Support

| Provider | Native Tools | Parallel Tools | Streaming Tools |
|----------|--------------|----------------|-----------------|
| **Anthropic** | Yes | Yes | Yes |
| **OpenAI** | Yes | Yes | Yes |
| **Google** | Yes | Yes | Yes |
| **xAI** | Yes | Yes | Yes |
| **DeepSeek** | Yes (chat only) | Yes | Yes |
| **Mistral** | Yes | Yes | Yes |
| **Groq** | Yes | Yes | Yes |
| **Cerebras** | Yes | Yes | Yes |
| **Together** | Yes | Yes | Yes |
| **Fireworks** | Yes | Yes | Yes |
| **OpenRouter** | Yes | Yes | Yes |
| **Moonshot** | Yes | Yes | Yes |
| **Azure** | Yes | Yes | Yes |
| **Bedrock** | Yes | Yes | Yes |
| **Vertex** | Yes | Yes | Yes |
| **Ollama** | Model-dependent | Model-dependent | No |
| **LM Studio** | Yes (llama.cpp) | Yes | No |
| **vLLM** | Yes | Yes | Yes |

### Context Window Sizes

| Provider | Model | Context Window |
|----------|-------|----------------|
| **Google** | gemini-1.5-pro | 2,000,000 |
| **Google** | gemini-2.0-flash | 1,000,000 |
| **Moonshot** | kimi-k2 | 256,000 |
| **Anthropic** | claude-* | 200,000 |
| **DeepSeek** | deepseek-* | 128,000 |
| **OpenAI** | gpt-4o | 128,000 |
| **Groq** | llama-3.3-70b | 128,000 |
| **Mistral** | mistral-large | 128,000 |
| **Azure** | gpt-4o | 128,000 |
| **Bedrock** | claude-3-5-sonnet | 200,000 |

### Vision Support

| Provider | Vision Models |
|----------|---------------|
| **Anthropic** | All Claude 3/3.5/4 models |
| **OpenAI** | gpt-4o, gpt-4-vision |
| **Google** | All Gemini models |
| **Azure** | gpt-4o, gpt-4-vision |
| **Bedrock** | Claude 3, Llama 3.2 multimodal |
| **Ollama** | LLaVA variants only |

---

## Environment Variables Reference

| Provider | Environment Variable | Required |
|----------|---------------------|----------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Yes |
| **OpenAI** | `OPENAI_API_KEY` | Yes |
| **Google** | `GOOGLE_API_KEY` | Yes |
| **xAI** | `XAI_API_KEY` | Yes |
| **DeepSeek** | `DEEPSEEK_API_KEY` | Yes |
| **Mistral** | `MISTRAL_API_KEY` | Yes |
| **Groq** | `GROQ_API_KEY` or `GROQCLOUD_API_KEY` | Yes |
| **Cerebras** | `CEREBRAS_API_KEY` | Yes |
| **Together** | `TOGETHER_API_KEY` | Yes |
| **Fireworks** | `FIREWORKS_API_KEY` | Yes |
| **OpenRouter** | `OPENROUTER_API_KEY` | Yes |
| **Moonshot** | `MOONSHOT_API_KEY` | Yes |
| **Azure** | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | Yes |
| **AWS Bedrock** | AWS CLI credentials | Yes |
| **Vertex AI** | gcloud authentication | Yes |
| **Hugging Face** | `HF_API_KEY` | Optional |
| **Replicate** | `REPLICATE_API_TOKEN` | Yes |
| **Ollama** | `OLLAMA_HOST` or `OLLAMA_ENDPOINTS` | Optional |
| **LM Studio** | `LMSTUDIO_ENDPOINTS` | Optional |
| **vLLM** | `VICTOR_VLLM_HOST` | Optional |
| **llama.cpp** | `VICTOR_LLAMACPP_HOST` | Optional |

---

## Choosing a Provider

### Decision Guide

**For Local Development**:
1. **Beginner**: Ollama (easiest setup)
2. **Windows**: LM Studio (GUI-based)
3. **Production**: vLLM (fastest, highest throughput)
4. **CPU-only**: llama.cpp (smallest footprint)

**For Cloud Development**:
1. **Best Overall**: Anthropic Claude Sonnet
2. **Code Generation**: OpenAI GPT-4o, DeepSeek Coder
3. **Cost-Effective**: Google Gemini 2.0 Flash (free), DeepSeek
4. **Fastest**: Groq, Cerebras (300+ tokens/s)
5. **Large Context**: Google Gemini 1.5 Pro (2M tokens)

**For Enterprise**:
1. **Azure Ecosystem**: Azure OpenAI
2. **AWS Ecosystem**: AWS Bedrock
3. **GCP Ecosystem**: Vertex AI

**For Experimentation**:
1. **Model Variety**: Hugging Face (100K+ models)
2. **Price Comparison**: OpenRouter (unified API)
3. **Latest Models**: Together AI

---

## Troubleshooting

### Common Issues

**API Key Not Found**:
```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Set in current shell
export ANTHROPIC_API_KEY=sk-ant-...

# Or use keyring
victor keys --set anthropic --keyring
```

**Provider Not Available**:
```bash
# Check installed providers
victor providers

# Reinstall with provider extras
pip install "victor-ai[anthropic,openai]"
```

**Model Not Found**:
```bash
# List available models
victor providers --provider anthropic

# Check model name
victor chat --provider anthropic --model claude-sonnet-4-20250514
```

**Connection Timeout**:
```bash
# Check network
curl https://api.anthropic.com/v1/messages

# Increase timeout
victor chat --timeout 120
```

**Rate Limiting**:
```bash
# Switch provider mid-conversation
/provider openai

# Or use local model
/provider ollama
```

**Ollama Not Reachable**:
```bash
# Check Ollama status
ollama list

# Restart Ollama
ollama serve
```

---

## Additional Resources

- **Provider Reference**: [Full Reference Documentation](../reference/providers/)
- **Provider Comparison**: [Comparison Table](../reference/providers/comparison.md)
- **Setup Guide**: [Detailed Setup](../reference/providers/setup.md)
- **Tool Catalog**: [55 Tools](../reference/tools/catalog.md)
- **Configuration**: [Profiles and Settings](../reference/configuration/)

---

**Next**: [CLI Reference](cli-reference.md) | [Session Management](session-management.md) | [Troubleshooting](troubleshooting.md)
