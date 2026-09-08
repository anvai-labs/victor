# Provider Reference

Complete reference for all 21 supported LLM providers in Victor.

## Quick Links

| Category | Providers | Setup Guide |
|----------|-----------|-------------|
| **Local** | Ollama, LM Studio, vLLM, llama.cpp | [Local Providers →](#local-providers) |
| **Cloud** | Anthropic, OpenAI, Google, xAI, etc. | [Cloud Providers →](#cloud-providers) |
| **Enterprise** | Azure, AWS Bedrock, Vertex AI | [Enterprise →](#enterprise-providers) |
| **Platforms** | Hugging Face, Replicate | [Platforms →](#open-model-platforms) |

## Provider Comparison

### Startup Behavior

Victor provider implementations are registered lazily and imported on first use. This avoids eager imports for optional runtimes and reduces startup cost for `import victor` and `Agent.create()`.

MLX has an extra safety gate:
- `VICTOR_ENABLE_MLX_PROVIDER=0`: disables MLX aliases completely.
- `VICTOR_MLX_SKIP_PREFLIGHT=1`: bypasses the MLX subprocess runtime preflight (only if your MPS/Metal setup is known good).

### Quick Comparison Matrix

| Provider | Models | Tool Calling | Streaming | Vision | API Key | Pricing |
|----------|--------|-------------|-----------|--------|---------|---------|
| **Anthropic** | Claude 3.5, Opus, Haiku | ✅ | ✅ | ✅ | Yes | Pay-per-use |
| **OpenAI** | GPT-4, o1 | ✅ | ✅ | ✅ | Yes | Pay-per-use |
| **Google** | Gemini 2.0 | ✅ | ✅ | ✅ | Yes | Pay-per-use |
| **xAI** | Grok | ✅ | ✅ | ❌ | Yes | Pay-per-use |
| **DeepSeek** | DeepSeek-V3 | ✅ | ✅ | ❌ | Yes | Pay-per-use |
| **Ollama** | qwen2.5, llama3, mistral | ✅ | ✅ | ❌ | No | Free |
| **vLLM** | Any open model | ✅ | ✅ | ✅ | No | Free |
| **Azure** | GPT-4, Claude | ✅ | ✅ | ✅ | Yes | Pay-per-use |
| **AWS Bedrock** | Claude, Llama, Titan | ✅ | ✅ | ✅ | Yes | Pay-per-use |
| **Hugging Face** | 100K+ models | ✅ | ✅ | ✅ | Optional | Free/Paid |
| **Replicate** | 20K+ models | ✅ | ✅ | ✅ | Yes | Pay-per-use |
| **And 9 more...** | See below | Varied | Varied | Varied | Varied | Varied |

[Full Comparison Table →](comparison.md)

### Feature Support Matrix

#### Tool Calling

| Provider | Tool Calling | Notes |
|----------|-------------|-------|
| **Anthropic** | ✅ Native | Excellent tool use |
| **OpenAI** | ✅ Native | Function calling |
| **Google** | ✅ Native | Function calling |
| **Azure** | ✅ Native | Via OpenAI/Anthropic |
| **AWS Bedrock** | ✅ Native | Depends on model |
| **Ollama** | ✅ Native | Via tool-calling format |
| **vLLM** | ✅ Native | Via tool-calling format |
| **Hugging Face** | ✅ Native | Model-dependent |
| **Replicate** | ✅ Native | Model-dependent |
| **OpenRouter** | ✅ Native | Model-dependent |

#### Streaming

| Provider | Streaming | Notes |
|----------|-----------|-------|
| **Anthropic** | ✅ SSE | Server-sent events |
| **OpenAI** | ✅ SSE | Server-sent events |
| **Google** | ✅ SSE | Server-sent events |
| **xAI** | ✅ SSE | Server-sent events |
| **DeepSeek** | ✅ SSE | Server-sent events |
| **Ollama** | ✅ SSE | Server-sent events |
| **vLLM** | ✅ SSE | Server-sent events |
| **All Cloud** | ✅ SSE | Server-sent events |

#### Vision/Multimodal

| Provider | Vision | Notes |
|----------|--------|-------|
| **Anthropic** | ✅ Excellent | Claude 3.5 Sonnet |
| **OpenAI** | ✅ Excellent | GPT-4V, GPT-4o |
| **Google** | ✅ Excellent | Gemini 2.0 Pro |
| **Azure** | ✅ Excellent | Via OpenAI/Anthropic |
| **AWS Bedrock** | ✅ Good | Claude 3, Titan |
| **Ollama** | ❌ Limited | LLaVA variants only |
| **Hugging Face** | ✅ Varied | Model-dependent |

---

## Local Providers

**No API key required.** Run models locally with full privacy and control.

### Ollama (Recommended)

**Best for**: Beginners, ease of use, local development

**Models Available**:
- qwen2.5-coder:7b, qwen2.5:14b, qwen2.5:32b
- llama3.2:3b, llama3.2:7b, llama3.2:14b
- mistral:7b, mixtral:8x7b
- deepseek-coder:6.7b
- codellama:7b, codellama:13b, codellama:34b
- And 100+ more

**Installation**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Start Victor (auto-detects Ollama)
victor chat "Hello!"
```

**Configuration**:
```bash
# Set custom Ollama host
export OLLAMA_HOST=127.0.0.1:11434

# Specify model
victor chat --provider ollama --model qwen2.5-coder:7b
```

**Performance**:
- **Startup**: <1s
- **Memory**: 4-16GB (depends on model size)
- **Speed**: 10-50 tokens/s (depends on hardware)

**Troubleshooting**:
```bash
# Check Ollama status
ollama list

# Test model
ollama run qwen2.5-coder:7b "Hello"

# Check logs
ollama logs
```

[Full Setup Guide →](setup.md#local-example-ollama)

### LM Studio

**Best for**: GUI-based model management, Windows users

**Models Available**:
- Download any GGUF model from Hugging Face
- Popular: Llama 3, Mistral, Qwen2.5, DeepSeek

**Installation**:
```bash
# Download from https://lmstudio.ai/
# Install and run LM Studio
# Load a model in the app

# Start Victor
export VICTOR_LM_STUDIO_HOST=127.0.0.1:1234
victor chat --provider lm-studio
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  lm-studio:
    provider: lm-studio
    base_url: http://127.0.0.1:1234/v1
    model: llama-3.2-3b
```

**Advantages**:
- User-friendly GUI
- Model discovery built-in
- Easy GPU configuration

[Full Setup Guide →](setup.md#other-providers)

### vLLM

**Best for**: Production, high-throughput serving

**Models Available**:
- Any open-source model (Llama, Mistral, Qwen, etc.)
- Optimized for inference speed

**Installation**:
```bash
# Install vLLM
pip install vllm

# Start server
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000

# Start Victor
export VICTOR_VLLM_HOST=127.0.0.1:8000
victor chat --provider vllm
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  vllm:
    provider: vllm
    base_url: http://127.0.0.1:8000/v1
    model: meta-llama/Llama-3.2-3B-Instruct
```

**Advantages**:
- Fastest inference (PagedAttention)
- Production-ready
- OpenAI-compatible API

**Performance**:
- **Throughput**: Up to 20x faster than Ollama
- **Batching**: Automatic request batching
- **Memory**: Efficient with PagedAttention

[Full Setup Guide →](setup.md#other-providers)

### llama.cpp

**Best for**: Maximum compatibility, CPU inference

**Models Available**:
- Any GGUF format model
- Runs on CPU, GPU, Metal, CUDA

**Installation**:
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Start server
./server --model /path/to/model.gguf --port 8080

# Start Victor
export VICTOR_LLAMACPP_HOST=127.0.0.1:8080
victor chat --provider llama-cpp
```

**Advantages**:
- Lightweight (<5MB binary)
- Runs on CPU
- Broad model support
- Low memory footprint

[Full Setup Guide →](setup.md#other-providers)

---

## Cloud Providers

**API key required.** More powerful models with faster execution.

### Anthropic

**Best for**: Complex reasoning, coding, long context

**Models Available**:
- **claude-sonnet-4-20250514**: Latest, best overall (200K context)
- **claude-3-5-sonnet-20241022**: Excellent balance (200K context)
- **claude-3-5-haiku-20241022**: Fast, affordable (200K context)
- **claude-3-opus-20240229**: Most capable (200K context)

**Installation**:
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Start Victor
victor chat --provider anthropic --model claude-sonnet-4-20250514
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 4096
```

**Pricing** (as of 2025):
- **Claude Sonnet**: $3/input MTok, $15/output MTok
- **Claude Haiku**: $0.25/input MTok, $1.25/output MTok
- **Claude Opus**: $15/input MTok, $75/output MTok

**Features**:
- ✅ Excellent tool calling
- ✅ Streaming
- ✅ Vision (images, charts, diagrams)
- ✅ Large context (200K tokens)
- ✅ Artifacts (code preview)

**Rate Limits**:
- Free tier: 5 requests/min
- Paid: 50-100 requests/min (depends on tier)

[Full Documentation →](https://docs.anthropic.com/)

### OpenAI

**Best for**: Code generation, o1 reasoning, broad ecosystem

**Models Available**:
- **gpt-4o**: Latest, multimodal (128K context)
- **gpt-4o-mini**: Fast, affordable (128K context)
- **o1-preview**: Advanced reasoning (limited availability)
- **o1-mini**: Fast reasoning (limited availability)

**Installation**:
```bash
# Set API key
export OPENAI_API_KEY=sk-proj-...

# Start Victor
victor chat --provider openai --model gpt-4o
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  openai:
    provider: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    max_tokens: 4096
```

**Pricing** (as of 2025):
- **GPT-4o**: $2.50/input MTok, $10/output MTok
- **GPT-4o-mini**: $0.15/input MTok, $0.60/output MTok
- **o1-preview**: $15/input MTok, $60/output MTok
- **o1-mini**: $3/input MTok, $12/output MTok

**Features**:
- ✅ Excellent function calling
- ✅ Streaming
- ✅ Vision (images, audio)
- ✅ Large context (128K tokens)
- ✅ o1 reasoning models

**Rate Limits**:
- Tier 1: 3,000 RPM (requests/min)
- Tier 2: 10,000 RPM
- Tier 3: 30,000 RPM

[Full Documentation →](https://platform.openai.com/docs/)

### Google

**Best for**: Multimodal, Gemini features

**Models Available**:
- **gemini-2.0-flash-exp**: Free, fast (1M context)
- **gemini-2.0-flash-thinking-exp**: Reasoning (1M context)
- **gemini-1.5-pro**: Balanced (2M context!)
- **gemini-1.5-flash**: Fast (2M context!)

**Installation**:
```bash
# Set API key
export GOOGLE_API_KEY=...

# Start Victor
victor chat --provider google --model gemini-2.0-flash-exp
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  gemini:
    provider: google
    model: gemini-2.0-flash-exp
    api_key_env: GOOGLE_API_KEY
```

**Pricing** (as of 2025):
- **Gemini 2.0 Flash**: Free (experimental)
- **Gemini 1.5 Pro**: $1.25/input MTok, $5/output MTok
- **Gemini 1.5 Flash**: $0.075/input MTok, $0.30/output MTok

**Features**:
- ✅ Function calling
- ✅ Streaming
- ✅ Vision (images, video, audio)
- ✅ Massive context (up to 2M tokens!)
- ✅ Thinking models

**Advantages**:
- Largest context window (2M tokens)
- Free experimental models
- Excellent multimodal

[Full Documentation →](https://ai.google.dev/gemini-api/docs)

### xAI

**Best for**: Grok models, real-time knowledge

**Models Available**:
- **grok-beta**: Latest (128K context)

**Installation**:
```bash
# Set API key
export XAI_API_KEY=...

# Start Victor
victor chat --provider xai --model grok-beta
```

**Pricing**:
- **Grok Beta**: $5/input MTok, $15/output MTok

**Features**:
- ✅ Function calling
- ✅ Streaming
- ❌ No vision yet

### DeepSeek

**Best for**: Code, Chinese language

**Models Available**:
- **deepseek-chat**: General purpose (128K context)
- **deepseek-coder**: Code-specialized (128K context)

**Installation**:
```bash
# Set API key
export DEEPSEEK_API_KEY=...

# Start Victor
victor chat --provider deepseek --model deepseek-chat
```

**Pricing**:
- **DeepSeek Chat**: $0.27/input MTok, $1.10/output MTok
- **DeepSeek Coder**: $0.14/input MTok, $0.28/output MTok

**Features**:
- ✅ Function calling
- ✅ Streaming
- ❌ No vision yet

**Advantages**:
- Very affordable
- Excellent code generation
- Strong Chinese language support

### Mistral

**Best for**: European models, cost-effective

**Models Available**:
- **mistral-large-latest**: Flagship (128K context)
- **mistral-code-scale**: Code-specialized (32K context)
- **mistral-small-latest**: Fast, affordable (32K context)

**Installation**:
```bash
# Set API key
export MISTRAL_API_KEY=...

# Start Victor
victor chat --provider mistral --model mistral-large-latest
```

**Pricing**:
- **Mistral Large**: $2/input MTok, $6/output MTok
- **Mistral Small**: $0.20/input MTok, $0.60/output MTok

**Features**:
- ✅ Function calling
- ✅ Streaming
- ✅ Vision (selected models)

### Together AI

**Best for**: Open model hosting, fast inference

**Models Available**:
- 100+ open models (Llama, Mistral, Qwen, etc.)
- Fine-tuned models available

**Installation**:
```bash
# Set API key
export TOGETHER_API_KEY=...

# Start Victor
victor chat --provider together --meta-llama/Llama-3.2-3B-Instruct-Turbo
```

**Pricing**:
- Pay-per-use based on model
- Typically $0.10-$1/Mtok

**Features**:
- ✅ Function calling
- ✅ Streaming
- ✅ Vision (selected models)

**Advantages**:
- Fast inference (up to 150 tokens/s)
- Wide model selection
- GPU hosting included

### Fireworks AI

**Best for**: Fast inference, RedPajama models

**Models Available**:
- 50+ open models
- Firework-trained models

**Installation**:
```bash
# Set API key
export FIREWORKS_API_KEY=...

# Start Victor
victor chat --provider fireworks
```

**Pricing**:
- Pay-per-use based on model
- Typically $0.20-$2/Mtok

**Features**:
- ✅ Function calling
- ✅ Streaming
- Fast inference

### OpenRouter

**Best for**: Model marketplace, unified API

**Models Available**:
- 200+ models from multiple providers
- Compare prices and performance

**Installation**:
```bash
# Set API key
export OPENROUTER_API_KEY=...

# Start Victor
victor chat --provider openrouter --model anthropic/claude-sonnet-4
```

**Pricing**:
- Pay-per-use
- Small fee on top of provider pricing

**Features**:
- ✅ Function calling
- ✅ Streaming
- ✅ Vision (selected models)
- Model comparison

**Advantages**:
- Single API for all providers
- Price comparison
- Fallback routing

### Groq

**Best for**: Ultra-fast inference

**Models Available**:
- Llama, Mistral, Gemma models
- Hosted on Groq LPU

**Installation**:
```bash
# Set API key
export GROQ_API_KEY=...

# Start Victor
victor chat --provider groq --model llama3.2-3b-8k
```

**Pricing**:
- Free tier available
- Paid: $0.19-$0.59/Mtok

**Features**:
- ✅ Function calling
- ✅ Streaming
- ❌ No vision yet

**Advantages**:
- **Fastest inference** (300+ tokens/s)
- Low latency
- Free tier

### Moonshot

**Best for**: Chinese models, cost-effective

**Models Available**:
- moonshot-v1-8k
- moonshot-v1-32k
- moonshot-v1-128k

**Installation**:
```bash
# Set API key
export MOONSHOT_API_KEY=...

# Start Victor
victor chat --provider moonshot
```

**Pricing**:
- ¥12/input MTok, ¥12/output MTok

**Features**:
- ✅ Function calling
- ✅ Streaming

### Cerebras

**Best for**: Ultra-fast inference

**Models Available**:
- Llama, Mistral models
- Hosted on Cerebras CS-2 hardware

**Installation**:
```bash
# Set API key
export CEREBRAS_API_KEY=...

# Start Victor
victor chat --provider cerebras
```

**Pricing**:
- Pay-per-use
- Very competitive

**Features**:
- ✅ Function calling
- ✅ Streaming
- Fastest inference (500+ tokens/s)

---

## Enterprise Providers

**For enterprise deployments** with compliance and security requirements.

### Azure OpenAI

**Best for**: Enterprise compliance, Azure integration

**Models Available**:
- GPT-4, GPT-4o, o1
- Same as OpenAI but hosted on Azure

**Installation**:
```bash
# Set credentials
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...

# Start Victor
victor chat --provider azure
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  azure:
    provider: azure
    api_key_env: AZURE_OPENAI_API_KEY
    api_base_env: AZURE_OPENAI_ENDPOINT
    api_version: 2024-02-01
    deployment: gpt-4o
```

**Advantages**:
- Azure AD integration
- Compliance (SOC2, HIPAA, GDPR)
- Private networking
- Data residency

### AWS Bedrock

**Best for**: AWS ecosystem, enterprise deployment

**Models Available**:
- Claude 3 (Anthropic)
- Llama 3 (Meta)
- Titan (Amazon)
- Mistral AI models

**Installation**:
```bash
# Configure AWS CLI
aws configure

# Start Victor
victor chat --provider bedrock --model anthropic.claude-3-sonnet-20240229-v1:0
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  bedrock:
    provider: bedrock
    region_name: us-east-1
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
```

**Advantages**:
- AWS IAM authentication
- VPC endpoints
- Compliance (SOC2, HIPAA)
- Data encryption

### Google Vertex AI

**Best for**: Google Cloud, enterprise deployment

**Models Available**:
- Gemini 1.5/2.0
- Same as Google AI but hosted on Vertex AI

**Installation**:
```bash
# Configure gcloud
gcloud auth application-default login

# Start Victor
victor chat --provider vertex --model gemini-2.0-flash-exp
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  vertex:
    provider: vertex
    project: your-project-id
    location: us-central1
    model: gemini-2.0-flash-exp
```

**Advantages**:
- Google Cloud authentication
- VPC endpoints
- Compliance (SOC2, ISO)
- Enterprise support

---

## Open Model Platforms

### Hugging Face

**Best for**: 100K+ models, research models

**Models Available**:
- 100,000+ models on Hugging Face Hub
- Llama, Mistral, Qwen, BLOOM, etc.
- Task-specific models (code, math, reasoning)

**Installation**:
```bash
# Optional: Set API key for private models
export HF_API_KEY=...

# Start Victor (no API key needed for public models)
victor chat --provider huggingface --model Qwen/Qwen2.5-Coder-7B-Instruct
```

**Configuration**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  huggingface:
    provider: huggingface
    model: Qwen/Qwen2.5-Coder-7B-Instruct
    api_key_env: HF_API_KEY  # Optional
```

**Pricing**:
- Free for inference (serverless)
- Paid Inference Endpoints available
- PRO subscription for faster inference

**Features**:
- ✅ Function calling (model-dependent)
- ✅ Streaming
- ✅ Vision (model-dependent)
- Model discovery
- Dataset integration

**Advantages**:
- Largest model collection
- Free tier available
- Research models
- Community models

### Replicate

**Best for**: Easy model hosting, API access

**Models Available**:
- 20,000+ models
- Diffusion, LLMs, audio, video

**Installation**:
```bash
# Set API key
export REPLICATE_API_TOKEN=...

# Start Victor
victor chat --provider replicate --model meta/llama-3.2-3b-instruct
```

**Pricing**:
- Pay-per-use
- Hardware costs passed through

**Features**:
- ✅ Function calling (model-dependent)
- ✅ Streaming
- ✅ Vision (model-dependent)
- Model versioning
- Easy deployment

---

## Choosing a Provider

### Decision Guide

**For Local Development**:
1. **Beginner**: Ollama (easiest setup)
2. **Windows**: LM Studio (GUI)
3. **Production**: vLLM (fastest)
4. **CPU-only**: llama.cpp

**For Cloud Development**:
1. **Best Overall**: Anthropic Claude Sonnet
2. **Code Generation**: OpenAI GPT-4o, DeepSeek Coder
3. **Cost-Effective**: Google Gemini 2.0 Flash (free)
4. **Fastest**: Groq, Cerebras (300+ tokens/s)
5. **Large Context**: Google Gemini 1.5 Pro (2M tokens)

**For Enterprise**:
1. **Azure**: Azure OpenAI (Azure integration)
2. **AWS**: AWS Bedrock (compliance)
3. **GCP**: Vertex AI (Google Cloud)

**For Experimentation**:
1. **Model Variety**: Hugging Face (100K+ models)
2. **Price Comparison**: OpenRouter (unified API)
3. **Research**: Together AI (latest models)

### Provider Switching

**Victor's key feature**: Switch providers mid-conversation without losing context.

```bash
# Start with Claude
victor chat --provider anthropic "Design a REST API"

# Switch to GPT-4 for implementation
/provider openai --model gpt-4o

# Finish with local model for privacy
/provider ollama --model qwen2.5-coder:7b
```

**Benefits**:
- Leverage different model strengths
- Cost optimization (use cheaper models for simple tasks)
- Privacy (use local models for sensitive data)
- Redundancy (fallback if provider is down)

### Configuration Examples

**Development Profile**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  development:
    provider: anthropic
    model: claude-sonnet-4-20250514
    max_tokens: 4096
    temperature: 0.7
```

**Production Profile**:
```yaml
profiles:
  production:
    provider: azure
    deployment: gpt-4o
    api_key_env: AZURE_OPENAI_API_KEY
    api_base_env: AZURE_OPENAI_ENDPOINT
    max_tokens: 8192
```

**Cost-Optimized Profile**:
```yaml
profiles:
  cost-optimized:
    provider: google
    model: gemini-2.0-flash-exp  # Free
    max_tokens: 4096
```

**Local Profile**:
```yaml
profiles:
  local:
    provider: ollama
    model: qwen2.5-coder:7b
    # No API key needed
```

## Environment Variables Reference

| Provider | Environment Variable | Required |
|----------|---------------------|----------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Yes |
| **OpenAI** | `OPENAI_API_KEY` | Yes |
| **Google** | `GOOGLE_API_KEY` | Yes |
| **xAI** | `XAI_API_KEY` | Yes |
| **DeepSeek** | `DEEPSEEK_API_KEY` | Yes |
| **Mistral** | `MISTRAL_API_KEY` | Yes |
| **Together** | `TOGETHER_API_KEY` | Yes |
| **Fireworks** | `FIREWORKS_API_KEY` | Yes |
| **OpenRouter** | `OPENROUTER_API_KEY` | Yes |
| **Groq** | `GROQ_API_KEY` | Yes |
| **Moonshot** | `MOONSHOT_API_KEY` | Yes |
| **Cerebras** | `CEREBRAS_API_KEY` | Yes |
| **Azure** | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | Yes |
| **AWS Bedrock** | AWS CLI credentials | Yes |
| **Vertex AI** | gcloud authentication | Yes |
| **Hugging Face** | `HF_API_KEY` | Optional |
| **Replicate** | `REPLICATE_API_TOKEN` | Yes |
| **Ollama** | `OLLAMA_HOST` | Optional |
| **LM Studio** | `VICTOR_LM_STUDIO_HOST` | Optional |
| **vLLM** | `VICTOR_VLLM_HOST` | Optional |
| **llama.cpp** | `VICTOR_LLAMACPP_HOST` | Optional |

## Troubleshooting

### Common Issues

**1. API Key Not Found**:
```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Set in current shell
export ANTHROPIC_API_KEY=sk-...

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export ANTHROPIC_API_KEY=sk-...' >> ~/.bashrc
```

**2. Provider Not Available**:
```bash
# Check installed providers
victor providers

# Reinstall with provider extras
pip install "victor-ai[anthropic,openai]"
```

**3. Model Not Found**:
```bash
# List available models for provider
victor providers --provider anthropic

# Check model name spelling
victor chat --provider anthropic --model claude-sonnet-4-20250514
```

**4. Connection Timeout**:
```bash
# Check network connectivity
curl https://api.anthropic.com/v1/messages

# Increase timeout
victor chat --timeout 60
```

**5. Rate Limiting**:
```bash
# Switch provider
/provider openai

# Or use local model
/provider ollama
```

### Getting Help

- **Documentation**: [Full docs →](../../index.md)
- **Issues**: [GitHub Issues →](https://github.com/anvai-labs/victor/issues)
- **Discussions**: [GitHub Discussions →](https://github.com/anvai-labs/victor/discussions)
- **Setup Guide**: [Detailed setup →](setup.md)

---

**Next**: [Provider Comparison →](comparison.md) | [Tool Catalog →](../tools/catalog.md) | [Configuration →](../configuration/index.md)
