#!/bin/bash
# Victor Quick Start Script
# This script helps you get started with Victor in under 5 minutes

set -e

echo "======================================================================"
echo "🚀 Victor Quick Start"
echo "======================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "✅ Python $(python3 --version) detected"
echo ""

# Check if vic/victor is already installed
if command -v victor &> /dev/null; then
    echo "✅ Victor is already installed ($(victor --version))"
    ALREADY_INSTALLED=true
else
    echo "📦 Installing Victor..."
    pip install -e .
    echo "✅ Victor installed successfully!"
    ALREADY_INSTALLED=false
fi

echo ""

# Preferred default: a reachable InferFlux server (self-hosted llama.cpp-class
# inference with native tool calls). If it answers, point the default profile
# at it and skip the Ollama setup below.
INFERFLUX_HEALTH="http://127.0.0.1:8080/healthz"
if curl -sf ${INFERFLUX_API_KEY:+-H "Authorization: Bearer $INFERFLUX_API_KEY"} \
        --max-time 3 "$INFERFLUX_HEALTH" > /dev/null 2>&1; then
    echo "✅ InferFlux is reachable at http://127.0.0.1:8080 — default profile will use it"
    echo "   (set INFERFLUX_API_KEY if your server requires auth)"
    echo ""
fi

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running"
    else
        echo "⚠️  Ollama is installed but not running"
        echo "   Start it with: ollama serve"
    fi

    # Check if qwen2.5-coder is available
    if ollama list | grep -q "qwen2.5-coder"; then
        echo "✅ qwen2.5-coder model is available"
    else
        echo "📥 Downloading qwen2.5-coder:7b model (this may take a few minutes)..."
        ollama pull qwen2.5-coder:7b
        echo "✅ Model downloaded successfully!"
    fi
else
    echo "⚠️  Ollama is not installed (optional - only needed for offline mode)"
    echo "   Install from: https://ollama.ai"
fi

echo ""

# Create default profile if it doesn't exist
PROFILES_FILE="$HOME/.victor/profiles.yaml"
if [ ! -f "$PROFILES_FILE" ]; then
    echo "📝 Creating default profile configuration..."
    mkdir -p "$HOME/.victor"

    cat > "$PROFILES_FILE" << 'EOF'
profiles:
  # Default profile (Ollama + offline)
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

  # Claude profile (requires API key)
  claude:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 1.0
    max_tokens: 8192

  # GPT-4 profile (requires API key)
  gpt4:
    provider: openai
    model: gpt-4-turbo
    temperature: 0.8
    max_tokens: 4096

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  openai:
    api_key: ${OPENAI_API_KEY}

  ollama:
    base_url: http://localhost:11434
EOF

    echo "✅ Profile configuration created at $PROFILES_FILE"
else
    echo "✅ Profile configuration already exists"
fi

echo ""

# Download embedding model
echo "📦 Checking embedding model..."
python3 << 'PYEOF'
try:
    from sentence_transformers import SentenceTransformer
    import time

    print("Loading all-MiniLM-L12-v2 (first time may download ~120MB)...")
    start = time.time()
    model = SentenceTransformer('all-MiniLM-L12-v2')
    load_time = time.time() - start

    print(f"✅ Embedding model loaded in {load_time:.2f}s (dimension: {model.get_sentence_embedding_dimension()})")
except ImportError:
    print("⚠️  sentence-transformers not installed (installing...)")
    import subprocess
    subprocess.check_call(['pip', 'install', 'sentence-transformers'])
    print("✅ sentence-transformers installed!")
PYEOF

echo ""

echo "======================================================================"
echo "✅ Quick Start Complete!"
echo "======================================================================"
echo ""
echo "🎉 Victor is ready to use!"
echo ""
echo "Try these commands:"
echo ""
echo "  1. Start interactive mode:"
echo "     $ victor main"
echo ""
echo "  2. Run a one-shot command:"
echo "     $ victor main \"Write a Python function to validate email addresses\""
echo ""
echo "  3. Run demo scripts:"
echo "     $ python3 examples/tool_selection_demo.py"
echo "     $ python3 examples/enterprise_workflow_demo.py"
echo "     $ python3 examples/airgapped_codebase_search.py"
echo ""
echo "  4. Configure API keys (optional - for cloud providers):"
echo "     $ export ANTHROPIC_API_KEY=your-key"
echo "     $ export OPENAI_API_KEY=your-key"
echo ""
echo "  5. Get help:"
echo "     $ victor --help"
echo "     $ victor providers"
echo "     $ victor profiles-cmd"
echo ""
echo "📚 Full documentation: docs/GETTING_STARTED.md"
echo ""
