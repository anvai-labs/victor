# Installation Guide

Detailed installation instructions for Victor AI framework with troubleshooting help.

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 2GB RAM minimum, 4GB+ recommended
- **Disk Space**: 500MB for package, 1GB+ for development dependencies

### Recommended

- **Python**: 3.11 or 3.12
- **Memory**: 8GB+ RAM
- **Disk Space**: 2GB+ for full development environment

## Verify Python Version

Before installing, check your Python version:

```bash
python --version
# or
python3 --version
```

You should see `Python 3.11.0` or higher.

**If you have Python < 3.10**:

**macOS** (using Homebrew):
```bash
brew install python@3.11
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.11
```

**Windows**: Download from [python.org](https://www.python.org/downloads/)

---

## Installation Methods

### Method 1: pipx (Recommended for CLI)

pipx installs Victor in an isolated environment:

```bash
# Install pipx if needed
pip install pipx
pipx ensurepath

# Install Victor
pipx install victor-ai

# Verify
victor --version
```

### Method 2: pip (Virtual Environment)

For Python projects:

```bash
# Create virtual environment
python -m venv victor-env
source victor-env/bin/activate  # Windows: victor-env\Scripts\activate

# Install Victor
pip install victor-ai

# Verify
victor --version
```

### Method 3: Install with Extras

Install with optional dependencies:

```bash
# Development tools (pytest, black, mypy)
pip install "victor-ai[dev]"

# All optional dependencies
pip install "victor-ai[all]"

# Specific extras
pip install "victor-ai[api,google,checkpoints]"
```

Available extras:
- `dev` - Development tools
- `api` - FastAPI HTTP server
- `google` - Google Gemini provider
- `checkpoints` - SQLite workflow persistence
- `native` - Rust extensions (Maturin)
- `all` - All optional dependencies

### Method 4: Docker

```bash
# Pull the image
docker pull ghcr.io/vjsingh1984/victor:latest

# Run interactively
docker run -it \
  -v ~/.victor:/root/.victor \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ghcr.io/vjsingh1984/victor:latest chat
```

### Method 5: Development Installation

For contributors:

```bash
# Clone repository
git clone https://github.com/anvai-labs/victor.git
cd victor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode (contracts first — victor-ai depends on the in-repo SDK)
pip install -e ./victor-contracts -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/unit -v
```

---

## Provider Configuration

Victor needs an LLM provider. Choose **local** (no API key) or **cloud** (API key required).

### Option A: Local Model (Ollama - Free, No API Key)

```bash
# Install Ollama
brew install ollama  # macOS
# OR
curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Start Ollama
ollama serve

# Pull a code-focused model
ollama pull qwen2.5-coder:7b

# Test with Victor
victor chat --provider ollama --model qwen2.5-coder:7b "Hello!"
```

### Option B: Cloud Providers (API Key Required)

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-your-key

# OpenAI
export OPENAI_API_KEY=sk-proj-your-key

# Google Gemini
export GOOGLE_API_KEY=your-key

# Azure OpenAI
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
```

**Secure key storage (recommended)**:
```bash
victor keys --set anthropic --keyring
victor keys --set openai --keyring
```

---

## Post-Installation Setup

### Initialize Victor

```bash
victor init
```

This creates:
- `~/.victor/profiles.yaml` - Provider and model profiles
- `~/.victor/config.yaml` - Global settings

### Verify Installation

```bash
# Check version
victor --version

# List providers
victor providers

# List tools
victor tools

# Run diagnostics
victor doctor
```

Expected output:
```
Victor AI v0.x.x
24 providers available
34 tools available
```

---

## Troubleshooting

### "command not found: victor"

**Cause**: Victor not in PATH

**Solutions**:
```bash
# If using pipx
pipx ensurepath
source ~/.bashrc  # or ~/.zshrc

# If using pip
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### "No module named 'victor'"

**Cause**: Wrong Python environment or not installed

**Solutions**:
```bash
# Check which Python
which python

# Install Victor
pip install victor-ai

# Verify installation
python -m pip show victor-ai
```

### "Permission denied" when installing

**Solution**: Use pipx or virtual environment:
```bash
# Recommended
pipx install victor-ai

# Alternative
python -m venv .venv
source .venv/bin/activate
pip install victor-ai
```

### "SSL certificate verification failed"

**Solution**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org victor-ai
```

### "Failed building wheel for X"

**Cause**: Missing build dependencies

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# macOS
xcode-select --install

# Windows: Install Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### "ImportError: cannot import name 'cached_property'"

**Cause**: Python version too old

**Solution**: Upgrade to Python 3.11+
```bash
python --version  # Should be 3.11+
```

### Ollama Connection Errors

**Cause**: Ollama server not running

**Solutions**:
```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434/api/version

# Pull a model
ollama pull qwen2.5-coder:7b

# Set custom host if needed
export OLLAMA_HOST=127.0.0.1:11434
```

### OpenSSL Issues (macOS Apple Silicon)

**Solution**:
```bash
brew install openssl
export LDFLAGS="-L/opt/homebrew/opt/openssl/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openssl/include"
pip install victor-ai
```

### Version Conflicts

**Solution**: Use pip-check to find conflicts:
```bash
pip install pip-check
pip-check
```

---

## Platform-Specific Notes

### macOS

```bash
# Install Python 3.11+ if needed
brew install python@3.11

# Install pipx
brew install pipx
pipx ensurepath

# Install Victor
pipx install victor-ai

# Optional: Ollama for local models
brew install ollama
```

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.11+
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Install pipx
python3.11 -m pip install --user pipx
python3.11 -m pipx ensurepath

# Restart terminal, then install Victor
pipx install victor-ai
```

### Windows (WSL2 Recommended)

```powershell
# Install WSL2 (from PowerShell as Administrator)
wsl --install

# After restart, in Ubuntu terminal:
sudo apt update && sudo apt upgrade
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 -m pip install --user pipx
python3.11 -m pipx ensurepath

# Restart terminal, then install Victor
pipx install victor-ai
```

---

## Virtual Environments

### venv

```bash
# Create
python -m venv victor-env

# Activate
source victor-env/bin/activate  # Linux/macOS
victor-env\Scripts\activate     # Windows

# Install
pip install victor-ai

# Deactivate
deactivate
```

### conda

```bash
# Create environment
conda create -n victor python=3.11

# Activate
conda activate victor

# Install
pip install victor-ai

# Deactivate
conda deactivate
```

### poetry

```bash
# Create project
poetry new my-project
cd my-project

# Add Victor
poetry add victor-ai

# Install
poetry install
```

---

## Upgrading

### pipx
```bash
pipx upgrade victor-ai
```

### pip
```bash
pip install --upgrade victor-ai
```

### Docker
```bash
docker pull ghcr.io/vjsingh1984/victor:latest
```

### Development
```bash
git pull origin main
pip install -e ./victor-contracts -e ".[dev]"
```

---

## Uninstalling

### pipx
```bash
pipx uninstall victor-ai
```

### pip
```bash
pip uninstall victor-ai
```

### Remove configuration
```bash
rm -rf ~/.victor
```

---

## Next Steps

- 📖 [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- 🤖 [First Agent Guide](first-agent.md) - Build your first agent
- 🔄 [First Workflow Guide](first-workflow.md) - Create workflows
- ⚙️ [Configuration Guide](configuration.md) - Customize settings

---

## Additional Resources

- [PyPI Package](https://pypi.org/project/victor-ai/)
- [GitHub Repository](https://github.com/anvai-labs/victor)
- [Documentation](https://victor-ai.readthedocs.io)

**Need help?** Run `victor doctor` for diagnostics or see [Quick Start](quickstart.md#troubleshooting).
