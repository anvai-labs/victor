# Troubleshooting Guide

Common issues and solutions for using Victor effectively.

## Quick Diagnostics

**Start here** if you're experiencing problems:

```bash
# 1. Check Victor version
victor --version

# 2. Check installed providers
victor providers

# 3. Test basic functionality
victor chat "Hello, Victor!" --provider ollama

# 4. Check configuration
victor config show

# 5. View logs
victor logs --tail 50
```

**Expected Output**:
- Version should be latest: `victor 0.x.x`
- Providers should list 24 providers
- Chat should respond successfully
- Config should show your profiles
- Logs should show recent activity

[Still having issues? → Jump to specific sections below](#common-issues)

---

## Installation Issues

### Issue: Command Not Found

**Symptom**:
```bash
victor --version
# zsh: command not found: victor
```

**Solutions**:

**1. Check installation**:
```bash
# Verify Victor is installed
pip list | grep victor

# If not installed, install it
pipx install victor-ai
# or
pip install victor-ai
```

**2. Check PATH** (if using pip):
```bash
# Find where pip installed Victor
pip show victor-ai | grep Location

# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**3. Reinstall with pipx** (recommended):
```bash
# Uninstall existing
pip uninstall victor-ai

# Install with pipx
pipx install victor-ai

# Verify
which victor
# Should output: /home/user/.local/bin/victor
```

### Issue: Python Version Incompatible

**Symptom**:
```bash
# During installation
ERROR: Package 'victor-ai' requires a different Python: 3.10.x not in '>=3.11'
```

**Solutions**:

**1. Check Python version**:
```bash
python --version
# Must be 3.10 or higher
```

**2. Install Python 3.11+**:

**macOS**:
```bash
brew install python@3.11
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.11
```

**Windows**:
- Download from [python.org](https://www.python.org/downloads/)

**3. Use correct Python**:
```bash
# Use python3.11 explicitly
python3.11 -m pip install victor-ai

# Or create virtual environment
python3.11 -m venv ~/.victor-venv
source ~/.victor-venv/bin/activate  # Windows: ~/.victor-venv\Scripts\activate
pip install victor-ai
```

### Issue: Dependencies Failed to Install

**Symptom**:
```bash
# During installation
ERROR: Could not build wheels for some packages
```

**Solutions**:

**1. Install build tools**:

**macOS**:
```bash
xcode-select --install
```

**Ubuntu/Debian**:
```bash
sudo apt install build-essential python3-dev
```

**Windows**:
- Install Visual Studio Build Tools

**2. Install with dev dependencies**:
```bash
pip install "victor-ai[dev]"
```

**3. Use pre-built wheels**:
```bash
pip install --only-binary :all: victor-ai
```

### Issue: Permission Denied

**Symptom**:
```bash
# During installation
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solutions**:

**1. Use pipx** (recommended):
```bash
pip install pipx
pipx install victor-ai
```

**2. Use virtual environment**:
```bash
python -m venv ~/.victor-venv
source ~/.victor-venv/bin/activate
pip install victor-ai
```

**3. Use --user flag** (not recommended):
```bash
pip install --user victor-ai
```

---

## Provider Issues

### Issue: API Key Not Found

**Symptom**:
```bash
victor chat --provider anthropic "Hello"
# ERROR: API key not found for provider 'anthropic'
```

**Solutions**:

**1. Set environment variable**:
```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Verify
echo $ANTHROPIC_API_KEY
```

**2. Add to shell profile** (persistent):
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

**3. Use profiles.yaml**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  claude:
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
```

**4. Set directly** (not recommended, insecure):
```bash
victor chat --provider anthropic --api-key sk-ant-...
```

**Environment Variables by Provider**:
```bash
# Cloud providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export MISTRAL_API_KEY=...

# Enterprise
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...

# Local (optional)
export OLLAMA_HOST=127.0.0.1:11434
export VICTOR_LM_STUDIO_HOST=127.0.0.1:1234
export VICTOR_VLLM_HOST=127.0.0.1:8000
export VICTOR_LLAMACPP_HOST=127.0.0.1:8080
```

### Issue: Provider Not Available

**Symptom**:
```bash
victor chat --provider together "Hello"
# ERROR: Provider 'together' not available
```

**Solutions**:

**1. Check installed providers**:
```bash
victor providers

# Should list 24 providers
```

**2. Install provider extras**:
```bash
# Reinstall with specific provider
pip install "victor-ai[together]"

# Or install all providers
pip install "victor-ai[all]"
```

**3. Check provider spelling**:
```bash
# Correct names
victor chat --provider anthropic "Hello"
victor chat --provider openai "Hello"
victor chat --provider ollama "Hello"

# NOT
victor chat --provider Anthropic "Hello"  # Wrong case
victor chat --provider open_ai "Hello"   # Wrong separator
```

### Issue: Model Not Found

**Symptom**:
```bash
victor chat --provider anthropic --model claude-3 "Hello"
# ERROR: Model 'claude-3' not found
```

**Solutions**:

**1. List available models**:
```bash
victor providers --provider anthropic --models

# Or check documentation
# https://docs.victor.ai/reference/providers/
```

**2. Use correct model name**:
```bash
# Correct
victor chat --provider anthropic --model claude-sonnet-4-20250514

# NOT
victor chat --provider anthropic --model claude-3  # Too generic
```

**3. Pull model first** (for local providers):
```bash
# Ollama
ollama pull qwen2.5-coder:7b

# Then use
victor chat --provider ollama --model qwen2.5-coder:7b
```

### Issue: Connection Timeout

**Symptom**:
```bash
victor chat --provider anthropic "Hello"
# ERROR: Connection timeout after 30s
```

**Solutions**:

**1. Check network connectivity**:
```bash
# Test API endpoint
curl https://api.anthropic.com/v1/messages

# Check DNS
nslookup api.anthropic.com
```

**2. Check firewall/proxy**:
```bash
# Set proxy if needed
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

**3. Increase timeout**:
```bash
victor chat --timeout 60 "Hello"
```

**4. Test with local provider**:
```bash
# If cloud fails, try local
victor chat --provider ollama "Hello"
```

### Issue: Rate Limiting

**Symptom**:
```bash
victor chat --provider anthropic "Hello"
# ERROR: Rate limit exceeded
```

**Solutions**:

**1. Wait and retry**:
```bash
# Wait 60 seconds
sleep 60
victor chat "Hello"
```

**2. Switch providers**:
```bash
# During conversation
/provider openai

# Or start new conversation
victor chat --provider openai "Hello"
```

**3. Use local provider as fallback**:
```bash
/provider ollama  # No rate limits
```

**4. Check your rate limit**:
```bash
# Anthropic
# https://console.anthropic.com/settings/limits

# OpenAI
# https://platform.openai.com/usage

# Google
# https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
```

### Issue: Authentication Failed

**Symptom**:
```bash
victor chat --provider anthropic "Hello"
# ERROR: Authentication failed (401)
```

**Solutions**:

**1. Verify API key is valid**:
```bash
# Test with curl
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

**2. Regenerate API key**:
- Visit provider console
- Delete old key
- Create new key
- Update environment variable

**3. Check for typos**:
```bash
# Make sure no extra spaces
echo $ANTHROPIC_API_KEY | tr -d ' '

# Should not have leading/trailing spaces
```

### Issue: Ollama Not Responding

**Symptom**:
```bash
victor chat --provider ollama "Hello"
# ERROR: Ollama not responding
```

**Solutions**:

**1. Check Ollama status**:
```bash
# List running models
ollama list

# Check logs
ollama logs

# Test directly
ollama run qwen2.5-coder:7b "Hello"
```

**2. Restart Ollama**:
```bash
# Stop Ollama
ollama stop

# Start Ollama
ollama serve

# In another terminal, test
victor chat --provider ollama "Hello"
```

**3. Pull model**:
```bash
ollama pull qwen2.5-coder:7b
```

**4. Check Ollama host**:
```bash
# Default is 127.0.0.1:11434
export OLLAMA_HOST=127.0.0.1:11434
```

### Issue: vLLM Connection Failed

**Symptom**:
```bash
victor chat --provider vllm "Hello"
# ERROR: Failed to connect to vLLM server
```

**Solutions**:

**1. Start vLLM server**:
```bash
# Install vLLM
pip install vllm

# Start server
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000
```

**2. Check vLLM status**:
```bash
curl http://127.0.0.1:8000/v1/models
```

**3. Set host**:
```bash
export VICTOR_VLLM_HOST=127.0.0.1:8000
```

**4. Check GPU/memory**:
```bash
# Check GPU availability
nvidia-smi

# Check memory
free -h
```

---

## Performance Issues

### Issue: Slow Responses

**Symptom**:
```bash
victor chat "Generate a REST API"
# Takes 30+ seconds to respond
```

**Solutions**:

**1. Check which provider you're using**:
```bash
# Cloud providers: 1-5 seconds (normal)
# Local providers: 5-30 seconds (depends on hardware)
victor providers --current
```

**2. Switch to faster provider**:
```bash
# Groq, Cerebras: <1 second
/provider groq

# Or local with GPU
/provider ollama  # If you have GPU
```

**3. Use smaller model**:
```bash
# For simple tasks
victor chat --model gemini-2.0-flash-exp "Quick question"
# or
victor chat --provider ollama --model llama3.2:3b "Quick question"
```

**4. Enable caching**:
```yaml
# ~/.victor/config.yaml
cache:
  enabled: true
  ttl: 3600
```

**5. Check your internet**:
```bash
# Test speed
ping api.anthropic.com
traceroute api.anthropic.com
```

### Issue: High Memory Usage

**Symptom**:
```bash
# Victor consuming >1GB memory
```

**Solutions**:

**1. Check memory usage**:
```bash
# On Linux/macOS
ps aux | grep victor

# On Windows
tasklist | findstr victor
```

**2. Reduce context size**:
```bash
victor chat --max-tokens 1024 "Hello"
```

**3. Use smaller model**:
```bash
# Local models vary in size
# 3B model: ~4GB RAM
# 7B model: ~8GB RAM
# 14B model: ~16GB RAM
victor chat --provider ollama --model llama3.2:3b "Hello"
```

**4. Clear cache**:
```bash
# Clear conversation cache
victor cache clear

# Clear all cache
rm -rf ~/.victor/cache/
```

**5. Restart Victor**:
```bash
# Stop and restart
victor quit
victor chat "Hello"
```

### Issue: CPU Usage 100%

**Symptom**:
```bash
# Victor consuming 100% CPU
```

**Solutions**:

**1. Check if using local model**:
```bash
# Local models use CPU heavily (normal)
victor providers --current
```

**2. Use GPU if available**:
```bash
# For Ollama
ollama run qwen2.5-coder:7b

# Check GPU usage
nvidia-smi
```

**3. Switch to cloud provider**:
```bash
/provider anthropic  # Uses provider's compute
```

**4. Use smaller local model**:
```bash
victor chat --provider ollama --model llama3.2:3b
```

---

## Configuration Issues

### Issue: Profile Not Found

**Symptom**:
```bash
victor --profile production chat "Hello"
# ERROR: Profile 'production' not found
```

**Solutions**:

**1. List available profiles**:
```bash
victor config profiles
```

**2. Create profile**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  production:
    provider: anthropic
    model: claude-sonnet-4-20250514
```

**3. Check profile syntax**:
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('~/.victor/profiles.yaml'))"
```

**4. Use default profile**:
```bash
victor chat "Hello"  # Uses default profile
```

### Issue: Invalid Configuration

**Symptom**:
```bash
victor chat "Hello"
# ERROR: Invalid configuration in ~/.victor/profiles.yaml
```

**Solutions**:

**1. Validate YAML syntax**:
```bash
# Check for syntax errors
cat ~/.victor/profiles.yaml | python -m yaml
```

**2. Check common YAML errors**:
```yaml
# WRONG (spaces vs tabs)
profiles:
  production:    # 2 spaces
    provider: anthropic
  └── model: claude  # Tab character (wrong!)

# CORRECT
profiles:
  production:      # 2 spaces
    provider: anthropic
    model: claude-sonnet-4-20250514  # 4 spaces (2+2)
```

**3. Reset configuration**:
```bash
# Backup current config
mv ~/.victor/profiles.yaml ~/.victor/profiles.yaml.bak

# Create new config
victor config init
```

### Issue: Settings Not Applied

**Symptom**:
```bash
# Settings in profiles.yaml not being used
```

**Solutions**:

**1. Check profile is being used**:
```bash
# Explicitly specify profile
victor --profile production chat "Hello"

# Or check default profile
victor config show
```

**2. Check for conflicting CLI flags**:
```bash
# CLI flags override profile settings
victor chat --provider ollama "Hello"  # Overrides profile
```

**3. Verify config location**:
```bash
# Should be at ~/.victor/profiles.yaml
ls -la ~/.victor/profiles.yaml
```

**4. Check for multiple configs**:
```bash
# Only one config file should exist
find ~ -name "profiles.yaml" -path "*/.victor/*"
```

---

## Context Issues

### Issue: Context Not Preserved Between Providers

**Symptom**:
```bash
/provider openai
# Loses conversation history
```

**Solutions**:

**1. Check ConversationController**:
```bash
# Context should be preserved automatically
# If not, check logs
victor logs --tail 100 | grep -i context
```

**2. Use same session**:
```bash
# Don't exit and restart
# Stay in same conversation
```

**3. Report bug**:
```bash
# If context is lost, this is a bug
# Create issue with logs
victor logs > victor-logs.txt
# Upload to GitHub issue
```

### Issue: Context Too Long

**Symptom**:
```bash
victor chat "Continue"
# ERROR: Context exceeds model limit
```

**Solutions**:

**1. Clear conversation**:
```bash
# Start new conversation
victor chat --new "Hello"
```

**2. Summarize and continue**:
```bash
# Ask Victor to summarize
victor chat "Summarize our conversation so far"

# Start new conversation with summary
victor chat --new "Here's context: [paste summary]. Continue with..."
```

**3. Use model with larger context**:
```bash
# Claude, Gemini: 1M-2M tokens
victor chat --provider anthropic "Continue"

# Check context window
victor providers --provider anthropic --info
```

---

## Tool Issues

### Issue: Tool Execution Failed

**Symptom**:
```bash
victor chat "Read auth.py"
# ERROR: Tool execution failed: read_file
```

**Solutions**:

**1. Check tool is available**:
```bash
victor tools | grep read_file
```

**2. Check file permissions**:
```bash
# File must be readable
ls -la auth.py

# Fix permissions
chmod +r auth.py
```

**3. Check file exists**:
```bash
# Use absolute path
victor chat "Read /home/user/project/auth.py"

# Or check current directory
pwd
```

**4. Enable debug logging**:
```bash
victor --debug chat "Read auth.py"
```

### Issue: Tool Not Found

**Symptom**:
```bash
victor chat "Execute pytest"
# ERROR: Tool 'pytest' not found
```

**Solutions**:

**1. List available tools**:
```bash
victor tools
```

**2. Install tool dependencies**:
```bash
# Some tools require external commands
# e.g., pytest requires pytest to be installed
pip install pytest
```

**3. Check tool name**:
```bash
# Tool names are specific
victor chat "Run tests"  # Victor chooses right tool
# NOT
victor chat "Execute tool pytest"  # Wrong
```

### Issue: Tool Deduplication - Expected Tool Missing

**Symptom**:
```bash
victor tools list
# Expected tool not in list (e.g., lgc_wikipedia missing)
```

**Solutions**:

**1. Check if tool was deduplicated**:
```bash
# Check logs for deduplication messages
victor logs --tail 100 | grep -i dedup
# Should show: "Skipped lgc_wikipedia (source=langchain) in favor of wikipedia"
```

**2. Check deduplication settings**:
```bash
# View current deduplication configuration
victor config show | grep dedup
```

**3. Add tool to whitelist** (if you need both):
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    tools:
      deduplication_whitelist:
        - wikipedia
        - lgc_wikipedia
```

**4. Disable deduplication** (not recommended):
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    tools:
      enable_tool_deduplication: false
```

**5. Check priority order**:
```bash
# Native tools have priority over adapters
# If you have native 'wikipedia', 'lgc_wikipedia' will be skipped
victor tools list | grep wikipedia
```

### Issue: Tool Has Unexpected Prefix

**Symptom**:
```bash
victor tools list
# Shows mcp_github_search instead of github_search
# Or lgc_wikipedia instead of wikipedia
```

**Solutions**:

**1. Check if naming enforcement is enabled**:
```bash
victor config show | grep naming_enforcement
# Should show: deduplication_naming_enforcement: true
```

**2. This is expected behavior**:
```bash
# LangChain tools get lgc_ prefix
# MCP tools get mcp_ prefix  
# Plugin tools get plg_ prefix
# Native tools have no prefix
```

**3. Disable naming enforcement** (if needed):
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    tools:
      deduplication_naming_enforcement: false
```

**4. Use full tool name**:
```bash
# Reference tool with prefix
victor chat "Search GitHub using mcp_github_search"
```

### Issue: Too Many Duplicate Tools

**Symptom**:
```bash
victor tools list
# Shows multiple tools with similar names
# e.g., search, lgc_search, mcp_search
```

**Solutions**:

**1. Check if deduplication is enabled**:
```bash
victor config show | grep enable_tool_deduplication
```

**2. Enable deduplication**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    tools:
      enable_tool_deduplication: true
```

**3. Check priority order**:
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    tools:
      deduplication_priority_order:
        - native      # Highest priority
        - langchain
        - mcp
        - plugin      # Lowest priority
```

**4. Verify logs show deduplication**:
```bash
victor logs --tail 50 | grep -i "conflict resolved"
# Should show which tools were kept/skipped
```

---

## Log Analysis

### Viewing Logs

```bash
# View recent logs
victor logs --tail 50

# Follow logs
victor logs --follow

# Filter by level
victor logs --level ERROR

# Save logs to file
victor logs > victor-logs.txt
```

### Finding Errors

```bash
# Search for errors
victor logs | grep ERROR

# Search for warnings
victor logs | grep WARN

# Search for specific provider
victor logs | grep anthropic

# Search for tool execution
victor logs | grep tool.execution
```

### Log Levels

| Level | Description | When to Use |
|-------|-------------|-------------|
| **DEBUG** | Detailed diagnostic info | Troubleshooting, development |
| **INFO** | General informational | Normal operation |
| **WARNING** | Warning messages | Potential issues |
| **ERROR** | Error messages | Failures, exceptions |
| **CRITICAL** | Critical failures | Crashes, data loss |

**Set log level**:
```yaml
# ~/.victor/config.yaml
logging:
  level: DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: /var/log/victor.log
```

---

## When to Open an Issue

**Before opening an issue**:
1. Check existing issues: https://github.com/anvai-labs/victor/issues
2. Search discussions: https://github.com/anvai-labs/victor/discussions
3. Try troubleshooting steps above
4. Collect logs and error messages

**When to open an issue**:
- ✅ Bug: Victor crashes or behaves incorrectly
- ✅ Feature request: New functionality
- ✅ Documentation: Unclear or missing docs
- ✅ Performance: Unexpected slowness

**When NOT to open an issue**:
- ❌ Usage questions: Use discussions
- ❌ Provider issues: Contact provider support
- ❌ Hardware issues: Check your system
- ❌ Network issues: Check your connection

**Issue Template**:

```markdown
## Description
[Brief description of the issue]

## Steps to Reproduce
1. Run this command: `victor chat "Hello"`
2. See error: [error message]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- OS: [macOS, Linux, Windows]
- Python version: [3.10, 3.11, etc.]
- Victor version: [from `victor --version`]
- Provider: [anthropic, openai, ollama, etc.]

## Logs
[Paste relevant logs here]

## Additional Context
[Any other information]
```

---

## Getting Help

### Documentation

- **Full Docs**: [Documentation Home](/index.md)
- **User Guide**: [User Guide](/user-guide/index.md)
- **Reference**: [Provider Reference](/user-guide/providers.md)
- **Configuration**: [Configuration Options](/reference/configuration-options.md)

### Community

- **GitHub Issues**: [Report bugs](https://github.com/anvai-labs/victor/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/anvai-labs/victor/discussions)
- **Discord**: [Join Discord](https://discord.gg/...)
- **Stack Overflow**: Tag with `victor-ai`

### Diagnostic Commands

**Run diagnostics and share output**:
```bash
# Full diagnostic report
victor doctor

# Save to file
victor doctor > victor-diagnostic.txt

# Share in GitHub issue
```

---

## Quick Reference

### Common Commands

```bash
# Diagnostics
victor --version
victor providers
victor tools
victor config show
victor logs --tail 50

# Testing
victor chat "Hello" --provider ollama
victor chat "Hello" --provider anthropic

# Configuration
victor config profiles
victor config show

# Logs
victor logs --tail 100
victor logs --follow
victor logs --level ERROR
```

### Environment Variables

```bash
# API Keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export GOOGLE_API_KEY=...

# Local Providers
export OLLAMA_HOST=127.0.0.1:11434
export VICTOR_VLLM_HOST=127.0.0.1:8000

# Debug
export VICTOR_LOG_LEVEL=DEBUG
export VICTOR_LOG_FILE=/var/log/victor.log
```

### Configuration Files

```bash
~/.victor/profiles.yaml   # Provider/model profiles
~/.victor/config.yaml      # Global settings
~/.victor/mcp.yaml         # MCP server config
.victor.md                  # Project context
CLAUDE.md                   # AI instructions
```

---

**Still stuck?** [Open an issue →](https://github.com/anvai-labs/victor/issues/new)

**Next**: [User Guide](/user-guide/index.md) | [Provider Reference](/user-guide/providers.md) | [Configuration Options](/reference/configuration-options.md)
