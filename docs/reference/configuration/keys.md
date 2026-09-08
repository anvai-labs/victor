# Victor API Keys Configuration Guide

This guide explains how to configure API keys for LLM providers and external data services in Victor.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Security Best Practices](#security-best-practices)
- [Key Storage Options](#key-storage-options)
- [LLM Provider Keys](#llm-provider-keys)
- [External Service Keys](#external-service-keys)
- [CLI Commands Reference](#cli-commands-reference)
- [Programmatic Access](#programmatic-access)
- [Troubleshooting](#troubleshooting)

## Overview

Victor uses a secure, multi-layered approach to API key management:

```
Priority Order (highest to lowest):
┌─────────────────────────────────────────────────────────────┐
│ 1. Environment Variables  (for CI/CD, containers)          │
├─────────────────────────────────────────────────────────────┤
│ 2. System Keyring         (encrypted OS storage)           │
│    - macOS: Keychain                                        │
│    - Windows: Credential Manager                            │
│    - Linux: Secret Service (GNOME Keyring/KWallet)         │
├─────────────────────────────────────────────────────────────┤
│ 3. Keys File              (~/.victor/api_keys.yaml)        │
│    - Permissions: 0600 (owner read/write only)              │
│    - Not recommended for production                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Set up a provider key (recommended: keyring)

```bash
# Store Anthropic API key in system keyring (most secure)
victor keys --set anthropic --keyring

# Store OpenAI API key
victor keys --set openai --keyring
```

### 2. Set up a service key

```bash
# Store Finnhub API key for market data
victor keys --set-service finnhub --keyring

# Store FRED API key for economic data
victor keys --set-service fred --keyring
```

### 3. Verify configuration

```bash
# List all configured provider keys
victor keys --list

# List all configured service keys
victor keys --services
```

## Security Best Practices

### DO

- **Use system keyring** for persistent storage (`--keyring` flag)
- **Use environment variables** for CI/CD and containers
- **Audit key access** - Victor logs all key access attempts
- **Rotate keys regularly** - Update keys periodically
- **Use least privilege** - Only configure keys you need

### DON'T

- **Don't commit keys to git** - Never put keys in code repositories
- **Don't share keys** - Each developer should have their own keys
- **Don't use file storage in production** - Use keyring or env vars instead
- **Don't log keys** - Victor never logs actual key values

## Key Storage Options

### Option 1: System Keyring (Recommended)

The most secure option for persistent storage.

```bash
# Set provider key in keyring
victor keys --set anthropic --keyring

# Set service key in keyring
victor keys --set-service finnhub --keyring

# Delete from keyring
victor keys --delete-keyring anthropic
victor keys --delete-service-keyring finnhub
```

**Platform Support:**

| Platform | Backend | Notes |
|----------|---------|-------|
| macOS | Keychain | Built-in, no setup required |
| Windows | Credential Manager | Built-in, no setup required |
| Linux | Secret Service | Requires `gnome-keyring` or `kwallet` |

### Option 2: Environment Variables

Best for CI/CD pipelines, Docker containers, and automation.

```bash
# Set in shell profile (~/.bashrc, ~/.zshrc)
export ANTHROPIC_API_KEY="sk-ant-..."
export FINNHUB_API_KEY="your-key-here"
export FRED_API_KEY="your-key-here"

# Or use a .env file (not committed to git)
# .env
ANTHROPIC_API_KEY=sk-ant-...
FINNHUB_API_KEY=your-key-here
```

### Option 3: Keys File (Fallback)

For development only. Not recommended for production.

```bash
# Create template file
victor keys --setup

# Edit the file
vim ~/.victor/api_keys.yaml
```

```yaml
# ~/.victor/api_keys.yaml
api_keys:
  anthropic: "sk-ant-..."
  openai: "sk-..."

services:
  finnhub: "your-finnhub-key"
  fred: "your-fred-key"
```

**Important:** File must have `0600` permissions (owner read/write only).

### Option 4: Migrate from File to Keyring

If you have keys in a file and want to move them to keyring:

```bash
# Migrate all keys from file to keyring
victor keys --migrate

# Then delete the file
rm ~/.victor/api_keys.yaml
```

## LLM Provider Keys

### Premium Providers

| Provider | Env Variable | Registration |
|----------|--------------|--------------|
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Google | `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| xAI (Grok) | `XAI_API_KEY` | [console.x.ai](https://console.x.ai/) |
| DeepSeek | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/) |

### Free-Tier Providers

These providers offer generous free tiers:

| Provider | Env Variable | Free Limits | Registration |
|----------|--------------|-------------|--------------|
| Groq | `GROQCLOUD_API_KEY` | 14,400 req/day | [console.groq.com](https://console.groq.com/) |
| Cerebras | `CEREBRAS_API_KEY` | Free tier | [cloud.cerebras.ai](https://cloud.cerebras.ai/) |
| Mistral | `MISTRAL_API_KEY` | 500K tokens/min | [console.mistral.ai](https://console.mistral.ai/) |
| Together | `TOGETHER_API_KEY` | $25 credits | [api.together.xyz](https://api.together.xyz/) |
| OpenRouter | `OPENROUTER_API_KEY` | Daily limits | [openrouter.ai](https://openrouter.ai/keys) |
| Google | `GOOGLE_API_KEY` | 15 RPM, 1M TPM | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

### Local Providers (No Key Required)

These run locally and don't need API keys:

- **Ollama**: `ollama serve`
- **LM Studio**: Desktop app with OpenAI-compatible API
- **vLLM**: `python -m vllm.entrypoints.openai.api_server`

## External Service Keys

### Market Data Services

| Service | Env Variable | Description | Free Tier |
|---------|--------------|-------------|-----------|
| Finnhub | `FINNHUB_API_KEY` | Stock data, sentiment, estimates | 60 calls/min |
| FRED | `FRED_API_KEY` | Federal Reserve economic data | Unlimited |
| Alpha Vantage | `ALPHAVANTAGE_API_KEY` | Stock/forex/crypto | 25 req/day |
| Polygon | `POLYGON_API_KEY` | Real-time market data | 5 calls/min |
| Tiingo | `TIINGO_API_KEY` | Historical EOD data | 500 req/hour |
| IEX Cloud | `IEX_API_KEY` | US equity data | 50K msg/month |

### Setting Up Finnhub (Example)

1. Register at [finnhub.io/register](https://finnhub.io/register)
2. Get your API key from the dashboard
3. Store it securely:

```bash
# Using keyring (recommended)
victor keys --set-service finnhub --keyring

# Or via environment variable
export FINNHUB_API_KEY="your-key-here"
```

### Setting Up FRED (Example)

1. Register at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Get your API key
3. Store it:

```bash
victor keys --set-service fred --keyring
```

## CLI Commands Reference

### Provider Keys

```bash
# List all provider keys and their status
victor keys --list

# Set a provider key (interactive, stored in file)
victor keys --set <provider>

# Set a provider key in system keyring (secure)
victor keys --set <provider> --keyring

# Delete provider key from keyring
victor keys --delete-keyring <provider>

# Create template file at ~/.victor/api_keys.yaml
victor keys --setup

# Migrate all keys from file to keyring
victor keys --migrate
```

### Service Keys

```bash
# List all service keys and their status
victor keys --services

# Set a service key (interactive, stored in file)
victor keys --set-service <service>

# Set a service key in system keyring (secure)
victor keys --set-service <service> --keyring

# Delete service key from keyring
victor keys --delete-service-keyring <service>
```

### Valid Provider Names

```
anthropic, openai, google, xai, grok, moonshot, kimi, deepseek,
groqcloud, cerebras, mistral, together, openrouter, fireworks,
vertex, vertexai, azure, azure-openai, bedrock, aws,
huggingface, hf, replicate
```

### Valid Service Names

```
finnhub, fred, alphavantage, polygon, tiingo, iex, quandl, nasdaq,
newsapi, marketaux, sec, openweather, geocoding
```

## Programmatic Access

### Getting Keys in Python

```python
from victor.config.api_keys import get_api_key, get_service_key

# Get LLM provider key
anthropic_key = get_api_key("anthropic")
if anthropic_key:
    # Use the key
    pass

# Get external service key
finnhub_key = get_service_key("finnhub")
if finnhub_key:
    # Use for market data API calls
    pass

# Check if keyring is available
from victor.config.api_keys import is_keyring_available
if is_keyring_available():
    print("System keyring is available")
```

### Setting Keys Programmatically

```python
from victor.config.api_keys import set_api_key, set_service_key

# Set provider key in keyring
set_api_key("anthropic", "sk-ant-...", use_keyring=True)

# Set service key in keyring
set_service_key("finnhub", "your-key", use_keyring=True)
```

### Using in Data Collectors

```python
# In scripts/scheduled/collect_dividends.py
from victor.config.api_keys import get_service_key

class DividendCollector:
    def __init__(self):
        self.api_key = get_service_key("finnhub")
        if not self.api_key:
            raise ValueError(
                "FINNHUB_API_KEY not configured. "
                "Run: victor keys --set-service finnhub --keyring"
            )
```

## Troubleshooting

### "Keyring not available"

Install the keyring package:

```bash
pip install keyring

# On Linux, also install a backend
sudo apt install gnome-keyring  # Ubuntu/Debian
sudo dnf install gnome-keyring  # Fedora
```

### "Permission denied" for keys file

Fix file permissions:

```bash
chmod 600 ~/.victor/api_keys.yaml
```

### Key not found but I set it

Check the resolution order:

1. Is the environment variable set? (`echo $ANTHROPIC_API_KEY`)
2. Is it in keyring? (`victor keys --list` shows "keyring" source)
3. Is it in the file? (Check `~/.victor/api_keys.yaml`)

### Audit logging

Victor logs all key access attempts. Check logs for:

```
SECRET_ACCESS: action=loaded provider=anthropic source=keyring success=True
```

## Configuration File Reference

The external registry is at `victor/config/api_keys_registry.yaml` and defines:

- All supported providers and services
- Environment variable mappings
- Registration URLs
- Free tier limits
- Documentation links

This file is the single source of truth for what keys are supported.

## See Also

- [Configuration Reference](index.md)
- [Security Policy](https://github.com/anvai-labs/victor/blob/develop/SECURITY.md)
- [API Keys Registry](https://github.com/anvai-labs/victor/blob/develop/victor/config/api_keys_registry.yaml)
