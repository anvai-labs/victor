<div align="center">

![Victor Banner](../assets/victor-banner.svg)

# Quick Start

</div>

> For Docker, see `../../docker/README.md` and `../../docker/QUICKREF.md`.

## Prerequisites
- Python 3.11+
- Optional: Ollama for local models

## Install
```bash
pipx install victor-ai
# or
pip install victor-ai
```

Initialize once:
```bash
victor init
```

## Run
```bash
victor chat
# one-shot
victor "refactor this file for clarity"
```

## Local Model (No API Key)
```bash
ollama pull qwen2.5-coder:7b
victor chat --provider ollama --model qwen2.5-coder:7b
```

## Cloud Model
```bash
victor keys --set anthropic --keyring
victor chat --provider anthropic --model claude-sonnet-4-5
```

## Common Tasks
```bash
victor "review src/main.py for bugs"
victor "write tests for src/app.py"
victor "summarize this repo"
```

## Troubleshooting
- Connection refused: start Ollama with `ollama serve`
- Model not found: `ollama pull <model>`
- Import errors: reinstall with `pip install -e ./victor-contracts -e ".[dev]"` for local dev

## Next Steps
- User Guide: `../USER_GUIDE.md`
- Provider Setup: `PROVIDER_SETUP.md`
- Local Models: `LOCAL_MODELS.md`
- Examples: `../../examples/README.md`
