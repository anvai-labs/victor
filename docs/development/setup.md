# Development Environment Setup

This guide walks you through setting up a complete development environment for contributing to Victor.

## Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Minimum Version | Check Command |
|-------------|-----------------|---------------|
| Python | 3.11+ | `python --version` |
| pip | 21.0+ | `pip --version` |
| Git | 2.0+ | `git --version` |

### Optional Prerequisites

| Tool | Purpose | Installation |
|------|---------|--------------|
| Docker | Code execution sandbox | [docker.com/get-docker](https://docs.docker.com/get-docker/) |
| Ollama | Local LLM testing | [ollama.ai](https://ollama.ai) |
| pre-commit | Git hooks | `pip install pre-commit` |

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/vijayksingh/victor.git
cd victor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode (contracts first — victor-ai depends on the in-repo SDK)
pip install -e ./victor-contracts -e ".[dev]"

# Verify installation
victor --version
pytest tests/unit -v --tb=short
```

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vijayksingh/victor.git
cd victor
```

### 2. Create Virtual Environment

Using a virtual environment isolates Victor's dependencies from your system Python:

```bash
# Create virtual environment
python -m venv .venv

# Activate (choose your platform)
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows CMD
.venv\Scripts\Activate.ps1         # Windows PowerShell
```

### 3. Install Development Dependencies

The `[dev]` extra includes testing, linting, and formatting tools. Install the
in-repo `victor-contracts` in the same command so victor-ai resolves the local
SDK instead of the PyPI release (equivalently, run `make install-dev`):

```bash
pip install -e ./victor-contracts -e ".[dev]"
```

This installs:
- **pytest** (8.0+) - Testing framework
- **pytest-asyncio** (0.23+) - Async test support
- **pytest-cov** (4.1+) - Coverage reporting
- **pytest-mock** (3.12+) - Mocking utilities
- **respx** (0.21+) - HTTP mocking
- **black** (24.0+) - Code formatter
- **ruff** (0.5+) - Fast linter
- **mypy** (1.10+) - Type checker

### 4. Install Optional Dependencies

Depending on your development focus, you may need additional packages:

```bash
# Documentation
pip install -e ".[docs]"

# Google Gemini provider
pip install -e ".[google]"

# HTTP API server development
pip install -e ".[api]"

# Checkpoint persistence with SQLite
pip install -e ".[checkpoints]"

# All tree-sitter language grammars
pip install -e ".[lang-all]"

# Everything for full development
pip install -e ".[dev,docs,api,google,lang-all]"
```

### 5. Verify Installation

```bash
# Check CLI is available
victor --version

# Run unit tests
pytest tests/unit -v --tb=short

# Run linters
make lint
```

## Pre-commit Hooks Setup

Pre-commit hooks automatically format and lint code before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually (optional)
pre-commit run --all-files
```

The hooks will run automatically on `git commit` and check:
- Black formatting
- Ruff linting
- Type hints with mypy

## IDE Setup

### VS Code (Recommended)

Create `.vscode/settings.json` in the project root:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,

    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    },

    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.rulers": [100]
    },

    "black-formatter.args": ["--line-length", "100"],
    "ruff.lint.args": ["--config", "pyproject.toml"],

    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/unit",
        "-v"
    ],

    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        ".ruff_cache": true,
        "htmlcov": true,
        "*.egg-info": true
    }
}
```

#### Recommended VS Code Extensions

Install these extensions for the best development experience:

- **Python** (`ms-python.python`) - Python language support
- **Pylance** (`ms-python.vscode-pylance`) - Type checking and IntelliSense
- **Black Formatter** (`ms-python.black-formatter`) - Code formatting
- **Ruff** (`charliermarsh.ruff`) - Fast linting
- **Python Test Explorer** (`littlefoxteam.vscode-python-test-adapter`) - Test UI

#### Launch Configurations

Create `.vscode/launch.json` for debugging:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Victor CLI",
            "type": "debugpy",
            "request": "launch",
            "module": "victor.ui.cli",
            "console": "integratedTerminal"
        },
        {
            "name": "Victor Chat",
            "type": "debugpy",
            "request": "launch",
            "module": "victor.ui.cli",
            "args": ["chat"],
            "console": "integratedTerminal"
        },
        {
            "name": "Current Test File",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v", "--tb=short"],
            "console": "integratedTerminal"
        }
    ]
}
```

### PyCharm

1. Open the project folder in PyCharm
2. Go to **File > Settings > Project > Python Interpreter**
3. Add the `.venv` interpreter
4. Go to **File > Settings > Tools > Black** and enable "On save"
5. Configure pytest as the test runner in **File > Settings > Tools > Python Integrated Tools**

## Environment Variables

Victor uses environment variables for API keys and configuration. Create a `.env` file in the project root (this file is gitignored):

```bash
# .env - Development environment variables
# NEVER commit this file

# LLM Provider API Keys (add the ones you use)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Optional: Skip .env loading in tests (auto-set by fixtures)
# VICTOR_SKIP_ENV_FILE=1
```

For testing, the `isolate_environment_variables` fixture automatically clears API keys to ensure tests are deterministic.

## Common Setup Issues

### ModuleNotFoundError: No module named 'victor'

Ensure you installed in editable mode (with the in-repo contracts):
```bash
pip install -e ./victor-contracts -e ".[dev]"
```

### Permission denied on `.venv`

On macOS/Linux, ensure proper permissions:
```bash
chmod -R u+rwX .venv
```

### mypy finds no typed modules

MyPy needs to be run from the project root:
```bash
cd /path/to/victor
mypy victor
```

### Tests fail with Docker errors

Many tests mock Docker. If you see Docker-related failures, ensure the `mock_code_execution_manager` fixture is applied:
```python
def test_my_feature(mock_code_execution_manager):
    # Test code here
```

### Pre-commit hooks fail

Update hooks to latest versions:
```bash
pre-commit autoupdate
pre-commit clean
pre-commit install
```

## Project Structure Overview

After setup, familiarize yourself with the project structure:

```
victor/
├── victor/                 # Main source code
│   ├── agent/              # Orchestration, tool pipeline
│   ├── providers/          # LLM providers (BaseProvider)
│   ├── tools/              # Tools (BaseTool, CostTier)
│   ├── framework/          # StateGraph DSL, Agent/Task API
│   ├── workflows/          # YAML execution, scheduling
│   ├── coding/             # AST, LSP, code review
│   ├── devops/             # Docker, Terraform, CI
│   ├── rag/                # Embeddings, retrieval
│   ├── dataanalysis/       # Pandas, statistics
│   ├── research/           # Web search, synthesis
│   └── config/             # Settings, model capabilities
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
└── CLAUDE.md               # AI assistant context
```

## Daily Development Commands

```bash
# Run tests
make test                              # Unit tests
make test-all                          # All tests including integration
pytest tests/unit/test_X.py::test_name # Single test

# Code quality
make format                            # Format code (black + ruff --fix)
make lint                              # Check formatting (ruff + black --check + mypy)

# Run Victor
victor                                 # Start interactive CLI
victor chat                            # CLI mode
victor chat --provider anthropic       # Specific provider
```

## Next Steps

- [Testing Guide](testing.md) - Learn testing patterns and fixtures
- [Code Style Guide](code-style.md) - Formatting and linting standards
- [Architecture Overview](../architecture.md) - System design
- [Contributing Guide](../../CONTRIBUTING.md) - Pull request process

---

**Need help?** Open an issue on [GitHub](https://github.com/vijayksingh/victor/issues) or start a [discussion](https://github.com/vijayksingh/victor/discussions).
