# Testing Guide

This guide covers Victor's testing infrastructure, patterns, and best practices for writing effective tests.

## Test Structure

Victor's tests are organized into two main categories:

```
tests/
├── unit/                      # Fast, isolated unit tests
│   ├── providers/             # Provider tests
│   ├── tools/                 # Tool tests
│   ├── mcp/                   # MCP protocol tests
│   ├── workflows/             # Workflow tests
│   ├── agent/                 # Orchestrator tests
│   └── ...
├── integration/               # Tests requiring external services
│   ├── test_ollama.py         # Ollama integration
│   ├── test_provider_switching.py
│   └── ...
└── conftest.py                # Shared fixtures
```

## Running Tests

### Quick Commands

```bash
# Run unit tests (recommended for development)
make test

# Run all tests including integration
make test-all

# Run with coverage report
make test-cov
```

### pytest Commands

```bash
# Unit tests only
pytest tests/unit -v

# Single test file
pytest tests/unit/test_orchestrator.py -v

# Single test function
pytest tests/unit/test_orchestrator.py::test_basic_chat -v

# Run tests matching a pattern
pytest tests/unit -k "tool" -v

# Skip slow tests
pytest -m "not slow" -v

# Run only integration tests
pytest -m integration -v

# Run only workflow tests
pytest -m workflows -v

# Parallel execution (requires pytest-xdist)
pytest -n auto tests/unit
```

### Coverage Reports

```bash
# Terminal coverage report
pytest tests/unit --cov=victor --cov-report=term-missing

# HTML coverage report
pytest tests/unit --cov=victor --cov-report=html
open htmlcov/index.html

# XML for CI tools
pytest tests/unit --cov=victor --cov-report=xml
```

## Test Markers

Victor uses pytest markers to categorize tests. Markers are defined in `pyproject.toml`:

| Marker | Purpose | Command |
|--------|---------|---------|
| `@pytest.mark.unit` | Fast, isolated tests | `pytest -m unit` |
| `@pytest.mark.integration` | Requires external services | `pytest -m integration` |
| `@pytest.mark.slow` | Long-running tests | `pytest -m "not slow"` (skip) |
| `@pytest.mark.workflows` | Workflow-related tests | `pytest -m workflows` |
| `@pytest.mark.agents` | Multi-agent tests | `pytest -m agents` |
| `@pytest.mark.hitl` | Human-in-the-loop tests | `pytest -m hitl` |

### Using Markers

```python
import pytest

@pytest.mark.unit
def test_fast_operation():
    """Quick unit test."""
    assert 1 + 1 == 2

@pytest.mark.integration
@pytest.mark.slow
async def test_external_api():
    """Test that requires external service."""
    # This test is skipped in normal runs
    pass

@pytest.mark.workflows
def test_workflow_execution():
    """Workflow-specific test."""
    pass
```

## Key Fixtures

Victor provides several fixtures in `tests/conftest.py` that handle common testing needs.

### Autouse fixtures (applied to every test)

`tests/conftest.py` applies these automatically to every test to guarantee isolation:

- `reset_api_keys_logger` — caplog propagation for the API-keys logger
- `reset_embedding_singleton` — prevents a stale embedding model leaking across tests
- `reset_global_state_manager` — resets the global state-manager singleton
- `isolate_environment_variables` — no real API keys (`get_api_key()` → `None`); skips `.env`
- `reset_service_container` — resets the DI container between tests
- `isolate_conversation_database` — temp DB path so tests never touch the production conversation DB

Because they are autouse, you don't request them by name — they run for you. (Verify the
current list against `tests/conftest.py`, which is the source of truth.)

### isolate_environment_variables (Auto-applied)

Prevents tests from loading actual API keys:

```python
def test_without_real_keys(isolate_environment_variables):
    # ANTHROPIC_API_KEY, OPENAI_API_KEY, etc. are cleared
    # VICTOR_SKIP_ENV_FILE is set to "1"
    # Tests won't accidentally use real credentials
    pass
```

This fixture:
- Sets `VICTOR_SKIP_ENV_FILE=1` to skip `.env` loading
- Clears all API key environment variables
- Mocks `get_api_key()` to return `None`

### Docker Mocking

Prevents Docker startup in unit tests:

```python
def test_with_mock_docker(mock_code_execution_manager):
    """Test without actually starting Docker."""
    # CodeSandbox is mocked
    # mock_instance.docker_available = False
    pass

def test_with_mock_client(mock_docker_client):
    """Test with mocked Docker client."""
    # docker.from_env() returns a mock
    pass
```

The `auto_mock_docker_for_orchestrator` fixture automatically mocks Docker for tests that:
- Have "orchestrator" in the test name
- Are in specific test paths (tool_selection, integration, etc.)

### Workflow Fixtures

```python
def test_empty_workflow(empty_workflow_graph):
    """Test with empty workflow graph."""
    # WorkflowGraph with TestState
    empty_workflow_graph.add_node("start", handler)
    pass

def test_linear_flow(linear_workflow_graph):
    """Test with A -> B -> C workflow."""
    # Pre-configured linear workflow
    # Nodes: a, b, c with handlers
    pass

def test_branching(branching_workflow_graph):
    """Test with conditional branching."""
    # Nodes: start, branch_a, branch_b, merge
    # Conditional routing based on state.branch
    pass
```

### HITL Fixtures

```python
def test_hitl_execution(hitl_executor):
    """Test with HITL executor."""
    # HITLExecutor with DefaultHITLHandler
    pass

async def test_auto_approval(auto_approve_handler):
    """Test with auto-approving handler."""
    response = await auto_approve_handler(request)
    assert response.approved is True

async def test_auto_rejection(auto_reject_handler):
    """Test with auto-rejecting handler."""
    response = await auto_reject_handler(request)
    assert response.approved is False

def test_approval_node(hitl_approval_node):
    """Test with pre-configured approval node."""
    # HITLNode with APPROVAL type, 5s timeout, ABORT fallback
    pass
```

### Multi-Agent Fixtures

```python
def test_team_member(mock_team_member):
    """Test with mock team member."""
    assert mock_team_member.role == SubAgentRole.EXECUTOR
    assert mock_team_member.tool_budget == 15

def test_team_specs(team_member_specs):
    """Test with team member specifications."""
    # List of TeamMemberSpec: researcher, executor, reviewer
    pass

async def test_coordination(mock_team_coordinator):
    """Test with mock coordinator."""
    result = await mock_team_coordinator.execute_team("task")
    assert result.success is True
```

### Mode Config Fixtures

```python
def test_mode(default_mode_config):
    """Test with default mode."""
    assert default_mode_config.tool_budget == 20
    assert default_mode_config.max_iterations == 40

def test_registry(mode_config_registry):
    """Test with fresh registry (not singleton)."""
    pass

def test_registered(registered_mode_registry):
    """Test with pre-registered verticals."""
    # Has "test_vertical" with "custom" mode
    pass
```

## Mocking Patterns

### HTTP Mocking with respx

Use `respx` for mocking HTTP calls in async tests:

```python
import pytest
import httpx
import respx

@pytest.mark.asyncio
@respx.mock
async def test_api_call():
    """Test with mocked HTTP."""
    # Setup mock
    mock_response = {"status": "ok", "data": [1, 2, 3]}
    route = respx.post("https://api.example.com/v1/endpoint").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Make the call
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.example.com/v1/endpoint")

    # Verify
    assert route.called
    assert response.json() == mock_response
```

### Mocking Provider Responses

```python
@pytest.mark.asyncio
@respx.mock
async def test_anthropic_chat():
    """Mock Anthropic API response."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })
    )

    # Test your code that calls Anthropic
    pass
```

### Mocking with unittest.mock

```python
from unittest.mock import MagicMock, AsyncMock, patch

def test_with_mock():
    """Test with patched function."""
    with patch("victor.tools.some_tool.external_call") as mock_call:
        mock_call.return_value = {"result": "mocked"}
        # Test code
        assert mock_call.called

@pytest.mark.asyncio
async def test_async_mock():
    """Test with async mock."""
    with patch("victor.providers.some_provider.fetch") as mock_fetch:
        mock_fetch.return_value = AsyncMock(return_value={"data": []})
        # Test async code
        pass
```

### Ollama Tests

Use the `requires_ollama` decorator for tests that need Ollama:

```python
from tests.conftest import requires_ollama, is_ollama_available

@requires_ollama()
async def test_ollama_integration():
    """Test requires running Ollama server."""
    # This test is skipped if Ollama is not available at localhost:11434
    pass

# Or use skipif directly
@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
async def test_ollama_feature():
    pass
```

## Writing Effective Tests

### Unit Test Structure

Follow the Arrange-Act-Assert pattern:

```python
@pytest.mark.asyncio
async def test_tool_execution():
    """Test tool executes correctly with valid input."""
    # Arrange: Set up test data and dependencies
    tool = MyTool()
    input_data = {"query": "test query"}

    # Act: Execute the code under test
    result = await tool.execute(**input_data)

    # Assert: Verify the results
    assert result["success"] is True
    assert "output" in result
    assert result["output"] == "expected output"
```

### Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("input_value,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("Victor", "VICTOR"),
    ("", ""),
])
def test_uppercase(input_value, expected):
    """Test uppercase transformation."""
    assert input_value.upper() == expected

@pytest.mark.parametrize("provider,model", [
    ("anthropic", "claude-sonnet-4-20250514"),
    ("openai", "gpt-4"),
    ("ollama", "llama2"),
])
async def test_provider_support(provider, model):
    """Test different provider/model combinations."""
    pass
```

### Testing Async Code

All async tests require the `pytest.mark.asyncio` marker:

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async function."""
    result = await some_async_function()
    assert result is not None

@pytest.mark.asyncio
async def test_async_generator():
    """Test async generator."""
    results = []
    async for item in async_generator():
        results.append(item)
    assert len(results) > 0
```

### Testing Exceptions

```python
def test_raises_on_invalid_input():
    """Test that invalid input raises exception."""
    with pytest.raises(ValueError, match="invalid input"):
        validate_input(None)

@pytest.mark.asyncio
async def test_async_exception():
    """Test async exception handling."""
    with pytest.raises(TimeoutError):
        await slow_operation(timeout=0.001)
```

### Using Temporary Files

```python
def test_file_operations(tmp_path):
    """Test with temporary directory."""
    # tmp_path is a pathlib.Path to a temp directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Test your file-handling code
    result = read_file(test_file)
    assert result == "test content"

def test_workspace(tmp_path):
    """Test with temporary workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create test files
    (workspace / "main.py").write_text("print('hello')")
    (workspace / "test.py").write_text("def test(): pass")

    # Test code that operates on directories
    pass
```

## Coverage Requirements

Victor aims for comprehensive test coverage:

| Code Type | Minimum Coverage | Target Coverage |
|-----------|------------------|-----------------|
| New code | 80% | 90%+ |
| Critical paths | 90% | 100% |
| Bug fixes | 100% (for fix) | 100% |

### Viewing Coverage

```bash
# Generate and open HTML report
make test-cov
open htmlcov/index.html
```

### Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["victor"]
omit = [
    "*/tests/*",
    "*/__init__.py",
    "victor/ui/cli.py",      # Requires interactive terminal
    "victor/ui/commands.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

## Debugging Tests

### Running with Debug Output

```bash
# Show print statements and logging
pytest tests/unit/test_specific.py -v -s

# Show full traceback
pytest tests/unit/test_specific.py -v --tb=long

# Stop on first failure
pytest tests/unit -x

# Enter debugger on failure
pytest tests/unit --pdb
```

### VS Code Debugging

Use the launch configuration from setup.md to debug tests:

1. Open the test file
2. Set breakpoints
3. Run "Current Test File" configuration

### Common Issues

**Test pollution**: If tests pass individually but fail together, check for:
- Singleton state not being reset
- Global variables being modified
- File system side effects

**Async warnings**: Ensure all async tests have `@pytest.mark.asyncio`

**Timeout errors**: Long-running tests should use `@pytest.mark.slow`

## CI/CD Integration

Tests run automatically on pull requests via GitHub Actions:

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ./victor-contracts -e ".[dev]"
      - run: pytest --cov=victor --cov-report=xml
```

## Next Steps

- [Code Style Guide](code-style.md) - Formatting and linting standards
- [Setup Guide](setup.md) - Development environment setup
- [Contributing Guide](https://github.com/anvai-labs/victor/blob/develop/CONTRIBUTING.md) - Pull request process

---

**Questions?** Check [existing tests](https://github.com/vijayksingh/victor/tree/main/tests) for examples or open a [discussion](https://github.com/vijayksingh/victor/discussions).
