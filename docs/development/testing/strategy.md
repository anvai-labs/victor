# Testing Strategy for Victor

## Current State

**Test Coverage**: 0% (as of 2025-11-24)
**Total Statements**: ~3,362
**Existing Tests**: 3 files (unit, integration tests for Ollama)

## Testing Goals

### Phase 1: Foundation (Target: 40% coverage)
- [ ] Core provider abstraction
- [ ] Tool registry and base tool
- [ ] Configuration management
- [ ] Basic file operations

### Phase 2: Features (Target: 70% coverage)
- [ ] MCP protocol (server, client, protocol)
- [ ] Advanced tools (database, Docker, HTTP)
- [ ] Multi-file editor with transactions
- [ ] Git tool with AI commits
- [ ] Web search tool
- [ ] Semantic search and indexing

### Phase 3: Integration (Target: 85% coverage)
- [ ] Provider integrations
- [ ] Agent orchestrator
- [ ] Context management
- [ ] UI components
- [ ] End-to-end workflows

## Test Structure

```
tests/
├── unit/                      # Unit tests for individual components
│   ├── providers/            # Provider tests
│   │   ├── test_base.py
│   │   ├── test_ollama.py
│   │   ├── test_anthropic.py
│   │   ├── test_openai.py
│   │   └── test_google.py
│   ├── tools/                # Tool tests
│   │   ├── test_base.py
│   │   ├── test_bash.py
│   │   ├── test_filesystem.py
│   │   ├── test_file_editor.py
│   │   ├── test_git_tool.py
│   │   ├── test_database_tool.py
│   │   ├── test_docker_tool.py
│   │   ├── test_http_tool.py
│   │   └── test_web_search.py
│   ├── mcp/                  # MCP tests
│   │   ├── test_protocol.py
│   │   ├── test_server.py
│   │   └── test_client.py
│   ├── editing/              # Multi-file editor tests
│   │   └── test_editor.py
│   ├── codebase/             # Semantic search tests
│   │   ├── test_indexer.py
│   │   └── test_embeddings.py
│   └── context/              # Context management tests
│       └── test_manager.py
├── integration/              # Integration tests
│   ├── test_ollama_integration.py
│   ├── test_provider_switching.py
│   ├── test_mcp_integration.py
│   └── test_tool_chaining.py
└── e2e/                      # End-to-end tests
    ├── test_coding_workflow.py
    ├── test_git_workflow.py
    └── test_multi_file_workflow.py
```

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual components in isolation

**Example: Database Tool**
```python
import pytest
from victor.tools.database_tool import DatabaseTool


class TestDatabaseTool:
    """Unit tests for DatabaseTool."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return DatabaseTool(allow_modifications=True)
    
    async def test_connect_sqlite(self, tool, tmp_path):
        """Test SQLite connection."""
        db_path = tmp_path / "test.db"
        result = await tool.execute(
            operation="connect",
            db_type="sqlite",
            database=str(db_path)
        )
        assert result.success
        assert "Connection ID:" in result.output
    
    async def test_dangerous_query_blocked(self, tool):
        """Test that dangerous queries are blocked."""
        result = await tool.execute(
            operation="query",
            connection_id="test",
            sql="DROP TABLE users"
        )
        assert not result.success
        assert "not allowed" in result.error
```

**Example: MCP Protocol**
```python
import pytest
from victor.integrations.mcp.protocol import MCPMessage, MCPMessageType


class TestMCPProtocol:
    """Unit tests for MCP protocol."""
    
    def test_message_creation(self):
        """Test MCP message creation."""
        msg = MCPMessage(
            id="123",
            method=MCPMessageType.LIST_TOOLS,
            params={}
        )
        assert msg.jsonrpc == "2.0"
        assert msg.id == "123"
        assert msg.method == MCPMessageType.LIST_TOOLS
    
    def test_message_serialization(self):
        """Test message serialization."""
        msg = MCPMessage(id="123", method=MCPMessageType.INITIALIZE)
        data = msg.model_dump(exclude_none=True)
        assert "jsonrpc" in data
        assert "id" in data
```

### 2. Integration Tests

**Purpose**: Test component interactions

**Example: Provider Tool Integration**
```python
import pytest
from victor.providers.ollama_provider import OllamaProvider
from victor.agent.orchestrator import AgentOrchestrator


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test agent using tools."""
    provider = OllamaProvider()
    agent = AgentOrchestrator(
        provider=provider,
        model="qwen2.5-coder:7b"
    )
    
    response = await agent.chat(
        "Read the file README.md and tell me the project name"
    )
    
    assert response.success
    assert "Victor" in response.content
    await provider.close()
```

### 3. End-to-End Tests

**Purpose**: Test complete workflows

**Example: Multi-File Editing Workflow**
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_file_editing_workflow(tmp_path):
    """Test complete multi-file editing workflow."""
    # 1. Create test files
    # 2. Initialize multi-file editor
    # 3. Make edits across multiple files
    # 4. Verify atomic commit
    # 5. Test rollback on error
    pass
```

## Testing Best Practices

### 1. Fixtures for Common Setup

```python
@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.py").write_text("# Test file")
    return workspace


@pytest.fixture
async def ollama_provider():
    """Create Ollama provider."""
    provider = OllamaProvider()
    yield provider
    await provider.close()
```

### 2. Mocking External Services

```python
@pytest.fixture
def mock_ollama_api(respx_mock):
    """Mock Ollama API responses."""
    respx_mock.post("http://localhost:11434/api/chat").mock(
        return_value={
            "message": {"content": "Mocked response"}
        }
    )
    return respx_mock
```

### 3. Parametrized Tests

```python
@pytest.mark.parametrize("db_type,expected", [
    ("sqlite", "SQLite"),
    ("postgresql", "PostgreSQL"),
    ("mysql", "MySQL"),
])
async def test_database_connection(db_type, expected):
    """Test different database types."""
    # Test implementation
    pass
```

### 4. Async Testing

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation."""
    result = await some_async_function()
    assert result is not None
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ./victor-contracts -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest --cov=victor --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

## Test Coverage Reporting

### Local Coverage

```bash
# Generate HTML coverage report
pytest --cov=victor --cov-report=html

# Open in browser
open htmlcov/index.html
```

### Coverage Badges

Add to README.md:
```markdown
[![Coverage](https://codecov.io/gh/vjsingh1984/victor/branch/main/graph/badge.svg)](https://codecov.io/gh/vjsingh1984/victor)
```

## Testing Priorities

### High Priority (Week 1)
1. Tool tests (database, Docker, HTTP)
2. MCP protocol tests
3. Multi-file editor tests
4. Provider base tests

### Medium Priority (Week 2)
5. Git tool tests
6. Web search tests
7. Context manager tests
8. Integration tests

### Lower Priority (Week 3)
9. UI component tests
10. E2E workflow tests
11. Performance tests
12. Load tests

## Test Requirements

### Python Version
- Python 3.11+ required
- Use `pyenv` or `conda` for version management

### Dependencies
```bash
pip install -e ./victor-contracts -e ".[dev]"  # Includes pytest, pytest-asyncio, pytest-cov
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/ -m integration

# With coverage
pytest --cov=victor --cov-report=term-missing

# Specific test
pytest tests/unit/tools/test_database_tool.py::TestDatabaseTool::test_connect_sqlite

# Parallel execution
pytest -n auto
```

## Success Metrics

- **Phase 1**: 40% coverage target, core functionality tested
- **Phase 2**: 70% coverage target, key features tested
- **Phase 3**: 85%+ coverage target, broader test suite
- **CI/CD**: Automated testing on PRs (when configured)
- **Quality**: Reduce critical bugs in tested code
- **Performance**: Aim for fast test runs on CI

## Next Steps

1. Set up a Python 3.11+ environment
2. Install Victor in editable mode
3. Create test directory structure
4. Implement Phase 1 tests (core functionality)
5. Set up CI/CD with GitHub Actions
6. Add coverage badges to README
7. Implement Phase 2 tests (features)
8. Implement Phase 3 tests (integration)
9. Achieve 85%+ coverage target

---

*Last Updated: 2025-11-24*
*Current Coverage: 0%*
*Target Coverage: 85%+*
