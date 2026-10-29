import pytest
from unittest.mock import patch, AsyncMock
from victor.tools.unified.code_tool import code_tool


@pytest.mark.asyncio
async def test_code_tool_test():
    """Test `code test` subcommand."""
    with patch("victor.tools.unified.code_tool.run_tests", new_callable=AsyncMock) as mock_test:
        mock_test.return_value = {"passed": 5, "failed": 0, "output": "5 passed"}

        result = await code_tool("code test pytest tests/")

        mock_test.assert_called_once_with(runner="pytest", path="tests/")
        assert "5 passed" in result


@pytest.mark.asyncio
async def test_code_tool_execute():
    """Test `code execute` subcommand."""
    with patch(
        "victor.tools.unified.code_tool.execute_python", new_callable=AsyncMock
    ) as mock_exec:
        mock_exec.return_value = "hello world\n"

        result = await code_tool("code execute \"print('hello world')\"")

        mock_exec.assert_called_once_with("print('hello world')")
        assert "hello world" in result


@pytest.mark.asyncio
async def test_code_tool_rejects_shell_operators():
    """`code` is not a shell — a pipe returns the actionable rejection, not a
    cryptic parse error."""
    result = await code_tool("code grep foo src | grep bar")
    assert "SHELL OPERATOR NOT SUPPORTED" in result
    assert "`shell` tool" in result


@pytest.mark.asyncio
async def test_code_tool_pipe_inside_code_string_not_flagged():
    """A `|` inside the code argument is content, not a shell operator."""
    with patch(
        "victor.tools.unified.code_tool.execute_python", new_callable=AsyncMock
    ) as mock_exec:
        mock_exec.return_value = "ok"
        await code_tool('code python "x = a | b"')
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_code_tool_python_alias():
    """`code python` should be the shell-style replacement for ad hoc Python execution."""
    with patch(
        "victor.tools.unified.code_tool.execute_python", new_callable=AsyncMock
    ) as mock_exec:
        mock_exec.return_value = "hello alias\n"

        result = await code_tool("code python \"print('hello alias')\"")

        mock_exec.assert_called_once_with("print('hello alias')")
        assert "hello alias" in result


@pytest.mark.asyncio
async def test_code_tool_python_heredoc():
    """`code python` should accept heredoc code without quote escaping."""
    code = '"""module docstring"""\nprint("hello heredoc")'
    with patch(
        "victor.tools.unified.code_tool.execute_python", new_callable=AsyncMock
    ) as mock_exec:
        mock_exec.return_value = "hello heredoc\n"

        result = await code_tool("code python <<'PY'\n" + code + "\nPY")

        mock_exec.assert_called_once_with(code)
        assert "hello heredoc" in result


@pytest.mark.asyncio
async def test_code_tool_metrics():
    """Test `code metrics` subcommand formatting."""
    with patch(
        "victor.tools.unified.code_tool.analyze_metrics", new_callable=AsyncMock
    ) as mock_metrics:
        mock_metrics.return_value = {"complexity": 10, "loc": 500}

        result = await code_tool("code metrics src/")

        assert "**complexity**: 10" in result
        assert "**loc**: 500" in result
