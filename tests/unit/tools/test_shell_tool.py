import pytest
from unittest.mock import patch, AsyncMock
from victor.tools.unified.shell_tool import shell_tool
from victor.tools.bash import shell as execute_bash


@pytest.mark.asyncio
async def test_execute_bash_failure_names_the_command():
    """Exit-code failures must attribute the failing command in the error
    message (log analysis: 22/30 ERROR lines in a 3h window carried only
    'Command failed with exit code N', unattributable in the file log)."""
    result = await execute_bash(cmd="exit 7")
    assert result["success"] is False
    assert result["return_code"] == 7
    assert "exit code 7" in result["error"]
    assert "exit 7" in result["error"]  # the command itself is named


@pytest.mark.asyncio
async def test_execute_bash_failure_command_preview_is_bounded():
    """Very long commands are truncated to 200 chars + '...' in the error."""
    long_cmd = "echo " + "a" * 500 + " && exit 3"
    result = await execute_bash(cmd=long_cmd)
    assert result["success"] is False
    assert len(result["error"]) < 1200  # no full command/stderr dump
    assert "..." in result["error"]


@pytest.mark.asyncio
async def test_shell_tool_basic():
    """Test `shell` formatting of stdout/stderr."""
    with patch("victor.tools.unified.shell_tool.execute_bash", new_callable=AsyncMock) as mock_bash:
        mock_bash.return_value = {"stdout": "files list", "stderr": ""}

        result = await shell_tool('shell "ls -la"')

        mock_bash.assert_called_once_with(cmd="ls -la", sandbox=False, readonly=False)
        assert "### STDOUT" in result
        assert "files list" in result


@pytest.mark.asyncio
async def test_shell_tool_sandbox():
    """Test `shell --sandbox` flag."""
    with patch("victor.tools.unified.shell_tool.execute_bash", new_callable=AsyncMock) as mock_bash:
        mock_bash.return_value = {"stdout": "sandboxed output", "stderr": "warn"}

        result = await shell_tool('shell "python script.py" --sandbox')

        mock_bash.assert_called_once_with(cmd="python script.py", sandbox=True, readonly=False)
        assert "### STDOUT" in result
        assert "### STDERR" in result
        assert "warn" in result


@pytest.mark.asyncio
async def test_shell_tool_preserves_pipe_and_redirection():
    """Shell wrapper must pass bash syntax through as an opaque command."""
    with patch("victor.tools.unified.shell_tool.execute_bash", new_callable=AsyncMock) as mock_bash:
        mock_bash.return_value = {"stdout": "ok", "stderr": ""}

        result = await shell_tool(r"printf 'a\nb\n' \| sed -n '1p' > /tmp/out.txt")

        mock_bash.assert_called_once_with(
            cmd=r"printf 'a\nb\n' \| sed -n '1p' > /tmp/out.txt",
            sandbox=False,
            readonly=False,
        )
        assert "### STDOUT" in result


@pytest.mark.asyncio
async def test_shell_tool_preserves_python_heredoc_with_docstring():
    """Heredoc Python cells should not be shlex-split before execution."""
    command = 'python - <<\'PY\'\n"""module docstring"""\nprint(\'hello\')\nPY'
    with patch("victor.tools.unified.shell_tool.execute_bash", new_callable=AsyncMock) as mock_bash:
        mock_bash.return_value = {"stdout": "hello\n", "stderr": ""}

        result = await shell_tool(command)

        mock_bash.assert_called_once_with(cmd=command, sandbox=False, readonly=False)
        assert "hello" in result


@pytest.mark.asyncio
async def test_shell_tool_timebound_sync_background():
    """Test `shell --timebound-sync` drops to background if it exceeds time."""
    import asyncio
    from victor.agent.background_tasks import BackgroundTaskDef

    # Mock execute_bash to sleep for 0.2s
    async def slow_bash(*args, **kwargs):
        await asyncio.sleep(0.2)
        return {"stdout": "done", "stderr": ""}

    with patch("victor.tools.unified.shell_tool.execute_bash", new=slow_bash):
        # Timebound sync for 0.05s. It should fail to finish in time and return a BackgroundTaskDef.
        result = await shell_tool('shell "sleep 1" --timebound-sync 0')

        # We expect the tool to detect pending task and return BackgroundTaskDef directly
        assert isinstance(result, BackgroundTaskDef)
        assert result.context == "shell_tool: sleep 1"
