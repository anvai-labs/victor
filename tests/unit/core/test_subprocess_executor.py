# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0

"""Tests for subprocess_executor module."""

import pytest

from victor.tools.subprocess_executor import (
    CommandErrorType,
    CommandResult,
    DANGEROUS_COMMANDS,
    DANGEROUS_PATTERNS,
    check_docker_available,
    check_git_available,
    check_npm_available,
    check_pip_available,
    is_dangerous_command,
    is_tool_available,
    parse_docker_ps,
    parse_git_status,
    run_command,
    run_command_async,
    run_docker,
    run_docker_async,
    run_git,
    run_git_async,
    run_npm,
    run_pip,
)


class TestCommandResult:
    """Tests for CommandResult dataclass."""

    def test_command_result_success(self):
        """Test successful CommandResult."""
        result = CommandResult(
            success=True,
            stdout="output",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
            command="echo test",
        )

        assert result.success is True
        assert result.stdout == "output"
        assert result.return_code == 0
        assert result.error_type == CommandErrorType.SUCCESS

    def test_command_result_failure(self):
        """Test failed CommandResult."""
        result = CommandResult(
            success=False,
            stdout="",
            stderr="error",
            return_code=1,
            error_type=CommandErrorType.EXECUTION_ERROR,
            command="false",
        )

        assert result.success is False
        assert result.return_code == 1
        assert result.error_type == CommandErrorType.EXECUTION_ERROR

    def test_command_result_with_timeout(self):
        """Test CommandResult with timeout."""
        result = CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.TIMEOUT,
            error_message="Command timed out",
        )

        assert result.success is False
        assert result.error_type == CommandErrorType.TIMEOUT

    def test_command_result_optional_fields(self):
        """Test CommandResult optional fields."""
        result = CommandResult(
            success=True,
            stdout="output",
            stderr="",
            return_code=0,
            error_type=CommandErrorType.SUCCESS,
            error_message=None,
            command="test",
            working_dir="/tmp",
            duration_ms=100.5,
        )

        assert result.working_dir == "/tmp"
        assert result.duration_ms == 100.5


class TestCommandErrorType:
    """Tests for CommandErrorType enum."""

    def test_error_types_exist(self):
        """Test that expected error types exist."""
        assert hasattr(CommandErrorType, "TIMEOUT")
        assert hasattr(CommandErrorType, "NOT_FOUND")
        assert hasattr(CommandErrorType, "PERMISSION_DENIED")
        assert hasattr(CommandErrorType, "DANGEROUS_COMMAND")
        assert hasattr(CommandErrorType, "SUCCESS")
        assert hasattr(CommandErrorType, "EXECUTION_ERROR")
        assert hasattr(CommandErrorType, "UNKNOWN")

    def test_error_type_values(self):
        """Test error type values are strings."""
        assert isinstance(CommandErrorType.SUCCESS.value, str)
        assert isinstance(CommandErrorType.TIMEOUT.value, str)


class TestRunCommand:
    """Tests for run_command function."""

    def test_run_simple_command(self):
        """Test running a simple command."""
        result = run_command(["echo", "hello"])

        assert result.success is True
        assert "hello" in result.stdout

    def test_run_command_string(self):
        """Test running a command as string."""
        result = run_command("echo hello", shell=True)

        assert result.success is True
        assert "hello" in result.stdout

    def test_run_command_with_error(self):
        """Test running a command that fails."""
        result = run_command(["false"])

        assert result.success is False
        assert result.return_code != 0

    def test_run_command_with_working_dir(self, tmp_path):
        """Test running command in specific directory."""
        result = run_command(["pwd"], working_dir=str(tmp_path))

        assert result.success is True
        assert str(tmp_path) in result.stdout

    def test_run_command_captures_stderr(self):
        """Test that stderr is captured."""
        result = run_command(["ls", "nonexistent_file_12345"])

        assert result.success is False
        # Error could be in stdout or stderr depending on shell
        assert result.stderr != "" or "No such file" in result.stdout or result.return_code != 0

    def test_run_command_with_timeout(self):
        """Test command with timeout."""
        result = run_command(["sleep", "10"], timeout=1)

        assert result.success is False
        assert result.error_type == CommandErrorType.TIMEOUT

    def test_run_command_returns_duration(self):
        """Test that command returns duration."""
        result = run_command(["echo", "test"])

        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    def test_run_command_with_env(self):
        """Test running command with custom environment."""
        result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True)

        assert result.success is True


class TestRunCommandAsync:
    """Tests for run_command_async function."""

    @pytest.mark.asyncio
    async def test_run_async_simple_command(self):
        """Test running a simple command asynchronously."""
        result = await run_command_async("echo async_test")

        assert result.success is True
        assert "async_test" in result.stdout

    @pytest.mark.asyncio
    async def test_run_async_with_working_dir(self, tmp_path):
        """Test running async command in specific directory."""
        result = await run_command_async("pwd", working_dir=str(tmp_path))

        assert result.success is True
        assert str(tmp_path) in result.stdout

    @pytest.mark.asyncio
    async def test_run_async_command_failure(self):
        """Test running async command that fails."""
        result = await run_command_async("false")

        assert result.success is False
        assert result.return_code != 0


class TestIsDangerousCommand:
    """Tests for is_dangerous_command function."""

    def test_dangerous_rm_rf(self):
        """Test that rm -rf is dangerous."""
        assert is_dangerous_command("rm -rf /") is True
        assert is_dangerous_command("rm -rf ~") is True

    def test_dangerous_format(self):
        """Test that format commands are dangerous."""
        assert is_dangerous_command("mkfs.ext4 /dev/sda") is True

    def test_safe_command(self):
        """Test that safe commands are not flagged."""
        assert is_dangerous_command("ls -la") is False
        assert is_dangerous_command("echo hello") is False
        assert is_dangerous_command("cat file.txt") is False

    def test_dangerous_patterns_exist(self):
        """Test that dangerous patterns list exists and has entries."""
        assert len(DANGEROUS_PATTERNS) > 0

    def test_dangerous_commands_exist(self):
        """Test that dangerous commands list exists and has entries."""
        assert len(DANGEROUS_COMMANDS) > 0


class TestRunGit:
    """Tests for run_git function (returns tuple)."""

    def test_run_git_version(self):
        """Test running git version."""
        success, stdout, stderr = run_git("--version")

        if success:
            assert "git" in stdout.lower()

    def test_run_git_status(self, tmp_path):
        """Test running git status in a non-git directory."""
        success, stdout, stderr = run_git("status", working_dir=str(tmp_path))

        # Should fail because tmp_path is not a git repo
        assert success is False


class TestRunGitAsync:
    """Tests for run_git_async function."""

    @pytest.mark.asyncio
    async def test_run_git_async_version(self):
        """Test running git version asynchronously."""
        success, stdout, stderr = await run_git_async("--version")

        if success:
            assert "git" in stdout.lower()


class TestRunDocker:
    """Tests for run_docker function (returns tuple)."""

    def test_run_docker_version(self):
        """Test running docker version."""
        success, stdout, stderr = run_docker("--version")

        if success:
            assert "docker" in stdout.lower()


class TestRunDockerAsync:
    """Tests for run_docker_async function."""

    @pytest.mark.asyncio
    async def test_run_docker_async_version(self):
        """Test running docker version asynchronously."""
        success, stdout, stderr = await run_docker_async("--version")

        if success:
            assert "docker" in stdout.lower()


class TestRunNpm:
    """Tests for run_npm function (returns tuple)."""

    def test_run_npm_version(self):
        """Test running npm version."""
        success, stdout, stderr = run_npm("--version")

        if success:
            # npm version is just a version number
            assert stdout.strip() != ""


class TestRunPip:
    """Tests for run_pip function (returns tuple)."""

    def test_run_pip_version(self):
        """Test running pip version."""
        success, stdout, stderr = run_pip("--version")

        if success:
            assert "pip" in stdout.lower()


class TestToolAvailability:
    """Tests for tool availability check functions."""

    def test_check_git_available(self):
        """Test checking git availability."""
        available = check_git_available()
        # Just verify it returns a boolean
        assert isinstance(available, bool)

    def test_check_docker_available(self):
        """Test checking docker availability."""
        available = check_docker_available()
        assert isinstance(available, bool)

    def test_check_npm_available(self):
        """Test checking npm availability."""
        available = check_npm_available()
        assert isinstance(available, bool)

    def test_check_pip_available(self):
        """Test checking pip availability."""
        available = check_pip_available()
        assert isinstance(available, bool)

    def test_is_tool_available(self):
        """Test generic tool availability check."""
        # echo should always be available
        assert is_tool_available("echo") is True
        # nonexistent tool should not be available
        assert is_tool_available("nonexistent_tool_12345") is False


class TestParseGitStatus:
    """Tests for parse_git_status function."""

    def test_parse_clean_status(self):
        """Test parsing clean git status."""
        output = "On branch main\nnothing to commit, working tree clean"
        result = parse_git_status(output)

        assert result is not None
        assert isinstance(result, dict)

    def test_parse_modified_status(self):
        """Test parsing git status with modified files."""
        output = """On branch main
Changes not staged for commit:
  modified:   file.txt

Untracked files:
  newfile.txt
"""
        result = parse_git_status(output)

        assert result is not None
        assert isinstance(result, dict)


class TestParseDockerPs:
    """Tests for parse_docker_ps function."""

    def test_parse_empty_output(self):
        """Test parsing empty docker ps output."""
        output = "CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES"
        result = parse_docker_ps(output)

        assert result is not None
        assert isinstance(result, list)

    def test_parse_with_containers(self):
        """Test parsing docker ps with containers."""
        output = """CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS   PORTS   NAMES
abc123         nginx     "nginx"   1h ago    Up 1h    80/tcp  web"""
        result = parse_docker_ps(output)

        assert result is not None
        assert isinstance(result, list)
