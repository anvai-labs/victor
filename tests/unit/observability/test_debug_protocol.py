# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Debug Adapter Protocol (DAP) types and data structures."""

from pathlib import Path

from victor.observability.debug.protocol import (
    AttachConfiguration,
    Breakpoint,
    DebugSession,
    DebugState,
    EvaluateResult,
    ExceptionInfo,
    LaunchConfiguration,
    Scope,
    SourceLocation,
    StackFrame,
    StopReason,
    Thread,
    Variable,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestDebugState:
    """Tests for DebugState enum."""

    def test_initializing(self):
        """Test INITIALIZING state."""
        assert DebugState.INITIALIZING.value == "initializing"

    def test_running(self):
        """Test RUNNING state."""
        assert DebugState.RUNNING.value == "running"

    def test_stopped(self):
        """Test STOPPED state."""
        assert DebugState.STOPPED.value == "stopped"

    def test_paused(self):
        """Test PAUSED state."""
        assert DebugState.PAUSED.value == "paused"

    def test_terminated(self):
        """Test TERMINATED state."""
        assert DebugState.TERMINATED.value == "terminated"

    def test_disconnected(self):
        """Test DISCONNECTED state."""
        assert DebugState.DISCONNECTED.value == "disconnected"


class TestStopReason:
    """Tests for StopReason enum."""

    def test_breakpoint(self):
        """Test BREAKPOINT reason."""
        assert StopReason.BREAKPOINT.value == "breakpoint"

    def test_step(self):
        """Test STEP reason."""
        assert StopReason.STEP.value == "step"

    def test_exception(self):
        """Test EXCEPTION reason."""
        assert StopReason.EXCEPTION.value == "exception"

    def test_pause(self):
        """Test PAUSE reason."""
        assert StopReason.PAUSE.value == "pause"

    def test_entry(self):
        """Test ENTRY reason."""
        assert StopReason.ENTRY.value == "entry"

    def test_function_breakpoint(self):
        """Test FUNCTION_BREAKPOINT reason."""
        assert StopReason.FUNCTION_BREAKPOINT.value == "function breakpoint"


# =============================================================================
# SOURCE LOCATION TESTS
# =============================================================================


class TestSourceLocation:
    """Tests for SourceLocation dataclass."""

    def test_creation_minimal(self):
        """Test minimal source location."""
        loc = SourceLocation(path=Path("test.py"), line=10)
        assert loc.path == Path("test.py")
        assert loc.line == 10
        assert loc.column is None

    def test_creation_full(self):
        """Test full source location."""
        loc = SourceLocation(
            path=Path("test.py"),
            line=10,
            column=5,
            end_line=15,
            end_column=20,
        )
        assert loc.column == 5
        assert loc.end_line == 15

    def test_str_basic(self):
        """Test string representation without column."""
        loc = SourceLocation(path=Path("test.py"), line=10)
        assert str(loc) == "test.py:10"

    def test_str_with_column(self):
        """Test string representation with column."""
        loc = SourceLocation(path=Path("test.py"), line=10, column=5)
        assert str(loc) == "test.py:10:5"


# =============================================================================
# BREAKPOINT TESTS
# =============================================================================


class TestBreakpoint:
    """Tests for Breakpoint dataclass."""

    def test_creation_minimal(self):
        """Test minimal breakpoint."""
        bp = Breakpoint(
            id=1,
            verified=True,
            source=SourceLocation(Path("test.py"), 10),
        )
        assert bp.id == 1
        assert bp.verified is True
        assert bp.enabled is True
        assert bp.hit_count == 0

    def test_creation_conditional(self):
        """Test conditional breakpoint."""
        bp = Breakpoint(
            id=2,
            verified=True,
            source=SourceLocation(Path("test.py"), 10),
            condition="x > 5",
        )
        assert bp.condition == "x > 5"

    def test_creation_hit_conditional(self):
        """Test hit count conditional breakpoint."""
        bp = Breakpoint(
            id=3,
            verified=True,
            source=SourceLocation(Path("test.py"), 10),
            hit_condition=">= 10",
        )
        assert bp.hit_condition == ">= 10"

    def test_creation_logpoint(self):
        """Test logpoint."""
        bp = Breakpoint(
            id=4,
            verified=True,
            source=SourceLocation(Path("test.py"), 10),
            log_message="x = {x}, y = {y}",
        )
        assert bp.log_message == "x = {x}, y = {y}"


# =============================================================================
# THREAD TESTS
# =============================================================================


class TestThread:
    """Tests for Thread dataclass."""

    def test_creation_minimal(self):
        """Test minimal thread."""
        thread = Thread(id=1, name="MainThread")
        assert thread.id == 1
        assert thread.name == "MainThread"
        assert thread.state == DebugState.RUNNING

    def test_creation_with_state(self):
        """Test thread with state."""
        thread = Thread(id=2, name="Worker", state=DebugState.STOPPED)
        assert thread.state == DebugState.STOPPED


# =============================================================================
# STACK FRAME TESTS
# =============================================================================


class TestStackFrame:
    """Tests for StackFrame dataclass."""

    def test_creation_minimal(self):
        """Test minimal stack frame."""
        frame = StackFrame(id=1, name="main")
        assert frame.id == 1
        assert frame.name == "main"
        assert frame.source is None
        assert frame.presentation_hint == "normal"

    def test_creation_full(self):
        """Test full stack frame."""
        frame = StackFrame(
            id=2,
            name="process_data",
            source=SourceLocation(Path("module.py"), 42),
            module_id="mymodule",
            presentation_hint="subtle",
            can_restart=True,
        )
        assert frame.source.line == 42
        assert frame.can_restart is True


# =============================================================================
# SCOPE TESTS
# =============================================================================


class TestScope:
    """Tests for Scope dataclass."""

    def test_creation_minimal(self):
        """Test minimal scope."""
        scope = Scope(name="Locals", variables_reference=1)
        assert scope.name == "Locals"
        assert scope.variables_reference == 1
        assert scope.expensive is False

    def test_creation_full(self):
        """Test full scope."""
        scope = Scope(
            name="Globals",
            variables_reference=2,
            named_variables=50,
            indexed_variables=0,
            expensive=True,
            source=SourceLocation(Path("test.py"), 1),
        )
        assert scope.named_variables == 50
        assert scope.expensive is True


# =============================================================================
# VARIABLE TESTS
# =============================================================================


class TestVariable:
    """Tests for Variable dataclass."""

    def test_creation_simple(self):
        """Test simple variable."""
        var = Variable(name="x", value="42")
        assert var.name == "x"
        assert var.value == "42"
        assert var.variables_reference == 0

    def test_creation_with_type(self):
        """Test variable with type."""
        var = Variable(name="count", value="10", type="int")
        assert var.type == "int"

    def test_creation_complex(self):
        """Test complex variable (has children)."""
        var = Variable(
            name="data",
            value="{'a': 1, 'b': 2}",
            type="dict",
            variables_reference=5,
            named_variables=2,
        )
        assert var.variables_reference == 5
        assert var.named_variables == 2


# =============================================================================
# EXCEPTION INFO TESTS
# =============================================================================


class TestExceptionInfo:
    """Tests for ExceptionInfo dataclass."""

    def test_creation_minimal(self):
        """Test minimal exception info."""
        exc = ExceptionInfo(exception_id="ValueError")
        assert exc.exception_id == "ValueError"
        assert exc.break_mode == "always"

    def test_creation_full(self):
        """Test full exception info."""
        exc = ExceptionInfo(
            exception_id="KeyError",
            description="Key 'name' not found",
            break_mode="unhandled",
            details={"traceback": ["frame1", "frame2"]},
        )
        assert exc.description == "Key 'name' not found"
        assert exc.break_mode == "unhandled"


# =============================================================================
# EVALUATE RESULT TESTS
# =============================================================================


class TestEvaluateResult:
    """Tests for EvaluateResult dataclass."""

    def test_creation_simple(self):
        """Test simple evaluate result."""
        result = EvaluateResult(result="42")
        assert result.result == "42"
        assert result.variables_reference == 0

    def test_creation_complex(self):
        """Test complex evaluate result."""
        result = EvaluateResult(
            result="[1, 2, 3]",
            type="list",
            variables_reference=10,
            indexed_variables=3,
        )
        assert result.type == "list"
        assert result.indexed_variables == 3


# =============================================================================
# DEBUG SESSION TESTS
# =============================================================================


class TestDebugSession:
    """Tests for DebugSession dataclass."""

    def test_creation_minimal(self):
        """Test minimal debug session."""
        session = DebugSession(
            id="session-1",
            name="Debug Test",
            language="python",
        )
        assert session.id == "session-1"
        assert session.state == DebugState.INITIALIZING
        assert session.threads == []
        assert session.breakpoints == {}

    def test_creation_full(self):
        """Test full debug session."""
        session = DebugSession(
            id="session-2",
            name="Debug Main",
            language="python",
            state=DebugState.RUNNING,
            program=Path("main.py"),
            working_directory=Path("/project"),
            arguments=["--verbose"],
            environment={"DEBUG": "1"},
            supports_conditional_breakpoints=True,
            supports_log_points=True,
        )
        assert session.program == Path("main.py")
        assert session.supports_conditional_breakpoints is True

    def test_get_breakpoints_for_file_exists(self):
        """Test get_breakpoints_for_file when breakpoints exist."""
        bp = Breakpoint(
            id=1,
            verified=True,
            source=SourceLocation(Path("test.py"), 10),
        )
        session = DebugSession(
            id="s1",
            name="Test",
            language="python",
            breakpoints={str(Path("test.py").resolve()): [bp]},
        )
        bps = session.get_breakpoints_for_file(Path("test.py"))
        assert len(bps) == 1
        assert bps[0].id == 1

    def test_get_breakpoints_for_file_not_exists(self):
        """Test get_breakpoints_for_file when no breakpoints."""
        session = DebugSession(id="s1", name="Test", language="python")
        bps = session.get_breakpoints_for_file("test.py")
        assert bps == []

    def test_get_active_thread_found(self):
        """Test get_active_thread when thread exists."""
        thread = Thread(id=1, name="Main")
        session = DebugSession(
            id="s1",
            name="Test",
            language="python",
            threads=[thread],
            current_thread_id=1,
        )
        active = session.get_active_thread()
        assert active is not None
        assert active.name == "Main"

    def test_get_active_thread_no_current(self):
        """Test get_active_thread with no current thread."""
        session = DebugSession(id="s1", name="Test", language="python")
        assert session.get_active_thread() is None

    def test_get_active_thread_not_found(self):
        """Test get_active_thread when current thread not in list."""
        session = DebugSession(
            id="s1",
            name="Test",
            language="python",
            threads=[Thread(id=1, name="Main")],
            current_thread_id=999,
        )
        assert session.get_active_thread() is None


# =============================================================================
# LAUNCH CONFIGURATION TESTS
# =============================================================================


class TestLaunchConfiguration:
    """Tests for LaunchConfiguration dataclass."""

    def test_creation_minimal(self):
        """Test minimal launch config."""
        config = LaunchConfiguration(
            program=Path("main.py"),
            language="python",
        )
        assert config.program == Path("main.py")
        assert config.language == "python"
        assert config.arguments == []
        assert config.stop_on_entry is False

    def test_creation_full(self):
        """Test full launch config."""
        config = LaunchConfiguration(
            program=Path("main.py"),
            language="python",
            arguments=["--verbose", "--config", "test.yaml"],
            working_directory=Path("/project"),
            environment={"DEBUG": "1", "LOG_LEVEL": "debug"},
            stop_on_entry=True,
            no_debug=False,
            extra_options={"justMyCode": True},
        )
        assert len(config.arguments) == 3
        assert config.stop_on_entry is True
        assert config.extra_options["justMyCode"] is True


# =============================================================================
# ATTACH CONFIGURATION TESTS
# =============================================================================


class TestAttachConfiguration:
    """Tests for AttachConfiguration dataclass."""

    def test_creation_default(self):
        """Test default attach config."""
        config = AttachConfiguration()
        assert config.process_id is None
        assert config.host is None
        assert config.port is None
        assert config.language == "python"

    def test_creation_process_id(self):
        """Test attach config with process ID."""
        config = AttachConfiguration(
            process_id=12345,
            language="python",
        )
        assert config.process_id == 12345

    def test_creation_remote(self):
        """Test remote attach config."""
        config = AttachConfiguration(
            host="localhost",
            port=5678,
            language="python",
            extra_options={"pathMappings": [{"localRoot": "/src", "remoteRoot": "/app"}]},
        )
        assert config.host == "localhost"
        assert config.port == 5678
