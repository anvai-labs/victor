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

"""Python debug adapter using debugpy.

This adapter provides full debugging support for Python using Microsoft's
debugpy library (the same debugger used by VS Code Python extension).

Features:
- Line and conditional breakpoints
- Step over/into/out
- Variable inspection
- Expression evaluation
- Multi-threaded debugging
- Remote debugging support

Requirements:
    pip install debugpy

Usage:
    adapter = PythonDebugAdapter()
    await adapter.initialize()
    session = await adapter.launch(config)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.observability.debug.adapter import (
    BaseDebugAdapter,
    DebugAdapterCapabilities,
    DebugEvent,
    DebugEventType,
)
from victor.observability.debug.protocol import (
    AttachConfiguration,
    Breakpoint,
    DebugSession,
    DebugState,
    EvaluateResult,
    LaunchConfiguration,
    Scope,
    SourceLocation,
    StackFrame,
    Thread,
    Variable,
)

logger = logging.getLogger(__name__)

# Check for debugpy availability
try:
    import debugpy
    from debugpy.common import messaging

    HAS_DEBUGPY = True
except ImportError:
    HAS_DEBUGPY = False
    debugpy = None
    messaging = None


class PythonDebugAdapter(BaseDebugAdapter):
    """Debug adapter for Python using debugpy.

    Communicates with debugpy using the Debug Adapter Protocol (DAP)
    over stdio or sockets.
    """

    def __init__(self, port: Optional[int] = None):
        """Initialize Python debug adapter.

        Args:
            port: Optional port for debugpy server (auto-selects if None)
        """
        super().__init__()
        self._port = port or self._find_free_port()
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._seq = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        return "debugpy"

    @property
    def languages(self) -> List[str]:
        return ["python", "py", "python3"]

    def _get_default_capabilities(self) -> DebugAdapterCapabilities:
        return DebugAdapterCapabilities(
            supports_conditional_breakpoints=True,
            supports_hit_conditional_breakpoints=True,
            supports_log_points=True,
            supports_function_breakpoints=True,
            supports_data_breakpoints=False,
            supports_step_back=False,
            supports_restart_frame=True,
            supports_goto_targets=False,
            supports_stepping_granularity=True,
            supports_terminate_request=True,
            supports_suspend_debuggee=True,
            supports_evaluate_for_hovers=True,
            supports_set_variable=True,
            supports_set_expression=True,
            supports_completions_request=True,
            supports_exception_options=True,
            supports_exception_filter_options=True,
            exception_breakpoint_filters=[
                {
                    "filter": "raised",
                    "label": "Raised Exceptions",
                    "description": "Break on all raised exceptions",
                    "default": False,
                },
                {
                    "filter": "uncaught",
                    "label": "Uncaught Exceptions",
                    "description": "Break on uncaught exceptions",
                    "default": True,
                },
            ],
            supports_modules_request=True,
            supports_loaded_sources_request=True,
            supported_languages=["python"],
        )

    async def initialize(self) -> DebugAdapterCapabilities:
        """Initialize the debugpy adapter."""
        if not HAS_DEBUGPY:
            raise RuntimeError("debugpy is not installed. Install with: pip install debugpy")

        if self._initialized:
            return self.capabilities

        self._initialized = True
        logger.info(f"Python debug adapter initialized on port {self._port}")
        return self.capabilities

    async def launch(self, config: LaunchConfiguration) -> DebugSession:
        """Launch a Python program for debugging."""
        session_id = str(uuid.uuid4())

        # Build debugpy command
        program = str(config.program)
        working_dir = str(config.working_directory or config.program.parent)

        cmd = [
            sys.executable,
            "-m",
            "debugpy",
            "--listen",
            f"127.0.0.1:{self._port}",
            "--wait-for-client",
        ]

        if config.stop_on_entry:
            cmd.append("--stop-on-entry")

        cmd.append(program)
        cmd.extend(config.arguments)

        # Prepare environment
        env = os.environ.copy()
        env.update(config.environment)

        # Start the debuggee process
        logger.info(f"Launching: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait a moment for debugpy to start listening
        await asyncio.sleep(0.5)

        # Connect to debugpy
        await self._connect()

        # Send initialize request
        await self._send_request(
            "initialize",
            {
                "clientID": "victor",
                "clientName": "Victor Debug",
                "adapterID": "debugpy",
                "pathFormat": "path",
                "linesStartAt1": True,
                "columnsStartAt1": True,
                "supportsVariableType": True,
                "supportsVariablePaging": True,
                "supportsRunInTerminalRequest": False,
            },
        )

        # Send launch request
        await self._send_request(
            "launch",
            {
                "program": program,
                "args": config.arguments,
                "cwd": working_dir,
                "env": config.environment,
                "stopOnEntry": config.stop_on_entry,
                "justMyCode": True,
            },
        )

        # Create session
        session = DebugSession(
            id=session_id,
            name=config.program.name,
            language="python",
            state=DebugState.RUNNING,
            program=config.program,
            working_directory=config.working_directory,
            arguments=config.arguments,
            environment=config.environment,
            supports_conditional_breakpoints=True,
            supports_hit_conditional_breakpoints=True,
            supports_log_points=True,
        )

        self._sessions[session_id] = session

        # Emit initialized event
        self._emit_event(
            DebugEvent(
                type=DebugEventType.INITIALIZED,
                session_id=session_id,
            )
        )

        return session

    async def attach(self, config: AttachConfiguration) -> DebugSession:
        """Attach to a running Python process."""
        session_id = str(uuid.uuid4())

        # Connect to debugpy server
        host = config.host or "127.0.0.1"
        port = config.port or 5678

        self._port = port
        await self._connect(host)

        # Send attach request
        await self._send_request(
            "attach",
            {
                "connect": {"host": host, "port": port},
                "justMyCode": True,
            },
        )

        session = DebugSession(
            id=session_id,
            name=f"Attached to {config.process_id or f'{host}:{port}'}",
            language="python",
            state=DebugState.RUNNING,
        )

        self._sessions[session_id] = session
        return session

    async def disconnect(self, session_id: str, terminate: bool = False) -> None:
        """Disconnect from debug session."""
        try:
            await self._send_request("disconnect", {"terminateDebuggee": terminate})
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            await self._cleanup()
            if session_id in self._sessions:
                del self._sessions[session_id]

    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        await self._cleanup()
        self._initialized = False

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    # Breakpoint operations

    async def set_breakpoints(
        self,
        session_id: str,
        source: Path,
        breakpoints: List[SourceLocation],
        conditions: Optional[Dict[int, str]] = None,
    ) -> List[Breakpoint]:
        """Set breakpoints in a source file."""
        conditions = conditions or {}

        bp_args = []
        for loc in breakpoints:
            bp = {"line": loc.line}
            if loc.line in conditions:
                bp["condition"] = conditions[loc.line]
            bp_args.append(bp)

        response = await self._send_request(
            "setBreakpoints",
            {
                "source": {"path": str(source)},
                "breakpoints": bp_args,
            },
        )

        result = []
        for i, bp_data in enumerate(response.get("breakpoints", [])):
            result.append(
                Breakpoint(
                    id=bp_data.get("id", i),
                    verified=bp_data.get("verified", False),
                    source=SourceLocation(
                        path=source,
                        line=bp_data.get("line", breakpoints[i].line),
                    ),
                    condition=conditions.get(breakpoints[i].line),
                    message=bp_data.get("message"),
                )
            )

        return result

    async def set_function_breakpoints(self, session_id: str, names: List[str]) -> List[Breakpoint]:
        """Set breakpoints on function names."""
        response = await self._send_request(
            "setFunctionBreakpoints",
            {
                "breakpoints": [{"name": name} for name in names],
            },
        )

        result = []
        for i, bp_data in enumerate(response.get("breakpoints", [])):
            # Function breakpoints don't have a known source location initially
            result.append(
                Breakpoint(
                    id=bp_data.get("id", i),
                    verified=bp_data.get("verified", False),
                    source=SourceLocation(
                        path=Path(bp_data.get("source", {}).get("path", "")),
                        line=bp_data.get("line", 0),
                    ),
                    message=bp_data.get("message"),
                )
            )

        return result

    async def clear_breakpoints(self, session_id: str, source: Path) -> None:
        """Clear all breakpoints in a file."""
        await self._send_request(
            "setBreakpoints",
            {
                "source": {"path": str(source)},
                "breakpoints": [],
            },
        )

    # Execution control

    async def continue_execution(self, session_id: str, thread_id: Optional[int] = None) -> None:
        """Continue execution."""
        args = {}
        if thread_id is not None:
            args["threadId"] = thread_id
        await self._send_request("continue", args)
        self._update_session_state(session_id, DebugState.RUNNING)

    async def pause(self, session_id: str, thread_id: Optional[int] = None) -> None:
        """Pause execution."""
        args = {}
        if thread_id is not None:
            args["threadId"] = thread_id
        await self._send_request("pause", args)

    async def step_over(self, session_id: str, thread_id: int) -> None:
        """Step over."""
        await self._send_request("next", {"threadId": thread_id})

    async def step_into(self, session_id: str, thread_id: int) -> None:
        """Step into."""
        await self._send_request("stepIn", {"threadId": thread_id})

    async def step_out(self, session_id: str, thread_id: int) -> None:
        """Step out."""
        await self._send_request("stepOut", {"threadId": thread_id})

    # Inspection

    async def get_threads(self, session_id: str) -> List[Thread]:
        """Get all threads."""
        response = await self._send_request("threads", {})

        return [
            Thread(
                id=t["id"],
                name=t.get("name", f"Thread {t['id']}"),
            )
            for t in response.get("threads", [])
        ]

    async def get_stack_trace(
        self,
        session_id: str,
        thread_id: int,
        start_frame: int = 0,
        levels: int = 20,
    ) -> List[StackFrame]:
        """Get stack trace."""
        response = await self._send_request(
            "stackTrace",
            {
                "threadId": thread_id,
                "startFrame": start_frame,
                "levels": levels,
            },
        )

        return [
            StackFrame(
                id=f["id"],
                name=f.get("name", ""),
                source=(
                    SourceLocation(
                        path=Path(f.get("source", {}).get("path", "")),
                        line=f.get("line", 0),
                        column=f.get("column"),
                    )
                    if f.get("source")
                    else None
                ),
            )
            for f in response.get("stackFrames", [])
        ]

    async def get_scopes(self, session_id: str, frame_id: int) -> List[Scope]:
        """Get variable scopes."""
        response = await self._send_request("scopes", {"frameId": frame_id})

        return [
            Scope(
                name=s.get("name", ""),
                variables_reference=s.get("variablesReference", 0),
                expensive=s.get("expensive", False),
            )
            for s in response.get("scopes", [])
        ]

    async def get_variables(
        self,
        session_id: str,
        variables_reference: int,
        filter_type: Optional[str] = None,
        start: int = 0,
        count: int = 100,
    ) -> List[Variable]:
        """Get variables."""
        args: Dict[str, Any] = {
            "variablesReference": variables_reference,
            "start": start,
            "count": count,
        }
        if filter_type:
            args["filter"] = filter_type

        response = await self._send_request("variables", args)

        return [
            Variable(
                name=v.get("name", ""),
                value=v.get("value", ""),
                type=v.get("type"),
                variables_reference=v.get("variablesReference", 0),
                evaluate_name=v.get("evaluateName"),
            )
            for v in response.get("variables", [])
        ]

    async def evaluate(
        self,
        session_id: str,
        expression: str,
        frame_id: Optional[int] = None,
        context: str = "repl",
    ) -> EvaluateResult:
        """Evaluate expression."""
        args: Dict[str, Any] = {
            "expression": expression,
            "context": context,
        }
        if frame_id is not None:
            args["frameId"] = frame_id

        response = await self._send_request("evaluate", args)

        return EvaluateResult(
            result=response.get("result", ""),
            type=response.get("type"),
            variables_reference=response.get("variablesReference", 0),
        )

    # DAP communication

    async def _connect(self, host: str = "127.0.0.1") -> None:
        """Connect to debugpy server."""
        max_retries = 10
        for i in range(max_retries):
            try:
                self._reader, self._writer = await asyncio.open_connection(host, self._port)
                logger.info(f"Connected to debugpy at {host}:{self._port}")
                return
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    raise RuntimeError(f"Could not connect to debugpy at {host}:{self._port}")

    async def _send_request(self, command: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Send DAP request and wait for response."""
        if self._writer is None:
            raise RuntimeError("Not connected to debugpy")

        self._seq += 1
        seq = self._seq

        request = {
            "seq": seq,
            "type": "request",
            "command": command,
            "arguments": arguments,
        }

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[seq] = future

        # Send request
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        self._writer.write(message.encode())
        await self._writer.drain()

        # Start reading responses if not already
        asyncio.create_task(self._read_messages())

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response.get("body", {})
        except asyncio.TimeoutError:
            del self._pending_requests[seq]
            raise RuntimeError(f"Timeout waiting for response to {command}")

    async def _read_messages(self) -> None:
        """Read and dispatch DAP messages."""
        if self._reader is None:
            return

        try:
            while True:
                # Read headers
                header = await self._reader.readline()
                if not header:
                    break

                content_length = 0
                while header and header.strip():
                    if header.startswith(b"Content-Length:"):
                        content_length = int(header.split(b":")[1].strip())
                    header = await self._reader.readline()

                if content_length == 0:
                    continue

                # Read content
                content = await self._reader.read(content_length)
                message = json.loads(content.decode())

                self._handle_message(message)

        except Exception as e:
            logger.error(f"Error reading messages: {e}")

    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming DAP message."""
        msg_type = message.get("type")

        if msg_type == "response":
            seq = message.get("request_seq")
            if seq in self._pending_requests:
                future = self._pending_requests.pop(seq)
                if not future.done():
                    if message.get("success"):
                        future.set_result(message)
                    else:
                        error = message.get("message", "Unknown error")
                        future.set_exception(RuntimeError(error))

        elif msg_type == "event":
            event_name = message.get("event")
            body = message.get("body", {})

            # Map DAP events to our events
            event_type = {
                "initialized": DebugEventType.INITIALIZED,
                "stopped": DebugEventType.STOPPED,
                "continued": DebugEventType.CONTINUED,
                "exited": DebugEventType.EXITED,
                "terminated": DebugEventType.TERMINATED,
                "thread": DebugEventType.THREAD_STARTED,
                "output": DebugEventType.OUTPUT,
                "breakpoint": DebugEventType.BREAKPOINT_CHANGED,
                "module": DebugEventType.MODULE_LOADED,
            }.get(event_name)

            if event_type:
                # Find session for this event
                session_id = next(iter(self._sessions.keys()), "")
                self._emit_event(
                    DebugEvent(
                        type=event_type,
                        session_id=session_id,
                        data=body,
                    )
                )

                # Update session state on stopped/continued
                if event_type == DebugEventType.STOPPED and session_id:
                    session = self._sessions.get(session_id)
                    if session:
                        session.state = DebugState.STOPPED
                        session.current_thread_id = body.get("threadId")
                elif event_type == DebugEventType.CONTINUED and session_id:
                    session = self._sessions.get(session_id)
                    if session:
                        session.state = DebugState.RUNNING

    def _find_free_port(self) -> int:
        """Find a free port for debugpy."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
