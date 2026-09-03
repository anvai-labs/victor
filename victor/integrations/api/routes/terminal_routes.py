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

"""Terminal routes: /terminal/suggest, /terminal/execute."""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from victor.integrations.api.fastapi_server import (
    TerminalCommandRequest,
    TerminalCommandResponse,
)

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create terminal routes bound to *server*."""
    router = APIRouter(tags=["Terminal"])

    @router.post("/terminal/suggest")
    async def terminal_suggest(intent: str = Query(..., min_length=1)) -> JSONResponse:
        """Suggest a terminal command based on user intent."""
        try:
            orchestrator = await server._get_orchestrator()
            prompt = f"""Generate a terminal command for this task. Return ONLY the command, nothing else.

Working directory: {server.workspace_root}
OS: {__import__('sys').platform}
Task: {intent}

Respond with just the command to run."""

            response = await orchestrator.chat(prompt)
            command = response.get("content", "").strip()

            if command.startswith("```"):
                lines = command.split("\n")
                command = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            command = command.strip()

            if not command:
                return JSONResponse({"error": "Could not generate command"}, status_code=400)

            dangerous_patterns = [
                "rm -rf /",
                "rm -rf ~",
                "> /dev/sd",
                "mkfs.",
                "dd if=",
                "sudo rm",
                ":(){",
                "chmod -R 777",
                "curl | sh",
                "wget | sh",
            ]
            is_dangerous = any(p in command.lower() for p in dangerous_patterns)

            cmd_id = f"cmd-{int(time.time() * 1000)}"
            return JSONResponse(
                {
                    "command_id": cmd_id,
                    "command": command,
                    "description": intent,
                    "is_dangerous": is_dangerous,
                    "status": "pending",
                }
            )

        except Exception:
            logger.exception("Terminal suggest error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/terminal/execute", response_model=TerminalCommandResponse)
    async def terminal_execute(
        request: TerminalCommandRequest,
    ) -> TerminalCommandResponse:
        """Execute a terminal command."""
        cmd_id = f"cmd-{int(time.time() * 1000)}"
        working_dir = request.working_dir or server.workspace_root

        resolved_dir = Path(working_dir).resolve()
        workspace_resolved = Path(server.workspace_root).resolve()
        if not str(resolved_dir).startswith(str(workspace_resolved)):
            return TerminalCommandResponse(
                command_id=cmd_id,
                command=request.command,
                status="failed",
                output="Working directory must be within workspace",
                exit_code=-1,
                is_dangerous=True,
                requires_approval=False,
            )

        dangerous_patterns = [
            "rm -rf /",
            "rm -rf ~",
            "> /dev/sd",
            "mkfs.",
            "dd if=",
            "sudo rm",
            ":(){",
            # Matched against the lowercased command — keep the pattern
            # lowercase too (the previous "chmod -R 777" could never match).
            "chmod -r 777",
        ]
        is_dangerous = any(p in request.command.lower() for p in dangerous_patterns)

        # Server-side approval policy: blocklisted (dangerous) commands ALWAYS
        # return pending — the caller-attested require_approval flag can no
        # longer bypass execution of a dangerous command (co-design review
        # U7-F2). Route auth is enforced by the app-level dependency.
        if is_dangerous:
            return TerminalCommandResponse(
                command_id=cmd_id,
                command=request.command,
                status="pending",
                is_dangerous=True,
                requires_approval=True,
            )

        try:
            cmd_parts = shlex.split(request.command)
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=working_dir,
            )

            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=request.timeout)
                output = stdout.decode("utf-8", errors="replace")
                exit_code = proc.returncode

                return TerminalCommandResponse(
                    command_id=cmd_id,
                    command=request.command,
                    status="completed" if exit_code == 0 else "failed",
                    output=output[:50000],
                    exit_code=exit_code,
                    is_dangerous=is_dangerous,
                    requires_approval=False,
                )

            except asyncio.TimeoutError:
                proc.kill()
                return TerminalCommandResponse(
                    command_id=cmd_id,
                    command=request.command,
                    status="failed",
                    output="Command timed out",
                    exit_code=-1,
                    is_dangerous=is_dangerous,
                    requires_approval=False,
                )

        except Exception as e:
            logger.exception("Terminal execute error")
            return TerminalCommandResponse(
                command_id=cmd_id,
                command=request.command,
                status="failed",
                output=str(e),
                exit_code=-1,
                is_dangerous=is_dangerous,
                requires_approval=False,
            )

    return router
