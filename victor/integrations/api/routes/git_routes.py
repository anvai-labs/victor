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

"""Git & Patch routes: /git/*, /patch/*."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from victor.integrations.api.change_tracker_ops import (
    apply_patch_request,
    create_patch_request,
)
from victor.integrations.api.fastapi_server import (
    GitCommitRequest,
    PatchApplyRequest,
    PatchCreateRequest,
)

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create git & patch routes bound to *server*."""
    router = APIRouter()

    # ---- Patch operations ----

    @router.post("/patch/apply", tags=["Patch"])
    async def apply_patch(request: PatchApplyRequest) -> JSONResponse:
        """Apply a patch."""
        result = await apply_patch_request(patch=request.patch, dry_run=request.dry_run)
        return JSONResponse(result)

    @router.post("/patch/create", tags=["Patch"])
    async def create_patch(request: PatchCreateRequest) -> JSONResponse:
        """Create a patch."""
        result = await create_patch_request(
            file_path=request.file_path, new_content=request.new_content
        )
        return JSONResponse(result)

    # ---- Git operations ----

    @router.get("/git/status", tags=["Git"])
    async def git_status() -> JSONResponse:
        """Get git status for the workspace."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "status", "--porcelain", "-b"],
                cwd=server.workspace_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return JSONResponse({"is_git_repo": False, "error": result.stderr})

            lines = result.stdout.strip().split("\n")
            branch_line = lines[0] if lines else ""

            branch = "unknown"
            tracking = None
            if branch_line.startswith("## "):
                branch_info = branch_line[3:]
                if "..." in branch_info:
                    parts = branch_info.split("...")
                    branch = parts[0]
                    tracking = parts[1].split()[0] if len(parts) > 1 else None
                else:
                    branch = branch_info.split()[0]

            staged = []
            unstaged = []
            untracked = []

            for line in lines[1:]:
                if not line.strip():
                    continue
                status = line[:2]
                filepath = line[3:]

                if status[0] in "MADRC":
                    staged.append({"status": status[0], "file": filepath})
                if status[1] in "MD":
                    unstaged.append({"status": status[1], "file": filepath})
                if status == "??":
                    untracked.append(filepath)

            return JSONResponse(
                {
                    "is_git_repo": True,
                    "branch": branch,
                    "tracking": tracking,
                    "staged": staged,
                    "unstaged": unstaged,
                    "untracked": untracked,
                    "is_clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
                }
            )

        except subprocess.TimeoutExpired:
            return JSONResponse({"error": "Git command timed out"}, status_code=500)
        except FileNotFoundError:
            return JSONResponse({"is_git_repo": False, "error": "Git not installed"})
        except Exception:
            logger.exception("Git status error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/git/commit", tags=["Git"])
    async def git_commit(request: GitCommitRequest) -> JSONResponse:
        """Create a git commit."""
        try:
            import subprocess

            if request.files:
                for f in request.files:
                    subprocess.run(
                        ["git", "add", f],
                        cwd=server.workspace_root,
                        capture_output=True,
                        timeout=10,
                    )

            message = request.message
            if request.use_ai and not message:
                diff_result = subprocess.run(
                    ["git", "diff", "--cached", "--stat"],
                    cwd=server.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if diff_result.stdout.strip():
                    orchestrator = await server._get_orchestrator()
                    prompt = f"Generate a concise git commit message for these changes:\n{diff_result.stdout[:2000]}"
                    response = await orchestrator.chat(prompt)
                    message = response.get("content", "Update files").strip()
                    message = message.replace("```", "").strip()
                    if message.startswith('"') and message.endswith('"'):
                        message = message[1:-1]

            if not message:
                raise HTTPException(status_code=400, detail="Commit message required")

            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=server.workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return JSONResponse({"success": False, "error": result.stderr or "Commit failed"})

            return JSONResponse({"success": True, "message": message, "output": result.stdout})

        except HTTPException:
            raise
        except Exception:
            logger.exception("Git commit error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/git/log", tags=["Git"])
    async def git_log(limit: int = Query(20, ge=1, le=100)) -> JSONResponse:
        """Get git commit log."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"-{limit}",
                    "--pretty=format:%H|%an|%ae|%ar|%s",
                ],
                cwd=server.workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return JSONResponse({"error": result.stderr}, status_code=500)

            commits = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 4)
                    if len(parts) >= 5:
                        commits.append(
                            {
                                "hash": parts[0],
                                "author": parts[1],
                                "email": parts[2],
                                "relative_date": parts[3],
                                "message": parts[4],
                            }
                        )

            return JSONResponse({"commits": commits})

        except Exception:
            logger.exception("Git log error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/git/diff", tags=["Git"])
    async def git_diff(
        staged: bool = Query(False),
        file: Optional[str] = Query(None),
    ) -> JSONResponse:
        """Get git diff."""
        try:
            import subprocess

            cmd = ["git", "diff"]
            if staged:
                cmd.append("--cached")
            if file:
                cmd.append("--")
                cmd.append(file)

            result = subprocess.run(
                cmd,
                cwd=server.workspace_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            return JSONResponse(
                {
                    "diff": result.stdout[:50000],
                    "truncated": len(result.stdout) > 50000,
                }
            )

        except Exception:
            logger.exception("Git diff error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    return router
