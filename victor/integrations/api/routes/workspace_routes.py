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

"""Workspace routes: /workspace/overview, /workspace/metrics, /workspace/security, /workspace/dependencies."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create workspace routes bound to *server*."""
    router = APIRouter(tags=["Workspace"])

    @router.get("/workspace/overview")
    async def workspace_overview(depth: int = Query(3, ge=1, le=10)) -> JSONResponse:
        """Get workspace structure overview."""
        try:
            import os

            root = Path(server.workspace_root)
            overview: Dict[str, Any] = {
                "root": str(root),
                "name": root.name,
                "file_counts": {},
                "total_files": 0,
                "total_size": 0,
            }

            exclude_dirs = {
                ".git",
                "node_modules",
                "__pycache__",
                ".venv",
                "venv",
                ".victor",
            }

            def scan_dir(path: Path, d: int = 0) -> Dict[str, Any]:
                if d > depth:
                    return {"name": path.name, "type": "directory", "truncated": True}

                result: Dict[str, Any] = {
                    "name": path.name,
                    "path": str(path.relative_to(root)),
                    "type": "directory",
                    "children": [],
                }

                try:
                    for entry in sorted(
                        path.iterdir(),
                        key=lambda x: (not x.is_dir(), x.name.lower()),
                    ):
                        if entry.name.startswith(".") and entry.name not in {
                            ".github",
                            ".vscode",
                        }:
                            continue
                        if entry.name in exclude_dirs:
                            continue

                        if entry.is_dir():
                            result["children"].append(scan_dir(entry, d + 1))
                        else:
                            ext = entry.suffix.lower()
                            overview["file_counts"][ext] = overview["file_counts"].get(ext, 0) + 1
                            overview["total_files"] += 1
                            try:
                                overview["total_size"] += entry.stat().st_size
                            except OSError:
                                pass

                            if d <= 1:
                                result["children"].append(
                                    {
                                        "name": entry.name,
                                        "path": str(entry.relative_to(root)),
                                        "type": "file",
                                        "extension": ext,
                                    }
                                )
                except PermissionError:
                    result["error"] = "Permission denied"

                return result

            overview["tree"] = scan_dir(root)

            return JSONResponse(overview)

        except Exception:
            logger.exception("Workspace overview error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/workspace/metrics")
    async def workspace_metrics() -> JSONResponse:
        """Get code metrics for the workspace."""
        try:
            orchestrator = await server._get_orchestrator()

            try:
                tool_result = await orchestrator.execute_tool("metrics", path=server.workspace_root)
                if tool_result.success:
                    return JSONResponse(tool_result.data)
            except Exception:
                pass

            root = Path(server.workspace_root)
            metrics: Dict[str, Any] = {
                "lines_of_code": 0,
                "files_by_type": {},
                "largest_files": [],
            }

            code_extensions = {
                ".py",
                ".ts",
                ".js",
                ".tsx",
                ".jsx",
                ".java",
                ".go",
                ".rs",
                ".cpp",
                ".c",
                ".h",
            }
            file_sizes = []

            for path in root.rglob("*"):
                if path.is_file() and not any(
                    p.startswith(".") for p in path.parts[len(root.parts) :]
                ):
                    ext = path.suffix.lower()
                    if ext in code_extensions:
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                lines = len(f.readlines())
                                metrics["lines_of_code"] += lines
                                metrics["files_by_type"][ext] = (
                                    metrics["files_by_type"].get(ext, 0) + 1
                                )
                                file_sizes.append(
                                    {
                                        "path": str(path.relative_to(root)),
                                        "lines": lines,
                                        "size": path.stat().st_size,
                                    }
                                )
                        except Exception:
                            pass

            file_sizes.sort(key=lambda x: x["lines"], reverse=True)
            metrics["largest_files"] = file_sizes[:10]

            return JSONResponse(metrics)

        except Exception:
            logger.exception("Workspace metrics error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/workspace/security")
    async def workspace_security() -> JSONResponse:
        """Get security scan results."""
        try:
            orchestrator = await server._get_orchestrator()

            try:
                tool_result = await orchestrator.execute_tool(
                    "scan",
                    path=server.workspace_root,
                    scan_type="secrets",
                )
                if tool_result.success:
                    return JSONResponse({"scan_completed": True, "results": tool_result.data})
            except Exception:
                pass

            root = Path(server.workspace_root)
            findings = []

            secret_patterns = [
                (
                    r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?[\w-]{20,}',
                    "API Key",
                ),
                (
                    r'(?i)(secret|password|passwd|pwd)\s*[:=]\s*["\'][^"\']{8,}',
                    "Secret/Password",
                ),
                (r"(?i)bearer\s+[\w-]{20,}", "Bearer Token"),
                (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API Key"),
                (r"ghp_[a-zA-Z0-9]{36}", "GitHub Token"),
                (r"AKIA[A-Z0-9]{16}", "AWS Access Key"),
            ]

            code_extensions = {
                ".py",
                ".ts",
                ".js",
                ".json",
                ".yaml",
                ".yml",
                ".env",
                ".sh",
            }

            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in code_extensions:
                    if any(
                        p.startswith(".") or p in {"node_modules", "__pycache__"}
                        for p in path.parts
                    ):
                        continue

                    try:
                        if path.stat().st_size > 1_000_000:
                            continue
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        for pattern, finding_type in secret_patterns:
                            for match in re.finditer(pattern, content):
                                line_num = content[: match.start()].count("\n") + 1
                                findings.append(
                                    {
                                        "file": str(path.relative_to(root)),
                                        "line": line_num,
                                        "type": finding_type,
                                        "severity": "high",
                                        "snippet": "[REDACTED]",
                                    }
                                )
                    except Exception:
                        pass

            return JSONResponse(
                {
                    "scan_completed": True,
                    "findings": findings[:50],
                    "total_findings": len(findings),
                    "severity_counts": {
                        "high": len([f for f in findings if f["severity"] == "high"]),
                        "medium": 0,
                        "low": 0,
                    },
                }
            )

        except Exception:
            logger.exception("Workspace security error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/workspace/dependencies")
    async def workspace_dependencies() -> JSONResponse:
        """Get dependency information."""
        try:
            root = Path(server.workspace_root)
            dependencies: Dict[str, Any] = {}

            for req_file in ["requirements.txt", "pyproject.toml", "setup.py"]:
                req_path = root / req_file
                if req_path.exists():
                    if req_file == "requirements.txt":
                        deps = []
                        for line in req_path.read_text().splitlines():
                            line = line.strip()
                            if line and not line.startswith("#"):
                                deps.append(line.split("==")[0].split(">=")[0].split("<")[0])
                        dependencies["python"] = {
                            "file": req_file,
                            "count": len(deps),
                            "packages": deps[:20],
                        }
                    break

            pkg_json = root / "package.json"
            if pkg_json.exists():
                try:
                    pkg_data = json.loads(pkg_json.read_text())
                    deps = list(pkg_data.get("dependencies", {}).keys())
                    dev_deps = list(pkg_data.get("devDependencies", {}).keys())
                    dependencies["node"] = {
                        "file": "package.json",
                        "dependencies": len(deps),
                        "devDependencies": len(dev_deps),
                        "packages": deps[:20],
                    }
                except json.JSONDecodeError:
                    pass

            cargo_toml = root / "Cargo.toml"
            if cargo_toml.exists():
                dependencies["rust"] = {"file": "Cargo.toml", "exists": True}

            go_mod = root / "go.mod"
            if go_mod.exists():
                dependencies["go"] = {"file": "go.mod", "exists": True}

            return JSONResponse({"workspace": str(root), "dependencies": dependencies})

        except Exception:
            logger.exception("Workspace dependencies error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    return router
