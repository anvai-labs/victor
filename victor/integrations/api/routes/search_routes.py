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

"""Search routes: /search/semantic, /search/code."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter

from victor.integrations.search_types import CodeSearchResult
from victor.integrations.api.fastapi_server import (
    APISearchResult,
    CodeSearchRequest,
    SearchRequest,
    SearchResponse,
)

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


class SearchPolicyRefusal(RuntimeError):
    """A required sandbox policy cannot enforce a search launch.

    The route must return a structured refusal without spawning. This is the
    same fail-closed direct-launch boundary used by terminal and git routes.
    """


def _admit_search_policy():
    """Return the enclosing admitted policy or copy current settings once."""
    from victor.runtime.policy_snapshot import (
        RunPolicySnapshot,
        current_run_policy_snapshot,
    )

    admitted_policy = current_run_policy_snapshot()
    if admitted_policy is not None:
        return admitted_policy
    try:
        from victor.config.settings import get_settings

        return RunPolicySnapshot.from_settings(get_settings())
    except Exception as exc:
        raise SearchPolicyRefusal(
            f"Sandbox policy could not be established ({exc}); refusing execution (fail-closed)."
        ) from exc


def _search_spawn(server: "VictorFastAPIServer", argv: list[str], **kwargs):
    """Run a search argv through the canonical required-launch decision.

    The launch goes through the shared :func:`prepare_launch` /
    :func:`run_launch` seam so a wrapped argv carries its backend-owned
    descriptors (bwrap's seccomp filter) and those descriptors are closed
    after spawn — forwarding the argv alone left bwrap without its filter
    and leaked one open descriptor per launch in the parent.
    """
    from victor.runtime.policy_snapshot import run_policy_context
    from victor.tools.sandbox import prepare_launch, resolve_required_launch, run_launch

    admitted_policy = _admit_search_policy()
    decision = resolve_required_launch(admitted_policy.to_sandbox_settings())
    if decision.refuse:
        raise SearchPolicyRefusal(decision.reason)
    backend = decision.backend if decision.required_active else None
    launch = prepare_launch(backend, list(argv), server.workspace_root)

    with run_policy_context(admitted_policy):
        return run_launch(launch, **kwargs)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create search routes bound to *server*."""
    router = APIRouter(tags=["Search"])

    @router.post("/search/semantic", response_model=SearchResponse)
    async def semantic_search(request: SearchRequest) -> SearchResponse:
        """Semantic code search."""
        if not request.query:
            return SearchResponse(results=[])

        try:
            orchestrator = await server._get_orchestrator()
            tool_result = await orchestrator.execute_tool(
                "semantic_code_search",
                query=request.query,
                max_results=request.max_results,
            )

            if tool_result.success:
                matches = tool_result.data.get("matches", [])
                code_results = [
                    CodeSearchResult(
                        file=r.get("file", ""),
                        line=r.get("line", 0),
                        content=r.get("content", ""),
                        score=r.get("score", 0.0),
                        context=r.get("context", ""),
                    )
                    for r in matches
                ]
                results = [APISearchResult.from_code_result(r) for r in code_results]
                return SearchResponse(results=results)
            return SearchResponse(results=[])

        except Exception as e:
            logger.exception("Semantic search error")
            return SearchResponse(results=[], error=str(e))

    @router.post("/search/code", response_model=SearchResponse)
    async def code_search(request: CodeSearchRequest) -> SearchResponse:
        """Code search (regex/literal)."""
        if not request.query:
            return SearchResponse(results=[])

        try:
            cmd = ["rg", "--json", "-n"]
            if not request.case_sensitive:
                cmd.append("-i")
            if not request.regex:
                cmd.append("-F")
            if request.file_pattern != "*":
                cmd.extend(["-g", request.file_pattern])
            cmd.extend(["-e", request.query, "--", server.workspace_root])

            result = _search_spawn(
                server,
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            code_results = []
            for line in result.stdout.splitlines():
                try:
                    match = json.loads(line)
                    if match.get("type") == "match":
                        data = match.get("data", {})
                        code_results.append(
                            CodeSearchResult(
                                file=data.get("path", {}).get("text", ""),
                                line=data.get("line_number", 0),
                                content=data.get("lines", {}).get("text", "").strip(),
                                score=1.0,
                            )
                        )
                except json.JSONDecodeError:
                    continue

            results = [APISearchResult.from_code_result(r) for r in code_results[:50]]
            return SearchResponse(results=results)

        except SearchPolicyRefusal as exc:
            return SearchResponse(results=[], error=str(exc))
        except Exception as e:
            logger.exception("Code search error")
            return SearchResponse(results=[], error=str(e))

    return router
