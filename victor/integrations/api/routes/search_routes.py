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
            import subprocess

            cmd = ["rg", "--json", "-n"]
            if not request.case_sensitive:
                cmd.append("-i")
            if not request.regex:
                cmd.append("-F")
            if request.file_pattern != "*":
                cmd.extend(["-g", request.file_pattern])
            cmd.append(request.query)
            cmd.append(server.workspace_root)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

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

        except Exception as e:
            logger.exception("Code search error")
            return SearchResponse(results=[], error=str(e))

    return router
