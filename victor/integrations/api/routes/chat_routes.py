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

"""Chat & Completions routes: /chat, /chat/stream, /completions."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from victor.core.context import bind_attribution
from victor.framework.wire_events import stream_sse
from victor.integrations.api.fastapi_server import (
    ChatRequest,
    ChatResponse,
    CompletionRequest,
    CompletionResponse,
    ResumeRequest,
    _new_chat_request_id,
)
from victor.observability.request_correlation import request_correlation_id

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create chat / completions routes bound to *server*."""
    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(request: ChatRequest, response: Response, http_request: Request) -> ChatResponse:
        """Chat endpoint (non-streaming)."""
        # FEP-0020 attribution join: the client_id resolved at the auth seam is
        # bound to the execution context so downstream cost/usage records carry
        # the authenticated subject (None when auth is not configured).
        client_id = await server._verify_api_key(http_request)
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        request_id = _new_chat_request_id()
        response.headers["X-Victor-Request-Id"] = request_id
        client = await server._get_victor_client()
        with request_correlation_id(request_id), bind_attribution(subject_id=client_id):
            chat_result = await client.chat(request.messages[-1].content)

        content = getattr(chat_result, "content", None) or ""
        tool_calls = getattr(chat_result, "tool_calls", None) or []
        status = getattr(chat_result, "status", "ok") or "ok"

        # FEP-0029: the turn durably paused on a policy ASK — surface the resume token so the
        # caller can approve later via /chat/resume. 202 Accepted signals "not final".
        if status == "awaiting_approval":
            response.status_code = 202
            return ChatResponse(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                status=status,
                run_id=getattr(chat_result, "run_id", None),
                approval_request=getattr(chat_result, "approval_request", None),
            )

        return ChatResponse(role="assistant", content=content, tool_calls=tool_calls)

    @router.post("/chat/resume", response_model=ChatResponse, tags=["Chat"])
    async def chat_resume(
        request: ResumeRequest, response: Response, http_request: Request
    ) -> ChatResponse:
        """Resume a durably-paused chat turn with a human approval decision (FEP-0029)."""
        from victor.framework.approval_pause import ApprovalDecision

        client_id = await server._verify_api_key(http_request)
        request_id = _new_chat_request_id()
        response.headers["X-Victor-Request-Id"] = request_id
        client = await server._get_victor_client()
        resume = getattr(client, "resume", None)
        if resume is None:
            raise HTTPException(status_code=501, detail="Resume is not supported by this server")

        decision = ApprovalDecision(
            approved=request.approved,
            response=request.response,
            responder=request.responder,
        )
        try:
            with request_correlation_id(request_id), bind_attribution(subject_id=client_id):
                result = await resume(request.run_id, decision)
        except ValueError as exc:
            # Unknown or already-resumed run_id.
            raise HTTPException(status_code=404, detail=str(exc))

        content = getattr(result, "content", None) or ""
        tool_calls = getattr(result, "tool_calls", None) or []
        status = getattr(result, "status", "ok") or "ok"
        if status == "awaiting_approval":
            # A chained pause (a second ASK during the continuation) — surface the new run_id.
            response.status_code = 202
        return ChatResponse(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            status=status,
            run_id=getattr(result, "run_id", None),
            approval_request=getattr(result, "approval_request", None),
        )

    @router.post("/chat/stream", tags=["Chat"])
    async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
        """Streaming chat endpoint (Server-Sent Events)."""
        # Authenticate before streaming starts so an invalid key is a plain 401;
        # the identity is bound inside the generator because the response body
        # is produced after this handler returns (FEP-0020 attribution join).
        client_id = await server._verify_api_key(http_request)
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        request_id = _new_chat_request_id()

        async def event_generator() -> AsyncIterator[str]:
            try:
                client = await server._get_victor_client()
                yield f'data: {json.dumps({"type": "request", "request_id": request_id})}\n\n'

                with request_correlation_id(request_id), bind_attribution(subject_id=client_id):
                    async for chunk in client.stream_chat(request.messages[-1].content):
                        if hasattr(chunk, "content") or hasattr(chunk, "tool_calls"):
                            content = getattr(chunk, "content", "")
                            tool_calls = getattr(chunk, "tool_calls", None)
                            if content:
                                event = {
                                    "type": "content",
                                    "content": content,
                                    "request_id": request_id,
                                }
                            elif tool_calls:
                                event = {
                                    "type": "tool_call",
                                    "tool_call": tool_calls,
                                    "request_id": request_id,
                                }
                            else:
                                continue
                        else:
                            event = chunk
                            if isinstance(event, dict):
                                event.setdefault("request_id", request_id)

                        yield f"data: {json.dumps(event)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Stream chat error")
                error_event = {
                    "type": "error",
                    "message": str(e),
                    "request_id": request_id,
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Victor-Request-Id": request_id,
            },
        )

    @router.post("/chat/wire-stream", tags=["Chat"])
    async def chat_wire_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
        """Streaming chat endpoint emitting the **v1 wire event contract** as SSE.

        Unlike :meth:`chat_stream` (whose ``type: content|tool_call`` payloads are a
        lossy legacy shape), this endpoint frames :mod:`victor.framework.wire_events`
        verbatim — the same cross-surface schema the TUI wire timeline consumes — so
        remote UIs get reasoning deltas, tool results, member lanes and terminal
        ``stream_end`` semantics with full fidelity (VAS-10/VAS-14 seam). Authentication
        and the FEP-0020 attribution join mirror ``chat_stream``.
        """
        client_id = await server._verify_api_key(http_request)
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        request_id = _new_chat_request_id()

        async def event_generator() -> AsyncIterator[str]:
            try:
                client = await server._get_victor_client()
                yield (f'data: {json.dumps({"event": "request", "request_id": request_id})}\n\n')
                with request_correlation_id(request_id), bind_attribution(subject_id=client_id):
                    async for sse_record in stream_sse(client, request.messages[-1].content):
                        yield sse_record
            except Exception as e:
                logger.exception("Wire-stream chat error")
                error_event = {"event": "error", "message": str(e), "request_id": request_id}
                yield f"data: {json.dumps(error_event)}\n\n"
                # The wire contract terminates EVERY stream with exactly one
                # stream_end — including failure paths (never truncate silently).
                yield f'data: {json.dumps({"event": "stream_end"})}\n\n'

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Victor-Request-Id": request_id,
            },
        )

    @router.post("/completions", response_model=CompletionResponse, tags=["Completions"])
    async def completions(request: CompletionRequest) -> CompletionResponse:
        """Get fast code completions with FIM support."""
        start_time = time.perf_counter()

        if not request.prompt:
            return CompletionResponse(completions=[], latency_ms=0.0)

        try:
            orchestrator = await server._get_orchestrator()
            provider = orchestrator.provider_manager.current_provider

            file_info = f" ({request.language})" if request.language else ""
            if request.file:
                file_info = f" in {request.file}{file_info}"

            if request.suffix:
                completion_prompt = f"""Complete the code at <FILL>. Only output the completion, nothing else.

{request.context or ''}

{request.prompt}<FILL>{request.suffix}"""
            else:
                completion_prompt = f"""Complete this {request.language or 'code'}{file_info}. Only output the completion.

{request.context or ''}

{request.prompt}"""

            stop_sequences = request.stop_sequences or [
                "\n\n",
                "\ndef ",
                "\nclass ",
                "\nfunction ",
                "\n//",
                "\n#",
            ]

            messages = [{"role": "user", "content": completion_prompt}]
            response = await provider.chat(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=stop_sequences,
            )

            content = getattr(response, "content", "") or ""
            completion = content.strip()
            if "\n\nExplanation:" in completion:
                completion = completion.split("\n\nExplanation:")[0]
            if "\n\nNote:" in completion:
                completion = completion.split("\n\nNote:")[0]

            latency_ms = (time.perf_counter() - start_time) * 1000

            return CompletionResponse(
                completions=[completion] if completion else [],
                latency_ms=round(latency_ms, 2),
            )

        except Exception as e:
            logger.exception("Completions error")
            latency_ms = (time.perf_counter() - start_time) * 1000
            return CompletionResponse(
                completions=[],
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )

    return router
