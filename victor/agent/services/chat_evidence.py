"""Bounded evidence capture helpers for the canonical chat service."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional


class ChatEvidenceMixin:
    """Capture task reports and bounded conversation evidence outside the chat hotspot."""

    _context: Any
    _logger: logging.Logger
    _task_report_start_handler: Optional[Callable[..., Any]]
    _task_report_finish_handler: Optional[Callable[..., Any]]

    def get_last_task_report(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the most recently completed canonical task report."""
        report = getattr(self, "_last_task_report", None)
        return dict(report) if isinstance(report, Mapping) else None

    def get_conversation_trace(self) -> Dict[str, Any]:
        """Return a bounded, JSON-safe trace without raw responses or images."""
        try:
            messages = list(self._context.get_messages())[-50:]
        except Exception:
            messages = []

        serialized_messages: List[Dict[str, str]] = []
        serialized_tool_calls: List[Dict[str, str]] = []
        turns = 0
        for message in messages:
            if isinstance(message, Mapping):
                role = str(message.get("role", "") or "")
                content = str(message.get("content", "") or "")
                tool_calls = message.get("tool_calls")
            else:
                role = str(getattr(message, "role", "") or "")
                content = str(getattr(message, "content", "") or "")
                tool_calls = getattr(message, "tool_calls", None)

            serialized_messages.append({"role": role, "content": content[:500]})
            if role == "user":
                turns += 1
            self._append_tool_calls(serialized_tool_calls, tool_calls)

        return {
            "messages": serialized_messages,
            "tool_calls": serialized_tool_calls[-100:],
            "turns": turns,
        }

    @staticmethod
    def _append_tool_calls(serialized: List[Dict[str, str]], tool_calls: Any) -> None:
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if len(serialized) >= 100:
                break
            if isinstance(tool_call, Mapping):
                function = tool_call.get("function")
                if isinstance(function, Mapping):
                    name = function.get("name", "")
                    arguments = function.get("arguments", "")
                else:
                    name = tool_call.get("name", "")
                    arguments = tool_call.get("arguments", "")
            else:
                function = getattr(tool_call, "function", None)
                name = getattr(function, "name", getattr(tool_call, "name", ""))
                arguments = getattr(
                    function,
                    "arguments",
                    getattr(tool_call, "arguments", ""),
                )
            serialized.append({"name": str(name or ""), "arguments": str(arguments or "")[:500]})

    async def _run_optional_callback(
        self,
        callback: Optional[Callable[..., Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke sync or async runtime callbacks without breaking the chat path."""
        if callback is None:
            return None
        try:
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            self._logger.debug("Runtime callback failed: %s", exc)
            return None

    async def _start_task_report(
        self,
        user_message: str,
        *,
        stream: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify the canonical runtime that a task report should begin."""
        self._last_task_report = None
        await self._run_optional_callback(
            self._task_report_start_handler,
            user_message,
            stream=stream,
            metadata=metadata or {},
        )

    async def _finish_task_report(
        self,
        success: bool,
        *,
        user_message: str,
        stream: bool,
        response: Optional[Any] = None,
        error: Optional[BaseException] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify the canonical runtime that a task report finished."""
        report = await self._run_optional_callback(
            self._task_report_finish_handler,
            success,
            user_message=user_message,
            stream=stream,
            response=response,
            error=error,
            metadata=metadata or {},
        )
        if isinstance(report, Mapping):
            self._last_task_report = dict(report)

    @staticmethod
    def _response_execution_success(response: Optional[Any]) -> bool:
        """Return whether a response represents successful task execution."""
        if response is None:
            return True
        metadata = getattr(response, "metadata", None)
        if isinstance(response, dict):
            metadata = response.get("metadata", metadata)
        if isinstance(metadata, dict) and metadata.get("agentic_loop_success") is False:
            return False
        return True
