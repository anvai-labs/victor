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

"""OpenAI-compatible utilities for Victor providers.

This module provides shared conversion functions for providers that use
OpenAI-compatible APIs (OpenAI, xAI, LMStudio, vLLM, Ollama).

These utilities help reduce code duplication across provider implementations.
"""

import hashlib
from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
import re
import uuid
from contextvars import ContextVar
from typing import Any, Dict, FrozenSet, List, Literal, Mapping, Optional, Tuple

import httpx

from victor.providers.base import Message, ToolDefinition

logger = logging.getLogger(__name__)

OpenAIChatMessageRole = Literal[
    "developer",
    "system",
    "user",
    "assistant",
    "tool",
    "function",
]
OPENAI_CHAT_MESSAGE_ROLES: FrozenSet[str] = frozenset(
    {"developer", "system", "user", "assistant", "tool", "function"}
)


def _validate_openai_chat_role(role: Any, *, index: Optional[int] = None) -> str:
    """Return a supported Chat Completions role or fail before provider I/O."""
    location = "message role" if index is None else f"messages[{index}].role"
    if not isinstance(role, str) or role not in OPENAI_CHAT_MESSAGE_ROLES:
        supported = ", ".join(sorted(OPENAI_CHAT_MESSAGE_ROLES))
        raise ValueError(f"{location} must be one of: {supported}; got {role!r}")
    return role


# Last-computed stability fingerprints for the serialized tools arrays.
# Exposed for observability/diagnostics so callers (or tests) can compare the
# serialized payload across turns without re-serializing. A changed fingerprint
# between turns means the `tools[]` field changed and any prefix cache keyed on
# it will miss.
_LAST_TOOLS_FINGERPRINT_OPENAI: Optional[str] = None
_LAST_TOOLS_FINGERPRINT_ANTHROPIC: Optional[str] = None


def _serialized_tools_fingerprint(converted: List[Dict[str, Any]]) -> str:
    """Compute a short stability hash of a serialized tools array.

    Identical ``tools[]`` arrays (same names, descriptions, and JSON schemas,
    regardless of insertion order) produce identical hashes. This is the core
    first-principles debugging signal for KV/prefix-cache behavior: if the
    fingerprint is stable across turns, the provider can cache the tools
    prefix; if it churns, the prefix cache misses every turn.
    """
    payload = json_dumps(converted, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


_LAST_TOOL_MESSAGE_CLEANUP_STATS: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "victor_openai_compat_last_tool_message_cleanup_stats",
    default=None,
)


def clear_last_tool_message_cleanup_stats() -> None:
    """Clear per-request tool-message cleanup diagnostics."""
    _LAST_TOOL_MESSAGE_CLEANUP_STATS.set(None)


def get_last_tool_message_cleanup_stats() -> Dict[str, Any]:
    """Return the latest tool-message cleanup diagnostics for this task context."""
    stats = _LAST_TOOL_MESSAGE_CLEANUP_STATS.get()
    if isinstance(stats, dict):
        return dict(stats)
    return {
        "history_repaired": False,
        "stripped_assistant_tool_calls": 0,
        "removed_orphaned_tool_responses": 0,
        "skipped_tool_messages_without_id": 0,
    }


def consume_last_tool_message_cleanup_stats() -> Dict[str, Any]:
    """Return and clear the latest tool-message cleanup diagnostics."""
    stats = get_last_tool_message_cleanup_stats()
    clear_last_tool_message_cleanup_stats()
    return stats


def convert_tools_to_openai_format(tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
    """Convert standard tools to OpenAI function calling format.

    This format is used by OpenAI, xAI, LMStudio, vLLM, and Ollama.

    Args:
        tools: Standard tool definitions

    Returns:
        OpenAI-formatted tools list
    """
    result = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]
    # Log serialized tool schemas at INFO so the payload is visible per-call for
    # first-principles debugging. The fingerprint is a stability hash of the
    # *serialized* array: stable across turns => prefix cache eligible; churning
    # => prefix cache misses every turn.
    tool_sigs = []
    for t in result:
        fn = t["function"]
        props = fn.get("parameters", {}).get("properties", {})
        tool_sigs.append(f"{fn['name']}({list(props.keys())})")
    fingerprint = _serialized_tools_fingerprint(result)
    global _LAST_TOOLS_FINGERPRINT_OPENAI
    _LAST_TOOLS_FINGERPRINT_OPENAI = fingerprint
    logger.info(
        "[ToolSchemas→LLM] OpenAI format: %d tools (fingerprint=%s): %s",
        len(result),
        fingerprint,
        ", ".join(tool_sigs),
    )
    return result


def convert_tools_to_anthropic_format(
    tools: List[ToolDefinition],
) -> List[Dict[str, Any]]:
    """Convert standard tools to Anthropic format.

    Args:
        tools: Standard tool definitions

    Returns:
        Anthropic-formatted tools list
    """
    result = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]
    # Log serialized tool schemas at INFO for first-principles debugging. The
    # fingerprint reflects tool content/schema only (cache_control markers are
    # added later by the provider and are intentionally NOT part of this hash).
    tool_sigs = []
    for t in result:
        props = t.get("input_schema", {}).get("properties", {})
        tool_sigs.append(f"{t['name']}({list(props.keys())})")
    fingerprint = _serialized_tools_fingerprint(result)
    global _LAST_TOOLS_FINGERPRINT_ANTHROPIC
    _LAST_TOOLS_FINGERPRINT_ANTHROPIC = fingerprint
    logger.info(
        "[ToolSchemas→LLM] Anthropic format: %d tools (fingerprint=%s): %s",
        len(result),
        fingerprint,
        ", ".join(tool_sigs),
    )
    return result


def convert_messages_to_openai_format(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert standard messages to OpenAI format.

    Args:
        messages: Standard message objects

    Returns:
        OpenAI-formatted messages list
    """
    result = []
    for index, msg in enumerate(messages):
        role = _validate_openai_chat_role(msg.role, index=index)
        formatted: Dict[str, Any] = {
            "role": role,
            "content": msg.content or "",
        }

        # Add tool calls if present
        if msg.tool_calls:
            formatted["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            json_dumps(tc.get("arguments", {}))
                            if isinstance(tc.get("arguments"), dict)
                            else tc.get("arguments", "{}")
                        ),
                    },
                }
                for tc in msg.tool_calls
            ]

        # Add tool_call_id for tool responses
        if role == "tool":
            if not msg.tool_call_id:
                raise ValueError(f"messages[{index}] with role=tool requires tool_call_id")
            formatted["tool_call_id"] = msg.tool_call_id

        # Add name if present (for function responses)
        if hasattr(msg, "name") and msg.name:
            formatted["name"] = msg.name

        if role == "function" and not msg.name:
            raise ValueError(f"messages[{index}] with role=function requires name")

        result.append(formatted)

    return result


def parse_openai_tool_calls(
    tool_calls_data: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Parse OpenAI-format tool calls into standard dictionaries.

    Args:
        tool_calls_data: Raw tool calls from OpenAI-compatible API

    Returns:
        List of tool call dictionaries, or None if no tool calls
    """
    if not tool_calls_data:
        return None

    tool_calls = []
    for tc in tool_calls_data:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")

        # Parse arguments
        try:
            arguments = json_loads(args_str) if isinstance(args_str, str) else args_str
        except JSONDecodeError:
            arguments = {"raw": args_str}

        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": arguments,
            }
        )

    return tool_calls if tool_calls else None


def parse_openai_stream_chunk(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse OpenAI-compatible streaming chunk.

    Args:
        chunk_data: Raw chunk data

    Returns:
        Parsed chunk with content, tool_calls, finish_reason
    """
    result: Dict[str, Any] = {
        "content": None,
        "tool_calls": None,
        "finish_reason": None,
    }

    choices = chunk_data.get("choices", [])
    if not choices:
        return result

    choice = choices[0]
    delta = choice.get("delta", {})

    # Extract content
    if "content" in delta:
        result["content"] = delta["content"]

    # Extract tool calls delta
    if "tool_calls" in delta:
        result["tool_calls"] = delta["tool_calls"]

    # Extract finish reason
    if "finish_reason" in choice and choice["finish_reason"]:
        result["finish_reason"] = choice["finish_reason"]

    return result


def fix_orphaned_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_calls/responses after compaction.

    OpenAI-compatible APIs require strict pairing:
    - Every ``tool_calls[].id`` in an assistant message must have a matching
      ``role=tool`` response with the same ``tool_call_id``.
    - Every ``role=tool`` message must have a corresponding assistant
      ``tool_calls`` entry.

    After context compaction, either side may be missing. This function:
    1. Strips ``tool_calls`` from assistant messages whose responses are gone.
    2. Removes ``role=tool`` messages whose assistant ``tool_calls`` are gone.

    Shared utility for all OpenAI-compatible providers (Z.AI, DeepSeek,
    OpenAI, xAI, Groq, LMStudio, etc.).

    Args:
        messages: List of formatted message dicts (already converted to
                  OpenAI format with ``role``, ``content``, ``tool_calls``,
                  ``tool_call_id`` keys).

    Returns:
        Cleaned message list with consistent tool_calls/response pairing.
    """
    repair_id = uuid.uuid4().hex[:8]

    # Collect all tool_call IDs present in tool responses
    present_response_ids = {
        m.get("tool_call_id") for m in messages if m.get("role") == "tool" and m.get("tool_call_id")
    }

    # Strip assistant tool_calls whose responses were compacted away
    stripped_calls_count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tc_ids = {tc["id"] for tc in msg["tool_calls"] if "id" in tc}
            missing_responses = tc_ids - present_response_ids
            if not tc_ids.issubset(present_response_ids):
                logger.debug(
                    "[fix_orphaned_tool_messages:%s] Stripping tool_calls from assistant "
                    "message: reason=missing_tool_responses ids=%s",
                    repair_id,
                    missing_responses,
                )
                del msg["tool_calls"]
                if msg.get("content") is None:
                    msg["content"] = ""
                stripped_calls_count += 1

    if stripped_calls_count:
        logger.info(
            "[fix_orphaned_tool_messages:%s] Stripped tool_calls from %d assistant messages",
            repair_id,
            stripped_calls_count,
        )

    # Re-collect declared tool_call IDs AFTER stripping assistant messages
    # This is critical - we need the IDs that are STILL in the messages after stripping
    declared_tool_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if "id" in tc:
                    declared_tool_call_ids.add(tc["id"])

    logger.debug(
        "[fix_orphaned_tool_messages:%s] declared_tool_call_ids=%s present_response_ids=%s",
        repair_id,
        declared_tool_call_ids,
        present_response_ids,
    )

    # Find orphaned tool responses (responses without matching tool_calls)
    orphaned_responses = present_response_ids - declared_tool_call_ids
    if orphaned_responses:
        logger.debug(
            "[fix_orphaned_tool_messages:%s] Found %d orphaned tool responses "
            "(reason=missing_assistant_tool_call ids=%s)",
            repair_id,
            len(orphaned_responses),
            orphaned_responses,
        )

    # Remove orphaned tool responses whose assistant tool_calls were
    # compacted away
    original_count = len(messages)
    messages = [
        m
        for m in messages
        if not (
            m.get("role") == "tool"
            and m.get("tool_call_id")
            and m["tool_call_id"] not in declared_tool_call_ids
        )
    ]

    removed_count = original_count - len(messages)
    if removed_count:
        logger.info(
            "[fix_orphaned_tool_messages:%s] Removed %d orphaned tool response messages",
            repair_id,
            removed_count,
        )

    stats = {
        "history_repaired": bool(stripped_calls_count or removed_count),
        "stripped_assistant_tool_calls": stripped_calls_count,
        "removed_orphaned_tool_responses": removed_count,
        "repair_id": repair_id,
        "skipped_tool_messages_without_id": 0,
    }
    _LAST_TOOL_MESSAGE_CLEANUP_STATS.set(stats)

    logger.debug(
        "[fix_orphaned_tool_messages:%s] Result: %d messages (was %d), stats=%s",
        repair_id,
        len(messages),
        original_count,
        stats,
    )

    return messages


def _message_field(message: Any, field_name: str, default: Any = None) -> Any:
    """Read a message field from either a provider Message or a raw mapping."""
    if isinstance(message, Mapping):
        return message.get(field_name, default)
    return getattr(message, field_name, default)


def _normalize_tool_call(tool_call: Any) -> Dict[str, Any]:
    """Normalize internal or OpenAI-style tool calls to OpenAI function-call shape."""
    if isinstance(tool_call, Mapping):
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            name = function.get("name", tool_call.get("name", ""))
            arguments = function.get("arguments", tool_call.get("arguments", {}))
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id", "")
        tool_call_type = tool_call.get("type", "function")
    else:
        function = getattr(tool_call, "function", None)
        if isinstance(function, Mapping):
            name = function.get("name", getattr(tool_call, "name", ""))
            arguments = function.get("arguments", getattr(tool_call, "arguments", {}))
        else:
            name = getattr(function, "name", getattr(tool_call, "name", ""))
            arguments = getattr(function, "arguments", getattr(tool_call, "arguments", {}))
        tool_call_id = getattr(tool_call, "id", "")
        tool_call_type = getattr(tool_call, "type", "function")

    if isinstance(arguments, dict):
        arguments = json_dumps(arguments)
    elif arguments is None:
        arguments = "{}"

    return {
        "id": tool_call_id,
        "type": tool_call_type or "function",
        "function": {"name": name or "", "arguments": arguments},
    }


def build_openai_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Build a fully-validated OpenAI-format message list from provider messages.

    Extracted from ZAIProvider._build_request_payload — the most complete message
    serialization in the codebase. Shared across all OpenAI-compatible providers.

    Handles:
    - Message input: accepts both Victor ``Message`` objects and OpenAI-style dicts
    - Assistant messages: serializes tool_calls to OpenAI ``{id, type, function}`` format;
      sets content=None when tool_calls are present (required by GLM/OpenAI spec)
    - Tool response messages: copies tool_call_id; generates a fallback MD5 ID when
      the original is missing or orphaned (prevents 400 errors after context compaction)
    - Second-pass orphaned cleanup via fix_orphaned_tool_messages()

    Args:
        messages: Standard Message objects or OpenAI-style dicts from the conversation history

    Returns:
        Cleaned OpenAI-format message list ready to send to any /v1/chat/completions API
    """
    logger.debug(
        "[build_openai_messages] Input: %d messages",
        len(messages),
    )
    clear_last_tool_message_cleanup_stats()
    formatted: List[Dict[str, Any]] = []
    valid_tool_call_ids: set = set()
    skipped_tool_messages_without_id = 0

    for index, msg in enumerate(messages):
        role = _message_field(msg, "role")
        if not role:
            logger.warning("[build_openai_messages] Message without role - SKIPPING")
            continue
        role = _validate_openai_chat_role(role, index=index)
        content = _message_field(msg, "content", "")
        entry: Dict[str, Any] = {
            "role": role,
            "content": content if content is not None else "",
        }
        name = _message_field(msg, "name")
        if name:
            entry["name"] = name

        if role == "assistant":
            tool_calls = _message_field(msg, "tool_calls")
            if tool_calls:
                openai_tcs = []
                for tc in tool_calls:
                    normalized_tc = _normalize_tool_call(tc)
                    tc_id = normalized_tc.get("id", "")
                    if tc_id:
                        openai_tcs.append(normalized_tc)
                        valid_tool_call_ids.add(tc_id)
                if openai_tcs:
                    entry["tool_calls"] = openai_tcs
                    if not entry["content"]:
                        entry["content"] = None
                logger.debug(
                    "[build_openai_messages] Assistant message with %d tool_calls: %s",
                    len(openai_tcs),
                    [tc.get("id", "") for tc in openai_tcs],
                )

        elif role == "tool":
            tool_call_id = _message_field(msg, "tool_call_id")
            # Skip tool messages without tool_call_id - they would cause 400 errors
            # and fix_orphaned_tool_messages() can't properly clean them up.
            if not tool_call_id:
                logger.warning(
                    "[build_openai_messages] Tool message without tool_call_id - SKIPPING"
                )
                skipped_tool_messages_without_id += 1
                continue
            entry["tool_call_id"] = tool_call_id
            # name field for function responses (required by some providers)
            # DeepSeek and some other providers reject empty content in tool messages.
            # Use a placeholder to avoid 400 errors while maintaining API compatibility.
            if not entry.get("content"):
                entry["content"] = "(no output)"
            logger.debug(
                "[build_openai_messages] Tool message with tool_call_id=%s name=%s",
                tool_call_id,
                entry.get("name", ""),
            )

        elif role == "function" and not name:
            raise ValueError(f"messages[{index}] with role=function requires name")

        formatted.append(entry)

    logger.debug(
        "[build_openai_messages] Before orphan cleanup: %d messages, valid_tool_call_ids=%s",
        len(formatted),
        valid_tool_call_ids,
    )
    result = fix_orphaned_tool_messages(formatted)
    stats = get_last_tool_message_cleanup_stats()
    if skipped_tool_messages_without_id:
        stats["skipped_tool_messages_without_id"] = skipped_tool_messages_without_id
        stats["history_repaired"] = bool(
            stats.get("history_repaired") or skipped_tool_messages_without_id
        )
        _LAST_TOOL_MESSAGE_CLEANUP_STATS.set(stats)
    logger.debug(
        "[build_openai_messages] After orphan cleanup: %d messages (stats=%s)",
        len(result),
        stats,
    )
    return result


def handle_httpx_status_error(
    exc: "httpx.HTTPStatusError",
    provider_name: str,
) -> Exception:
    """Map an httpx.HTTPStatusError to a typed ProviderError subclass.

    Extracted from ZAIProvider — provides richer 400 error messages by parsing
    the JSON error body for the ``error.message`` field. Shared across all
    httpx-based OpenAI-compatible providers.

    Args:
        exc: The HTTPStatusError raised by httpx
        provider_name: Provider name for error messages (e.g., "zai", "xai")

    Returns:
        The mapped exception (caller should raise it).
    """
    import logging

    logger = logging.getLogger(__name__)

    from victor.providers.base import (
        ProviderAuthError,
        ProviderError,
        ProviderRateLimitError,
    )

    status = exc.response.status_code
    raw = ""
    try:
        raw = exc.response.text[:500]
    except Exception:
        pass

    # Log HTTP errors for debugging (especially 400/401/429)
    logger.error(
        "Provider HTTP error: provider=%s status=%d error=%s",
        provider_name,
        status,
        raw[:200],
    )

    if status == 401:
        return ProviderAuthError(
            message=f"Authentication failed: {raw or 'HTTP 401'}",
            provider=provider_name,
            status_code=401,
        )
    if status == 429:
        return ProviderRateLimitError(
            message=f"Rate limit exceeded: {raw or 'HTTP 429'}",
            provider=provider_name,
            status_code=429,
        )
    if status == 400:
        msg = raw
        try:
            body = json_loads(raw)
            msg = body.get("error", {}).get("message", raw)
        except Exception:
            pass
        return ProviderError(
            message=f"{provider_name} request format error: {msg}",
            provider=provider_name,
            status_code=400,
            raw_error=exc,
        )
    return ProviderError(
        message=f"{provider_name} HTTP error {status}: {raw}",
        provider=provider_name,
        status_code=status,
        raw_error=exc,
    )


def accumulate_tool_call_delta(
    delta: Dict[str, Any],
    accumulated: List[Dict[str, Any]],
) -> None:
    """Accumulate a streaming tool-call delta into the running accumulator list.

    Extracted from ZAIProvider._parse_stream_chunk — handles OpenAI SSE streaming
    format where tool call data arrives in fragments across multiple chunks.

    Each entry in ``accumulated`` has shape: ``{id, name, arguments}`` where
    ``arguments`` is a string being built up by concatenation.

    Args:
        delta: The ``delta`` dict from an SSE chunk (``choices[0].delta``)
        accumulated: Mutable list; extended and mutated in place
    """
    tool_call_deltas = delta.get("tool_calls", [])
    for tc_delta in tool_call_deltas:
        idx = tc_delta.get("index", 0)
        while len(accumulated) <= idx:
            accumulated.append({"id": "", "name": "", "arguments": ""})
        if "id" in tc_delta:
            accumulated[idx]["id"] = tc_delta["id"]
        func = tc_delta.get("function", {})
        if "name" in func:
            accumulated[idx]["name"] = func["name"]
        if "arguments" in func:
            accumulated[idx]["arguments"] += func["arguments"]


def extract_thinking_content(response: str) -> Tuple[str, str]:
    """Extract ``<think>...</think>`` reasoning blocks from a response.

    Shared across OpenAI-compatible local providers (vLLM, llama.cpp, LM Studio,
    …) that may surface a model's chain-of-thought inline. Returns the thinking
    text and the response with the think blocks stripped.

    Args:
        response: Raw response text.

    Returns:
        Tuple of ``(thinking_content, main_content)``.
    """
    if not response:
        return ("", "")

    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)
    thinking = "\n".join(matches) if matches else ""
    content = re.sub(think_pattern, "", response, flags=re.DOTALL | re.IGNORECASE).strip()
    return (thinking, content)


def extract_tool_calls_from_content(
    content: str, id_prefix: str = "fallback"
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract tool calls embedded as JSON text when the server didn't parse them.

    Shared fallback for OpenAI-compatible local providers (e.g. vLLM not started
    with ``--enable-auto-tool-choice``, MLX). Recognizes three shapes:
    ``` ```json {...}``` ```, ``<TOOL_OUTPUT>{...}</TOOL_OUTPUT>``, and a bare
    inline ``{"name": ..., "arguments": {...}}``. Planning-style JSON (which also
    has ``name`` but different structure) is rejected via keyword heuristics.

    Args:
        content: Response content that may contain tool calls.
        id_prefix: Synthetic tool-call id prefix (e.g. ``"mlx"`` -> ``mlx_0``).

    Returns:
        Tuple of ``(parsed_tool_calls, remaining_content)``.
    """
    tool_calls: List[Dict[str, Any]] = []
    remaining = content

    # Planning keywords to avoid false positives
    planning_keywords = {"complexity", "steps", "desc", "duration"}

    # Pattern 1: JSON code block with tool call
    json_block_pattern = r"```json\s*\n?\s*(\{[^`]*\"name\"\s*:\s*\"[^\"]+\"[^`]*\})\s*\n?```"
    matches = re.findall(json_block_pattern, content, re.DOTALL)
    for match in matches:
        try:
            data = json_loads(match)
            if "name" in data:
                arguments = data.get("arguments", {})
                if isinstance(arguments, dict) and not any(
                    key in arguments for key in planning_keywords
                ):
                    tool_calls.append(
                        {
                            "id": f"{id_prefix}_{len(tool_calls)}",
                            "name": data.get("name", ""),
                            "arguments": arguments,
                        }
                    )
                    remaining = remaining.replace(f"```json\n{match}\n```", "").strip()
                    remaining = remaining.replace(f"```json{match}```", "").strip()
        except JSONDecodeError:
            pass

    # Pattern 2: <TOOL_OUTPUT> tags
    tool_output_pattern = r"<TOOL_OUTPUT>\s*(\{.*?\})\s*</TOOL_OUTPUT>"
    matches = re.findall(tool_output_pattern, content, re.DOTALL)
    for match in matches:
        try:
            data = json_loads(match)
            if "name" in data:
                arguments = data.get("arguments", {})
                if isinstance(arguments, dict) and not any(
                    key in arguments for key in planning_keywords
                ):
                    tool_calls.append(
                        {
                            "id": f"{id_prefix}_{len(tool_calls)}",
                            "name": data.get("name", ""),
                            "arguments": arguments,
                        }
                    )
                    remaining = re.sub(
                        r"<TOOL_OUTPUT>\s*" + re.escape(match) + r"\s*</TOOL_OUTPUT>",
                        "",
                        remaining,
                    )
        except JSONDecodeError:
            pass

    # Pattern 3: Inline JSON (for simple cases)
    if not tool_calls and content.strip().startswith("{") and "name" in content:
        try:
            data = json_loads(content.strip())
            if "name" in data:
                arguments = data.get("arguments", {})
                if isinstance(arguments, dict) and not any(
                    key in arguments for key in planning_keywords
                ):
                    tool_calls.append(
                        {
                            "id": f"{id_prefix}_0",
                            "name": data.get("name", ""),
                            "arguments": arguments,
                        }
                    )
                    remaining = ""
        except JSONDecodeError:
            pass

    return tool_calls, remaining.strip()
