# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Message persistence implementation for the canonical ChatService."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("victor.agent.services.chat_service")


class ChatPersistenceMixin:
    """Keep persistence ownership on ChatService without growing its facade."""

    @classmethod
    def publish_message(
        cls,
        runtime: Any,
        role: str,
        content: str,
        *,
        persist_synchronously: bool = False,
        require_persistence: bool = False,
        **fields: Any,
    ) -> None:
        """Publish one message with strict persistence preceding history mutation."""

        def persist() -> None:
            cls.persist_message(
                role=role,
                content=content,
                memory_manager=runtime.memory_manager,
                memory_session_id=runtime._memory_session_id,
                usage_logger=runtime.usage_logger,
                tool_name=fields.get("name"),
                tool_call_id=fields.get("tool_call_id"),
                tool_calls=fields.get("tool_calls"),
                metadata=fields.get("metadata"),
                persist_synchronously=persist_synchronously,
                **({"require_persistence": True} if require_persistence else {}),
            )

        if require_persistence:
            persist()  # Before trimming or resolving any in-memory tool-call ID.

        max_history = getattr(runtime.settings, "max_conversation_history", 100)
        if len(runtime.conversation._messages) >= max_history:
            for i, msg in enumerate(runtime.conversation._messages):
                if i == 0 and getattr(msg, "role", None) == "system":
                    continue
                runtime.conversation._messages.pop(i)
                break

        if role == "tool":
            logging.getLogger("victor.agent.orchestrator").debug(
                "add_message(role=tool): name=%s tool_call_id=%s content_len=%d",
                fields.get("name"),
                fields.get("tool_call_id"),
                len(content),
            )
        runtime.conversation.add_message(role, content, **fields)
        if not require_persistence:
            persist()

    @staticmethod
    def persist_message(
        role: str,
        content: str,
        memory_manager: Any,
        memory_session_id: Optional[str],
        usage_logger: Any,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        persist_synchronously: bool = False,
        require_persistence: bool = False,
    ) -> None:
        """Persist a message to memory and emit usage analytics.

        This is the canonical persistence helper for orchestrator message
        writes. It preserves the legacy logging behavior expected by existing
        analytics flows while keeping ownership on ``ChatService``.
        """
        if require_persistence and (not memory_manager or not memory_session_id):
            raise RuntimeError("Durable message persistence requires a store and session")
        if memory_manager and memory_session_id:
            try:
                from victor.agent.conversation.types import MessageRole

                role_map = {
                    "user": MessageRole.USER,
                    "assistant": MessageRole.ASSISTANT,
                    "system": MessageRole.SYSTEM,
                    "tool": MessageRole.TOOL,
                    "tool_result": MessageRole.TOOL,
                    "tool_call": MessageRole.TOOL_CALL,
                }
                msg_role = role_map.get(role, MessageRole.USER)

                add_kwargs: Dict[str, Any] = {
                    "session_id": memory_session_id,
                    "role": msg_role,
                    "content": content,
                }
                if tool_name:
                    add_kwargs["tool_name"] = tool_name
                if tool_call_id:
                    add_kwargs["tool_call_id"] = tool_call_id
                if tool_calls:
                    add_kwargs["tool_calls"] = tool_calls
                if metadata:
                    add_kwargs["metadata"] = metadata

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                def _persist_background_message() -> None:
                    primary_error: BaseException | None = None
                    try:
                        memory_manager.add_message(**add_kwargs)
                    except Exception as exc:
                        logger.debug("Failed to persist message in background: %s", exc)
                    except BaseException as exc:
                        primary_error = exc
                        raise
                    finally:
                        # The worker owns this thread-local handle. Do not close
                        # the caller's connection or shut down a shared store.
                        close = getattr(memory_manager, "close_thread_connection", None)
                        if callable(close):
                            try:
                                close()
                            except Exception as exc:
                                logger.warning(
                                    "Background persistence connection cleanup failed: %s",
                                    type(exc).__name__,
                                )
                                if primary_error is None:
                                    raise
                                primary_error.add_note(
                                    f"Background persistence cleanup failed: {type(exc).__name__}"
                                )

                def _consume_background_result(future: asyncio.Future) -> None:
                    try:
                        future.exception()
                    except asyncio.CancelledError:
                        logger.debug("Background message persistence was cancelled")

                if (
                    loop is not None
                    and loop.is_running()
                    and not (persist_synchronously or require_persistence)
                ):
                    future = loop.run_in_executor(None, _persist_background_message)
                    future.add_done_callback(_consume_background_result)
                else:
                    memory_manager.add_message(**add_kwargs)
            except Exception as e:
                if require_persistence:
                    raise
                logger.debug("Failed to persist message: %s", e)

        if not usage_logger:
            return

        try:
            if hasattr(usage_logger, "log_event"):
                if role == "user":
                    usage_logger.log_event("user_prompt", {"content": content})
                elif role == "assistant":
                    usage_logger.log_event("assistant_response", {"content": content})
                    if hasattr(usage_logger, "set_reasoning_context") and content:
                        usage_logger.set_reasoning_context(content)
                return

            if hasattr(usage_logger, "log_message"):
                usage_logger.log_message(role, content)
        except Exception as e:
            logger.debug("Failed to log message: %s", e)
