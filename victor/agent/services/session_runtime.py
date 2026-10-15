# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned session runtime helper."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from victor.agent.runtime.context import AgentRuntimeContext, require_runtime_context

if TYPE_CHECKING:
    from victor.agent.conversation.store import ConversationStore

logger = logging.getLogger(__name__)


class SessionRuntime:
    """Bridge orchestrator runtime state to the canonical session service."""

    def __init__(
        self, runtime_host: Any, *, validate_session: Optional[Callable[[str], None]] = None
    ) -> None:
        self._runtime = runtime_host
        self._validate_session = validate_session

    def bind_member(self, context: AgentRuntimeContext) -> None:
        """Register a member's canonical session before publishing its identity."""
        context = require_runtime_context(lambda: context)
        runtime = self._runtime
        identity = runtime._runtime_identity
        identity.validate_session_transition(context.session_id)
        store = runtime.memory_manager
        if store is not None:
            if store.get_session(context.session_id) is None:
                store.create_session(
                    session_id=context.session_id,
                    provider=runtime.provider_name,
                    model=runtime.model,
                )
            runtime._conversation_controller.set_conversation_store(store, context.session_id)
            runtime._memory_session_id = context.session_id
        identity.bind_member(context)
        self.sync_runtime_state()

    def bind_restored_session(self, store: ConversationStore, session_id: str) -> None:
        """Bind explicit persisted resume to the same store used for hydration.

        No history is republished here. Subsequent turns, retrieval and session
        metadata must all use the selected identity, including after cold startup.
        """
        runtime = self._runtime
        runtime._runtime_identity.validate_session_transition(session_id)
        if runtime.memory_manager is not None and runtime.memory_manager is not store:
            raise ValueError("Restored session must use the active conversation store")
        if store.get_session(session_id) is None:
            raise ValueError("Restored session is not present in its conversation store")
        if runtime.memory_manager is not None:
            runtime._conversation_controller.set_conversation_store(store, session_id)
            runtime._memory_session_id = session_id
        # Explicit resume can hydrate transient history while memory is disabled;
        # selecting history is not permission to enable persistence.
        runtime.active_session_id = session_id
        self.sync_runtime_state()

    def sync_runtime_state(self) -> None:
        """Keep SessionService bound to the current live runtime state."""
        runtime = self._runtime
        if runtime._session_service is None or not hasattr(
            runtime._session_service,
            "bind_runtime_components",
        ):
            return

        runtime._session_service.bind_runtime_components(
            lifecycle_manager=runtime._lifecycle_manager,
            memory_manager=runtime.memory_manager,
            checkpoint_manager=runtime._checkpoint_manager,
            cost_tracker=runtime._session_cost_tracker,
            memory_session_id=runtime._memory_session_id,
        )

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions for recovery."""
        self.sync_runtime_state()
        return self._runtime._session_service.get_recent_sessions(limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session."""
        runtime = self._runtime
        if self._validate_session is not None:
            self._validate_session(session_id)
        self.sync_runtime_state()
        success = runtime._session_service.recover_session(session_id)
        if success:
            runtime._memory_session_id = session_id
            runtime.active_session_id = session_id
        return success

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context messages from the canonical session service."""
        runtime = self._runtime
        self.sync_runtime_state()
        return runtime._session_service.get_memory_context(
            max_tokens=max_tokens,
            messages=runtime.messages,
        )

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current memory session."""
        self.sync_runtime_state()
        return self._runtime._session_service.get_session_stats()

    def record_prompt_optimization_metadata(self, turn_context: Any) -> None:
        """Persist the active live prompt-optimization identity for the session."""
        runtime = self._runtime
        metadata = getattr(turn_context, "prompt_optimization_metadata", None)
        if not isinstance(metadata, dict):
            return

        normalized = {
            "entries": list(metadata.get("entries") or []),
            "by_section": dict(metadata.get("by_section") or {}),
            "prompt_overlays": self._normalize_prompt_overlay_metadata(
                metadata.get("prompt_overlays")
            ),
        }
        runtime._active_prompt_optimization_metadata = normalized
        self._emit_prompt_overlay_metadata(normalized)

        if runtime._session_service is None or not hasattr(
            runtime._session_service, "update_session_metadata"
        ):
            return

        try:
            if (
                hasattr(runtime._session_service, "_current_session")
                and runtime._session_service._current_session is not None
            ):
                runtime._session_service.update_session_metadata(
                    {"prompt_optimization": normalized}
                )
        except Exception as exc:
            logger.debug("Failed to update session prompt-optimization metadata: %s", exc)

    def get_active_prompt_optimization_metadata(self) -> Dict[str, Any]:
        """Return the canonical prompt-optimization metadata for the live session."""
        payload = getattr(self._runtime, "_active_prompt_optimization_metadata", None)
        if not isinstance(payload, dict):
            return {"entries": [], "by_section": {}, "prompt_overlays": []}
        return {
            "entries": list(payload.get("entries") or []),
            "by_section": dict(payload.get("by_section") or {}),
            "prompt_overlays": self._normalize_prompt_overlay_metadata(
                payload.get("prompt_overlays")
            ),
        }

    @staticmethod
    def _normalize_prompt_overlay_metadata(value: Any) -> List[Dict[str, str]]:
        """Return trace-safe prompt overlay metadata without prompt text."""
        if not isinstance(value, list):
            return []

        overlays: List[Dict[str, str]] = []
        for raw in value:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            placement = str(raw.get("placement") or "turn_prefix").strip() or "turn_prefix"
            overlays.append(
                {
                    "name": name,
                    "placement": placement,
                }
            )
        return overlays

    def _emit_prompt_overlay_metadata(self, metadata: Dict[str, Any]) -> None:
        """Emit trace-safe prompt overlay metadata through the usage logger if present."""
        overlays = metadata.get("prompt_overlays") or []
        if not overlays:
            return

        usage_logger = getattr(self._runtime, "usage_logger", None)
        log_event = getattr(usage_logger, "log_event", None)
        if not callable(log_event):
            return

        try:
            log_event(
                "prompt_overlays.active",
                {
                    "prompt_overlays": list(overlays),
                    "prompt_overlay_count": len(overlays),
                },
            )
        except Exception as exc:
            logger.debug("Failed to emit prompt overlay metadata: %s", exc)
