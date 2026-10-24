"""Per-agent context lifecycle management.

This service is the common execution hook for root agents and subagents. It
keeps context compaction out of the orchestrator and scopes state by
``AgentRuntimeContext`` so sibling agents do not share histories.
"""

from __future__ import annotations

import logging
import re
import inspect
from typing import Any, Dict, Iterable, List, Optional

from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.services.context_service import (
    CompactionHistory,
    ContextService,
    ContextServiceConfig,
    ContextServiceRegistry,
)
from victor.core.provider_messages import Message

logger = logging.getLogger(__name__)


class LifecycleCompactionSummarizerAdapter:
    """Adapt legacy compaction summarizers to lifecycle-service kwargs."""

    def __init__(self, strategy: Any, ledger: Optional[Any] = None) -> None:
        self._strategy = strategy
        self._ledger = ledger

    async def summarize(
        self,
        *,
        runtime_context: AgentRuntimeContext,
        removed_messages: List[Dict[str, Any]],
        retained_messages: List[Dict[str, Any]],
        reason: str,
    ) -> str:
        """Summarize removed lifecycle messages with a legacy summarizer strategy."""
        if not removed_messages:
            return ""
        legacy_messages = [
            Message(
                role=str(message.get("role") or "unknown"),
                content=str(message.get("content") or ""),
                metadata=dict(message.get("metadata") or {}),
            )
            for message in removed_messages
        ]
        summarize_async = getattr(self._strategy, "summarize_async", None)
        if callable(summarize_async):
            summary = summarize_async(legacy_messages, self._ledger)
        else:
            summarize = getattr(self._strategy, "summarize", None)
            if not callable(summarize):
                return ""
            summary = summarize(legacy_messages, self._ledger)
        if inspect.isawaitable(summary):
            summary = await summary
        return str(summary or "")


class ContextLifecycleService:
    """Manage context hydration, threshold compaction, and parent handoff."""

    def __init__(
        self,
        *,
        registry: ContextServiceRegistry,
        conversation_store: Optional[Any] = None,
        compaction_summarizer: Optional[Any] = None,
    ) -> None:
        self._registry = registry
        self._conversation_store = conversation_store
        self._compaction_summarizer = compaction_summarizer

    @classmethod
    def with_defaults(
        cls,
        *,
        max_tokens: int,
        min_messages_to_keep: int = 6,
        overflow_threshold_percent: float = 90.0,
        default_compaction_strategy: str = "tiered",
        conversation_store: Optional[Any] = None,
        compaction_summarizer: Optional[Any] = None,
    ) -> "ContextLifecycleService":
        """Create a lifecycle service with a session-scoped context registry."""
        return cls(
            registry=ContextServiceRegistry(
                ContextServiceConfig(
                    max_tokens=max(1, int(max_tokens)),
                    min_messages_to_keep=min_messages_to_keep,
                    overflow_threshold_percent=overflow_threshold_percent,
                    default_compaction_strategy=default_compaction_strategy,
                )
            ),
            conversation_store=conversation_store,
            compaction_summarizer=compaction_summarizer,
        )

    def context_for(self, runtime_context: AgentRuntimeContext) -> ContextService:
        """Return the context service for one agent runtime session."""
        return self._registry.get_or_create(runtime_context)

    async def before_tool_output(
        self,
        runtime_context: AgentRuntimeContext,
        *,
        estimated_output_tokens: int,
        messages: Optional[Iterable[Any]] = None,
        provider_name: str = "",
        model_name: str = "",
        task_type: str = "",
        min_messages: int = 6,
        default_strategy: str = "tiered",
        persist_event: bool = True,
        history: Optional[CompactionHistory] = None,
    ) -> Dict[str, Any]:
        """Compact one agent context before injecting a large tool-output block."""
        service = self.context_for(runtime_context)
        if history is not None:
            service.clear_messages(retain_system=False)
            service.add_messages(list(history.messages))
        elif messages is not None:
            self._replace_messages(service, runtime_context, messages)

        messages_before = service.get_messages()
        tokens_before = service.get_context_size()
        decision = await service.prepare_for_tool_output_injection(
            estimated_output_tokens,
            provider_name=provider_name,
            model_name=model_name,
            task_type=task_type,
            min_messages=min_messages,
            default_strategy=default_strategy,
            history=history,
        )
        tokens_after = service.get_context_size()
        tokens_freed = max(0, tokens_before - tokens_after)
        decision.update(runtime_context.identity_metadata())
        decision["tokens_before"] = tokens_before
        decision["tokens_after"] = tokens_after
        decision["tokens_freed"] = tokens_freed
        if tokens_freed and not decision.get("saved_tokens"):
            decision["saved_tokens"] = tokens_freed

        if persist_event and decision.get("compacted"):
            decision["summary"] = await self._summarize_compaction(
                runtime_context,
                messages_before=messages_before,
                messages_after=service.get_messages(),
                reason="pre_tool_output",
            )
            event_id = self._record_compaction_event(runtime_context, decision)
            if event_id:
                decision["compaction_event_id"] = event_id
        return decision

    async def after_agent_turn(
        self,
        runtime_context: AgentRuntimeContext,
        *,
        messages: Optional[Iterable[Any]] = None,
        strategy: Optional[str] = None,
        min_messages: Optional[int] = None,
        persist_event: bool = True,
        history: Optional[CompactionHistory] = None,
    ) -> Dict[str, Any]:
        """Refresh one agent context and compact if it crossed threshold."""
        service = self.context_for(runtime_context)
        if history is not None:
            service.clear_messages(retain_system=False)
            service.add_messages(list(history.messages))
        elif messages is not None:
            self._replace_messages(service, runtime_context, messages)

        messages_before = service.get_messages()
        tokens_before = service.get_context_size()
        result = await self._registry.compact_if_needed(
            runtime_context,
            strategy=strategy,
            min_messages=min_messages,
            history=history,
        )
        tokens_after = service.get_context_size()
        tokens_freed = max(0, tokens_before - tokens_after)
        result["tokens_before"] = tokens_before
        result["tokens_after"] = tokens_after
        result["tokens_freed"] = tokens_freed
        result["strategy"] = strategy or "tiered"

        if persist_event and result.get("compacted"):
            result["summary"] = await self._summarize_compaction(
                runtime_context,
                messages_before=messages_before,
                messages_after=service.get_messages(),
                reason="after_agent_turn",
            )
            event_id = self._record_compaction_event(runtime_context, result)
            if event_id:
                result["compaction_event_id"] = event_id

        return result

    def build_parent_handoff(
        self,
        runtime_context: AgentRuntimeContext,
        *,
        summary: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_summary_chars: int = 500,
    ) -> Dict[str, Any]:
        """Build a bounded child-to-parent handoff payload."""
        payload = runtime_context.identity_metadata()
        payload["status"] = status
        payload["summary"] = self._bounded_summary(summary, max_summary_chars=max_summary_chars)
        payload.update(dict(metadata or {}))
        return payload

    def _replace_messages(
        self,
        service: ContextService,
        runtime_context: AgentRuntimeContext,
        messages: Iterable[Any],
    ) -> None:
        service.clear_messages(retain_system=False)
        identity = runtime_context.identity_metadata()
        for message in messages:
            payload = self._normalize_message(message)
            metadata = dict(payload.get("metadata") or {})
            metadata.update({key: value for key, value in identity.items() if value is not None})
            payload["metadata"] = metadata
            service.add_message(payload)

    def _record_compaction_event(
        self,
        runtime_context: AgentRuntimeContext,
        result: Dict[str, Any],
    ) -> Optional[str]:
        if self._conversation_store is None:
            return None
        record = getattr(self._conversation_store, "record_compaction_event", None)
        if not callable(record):
            return None
        try:
            return record(
                session_id=runtime_context.session_id,
                agent_id=runtime_context.agent_id,
                strategy=str(result.get("strategy") or "tiered"),
                messages_removed=int(result.get("messages_removed", 0) or 0),
                tokens_freed=int(result.get("tokens_freed", 0) or 0),
                summary=str(result.get("summary") or ""),
                metadata=runtime_context.identity_metadata(),
            )
        except Exception as exc:
            logger.warning("Failed to persist context compaction event: %s", exc)
            return None

    @staticmethod
    def _normalize_message(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)
        role = getattr(message, "role", "unknown")
        if hasattr(role, "value"):
            role = role.value
        return {
            "role": str(role),
            "content": getattr(message, "content", str(message)),
            "metadata": dict(getattr(message, "metadata", {}) or {}),
        }

    async def _summarize_compaction(
        self,
        runtime_context: AgentRuntimeContext,
        *,
        messages_before: List[Any],
        messages_after: List[Any],
        reason: str,
    ) -> str:
        retained_ids = {id(message) for message in messages_after}
        removed_messages = [
            self._normalize_message(message)
            for message in messages_before
            if id(message) not in retained_ids
        ]
        retained_messages = [self._normalize_message(message) for message in messages_after]
        if self._compaction_summarizer is not None:
            summarize = getattr(self._compaction_summarizer, "summarize", None)
            if callable(summarize):
                summary = summarize(
                    runtime_context=runtime_context,
                    removed_messages=removed_messages,
                    retained_messages=retained_messages,
                    reason=reason,
                )
                if inspect.isawaitable(summary):
                    summary = await summary
                if summary:
                    return str(summary)
        return self._deterministic_summary(runtime_context, removed_messages, reason=reason)

    @staticmethod
    def _deterministic_summary(
        runtime_context: AgentRuntimeContext,
        removed_messages: List[Dict[str, Any]],
        *,
        reason: str,
    ) -> str:
        role_counts: Dict[str, int] = {}
        topics: List[str] = []
        seen_topics = set()
        for message in removed_messages:
            role = str(message.get("role") or "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            content = str(message.get("content") or "")
            for topic in re.findall(r"\b[A-Za-z][A-Za-z0-9_./-]{2,}\b", content[:500]):
                normalized = topic.lower()
                if normalized in seen_topics:
                    continue
                seen_topics.add(normalized)
                topics.append(topic)
                if len(topics) >= 5:
                    break
            if len(topics) >= 5:
                break

        count = len(removed_messages)
        parts = [f"{count} messages compacted for {runtime_context.display_name}"]
        if role_counts:
            role_summary = ", ".join(f"{role}={role_counts[role]}" for role in sorted(role_counts))
            parts.append(f"roles: {role_summary}")
        if topics:
            parts.append(f"topics: {', '.join(topics[:5])}")
        parts.append(f"reason: {reason}")
        return "; ".join(parts)

    @staticmethod
    def _bounded_summary(summary: str, *, max_summary_chars: int) -> str:
        text = str(summary or "")
        if len(text) <= max_summary_chars:
            return text
        return text[: max(0, max_summary_chars - 3)].rstrip() + "..."
