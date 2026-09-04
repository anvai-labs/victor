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

"""Turn-boundary context assembler.

Selects and arranges messages for each LLM call, respecting token budgets
while preserving the session ledger, recent turns, and high-value older messages.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, List, Optional, Tuple

from victor.config.orchestrator_constants import (
    ContextAssemblerConfig,
    CONTEXT_ASSEMBLER_CONFIG,
)
from victor.providers.base import Message

logger = logging.getLogger(__name__)


class TurnBoundaryContextAssembler:
    """Assembles the message list for each provider.chat() call.

    Behavior:
    1. Always keeps system prompt + rendered session ledger
    2. Keeps last `full_turn_count` turns in full
    3. For older messages: scores using score_fn, selects top within budget
    4. Returns new List[Message] — does NOT modify input
    5. If no ledger/score_fn, returns messages unchanged (graceful degradation)
    """

    def __init__(
        self,
        config: Optional[ContextAssemblerConfig] = None,
        session_ledger: Optional[object] = None,
        score_fn: Optional[Callable] = None,
        conversation_controller: Optional[object] = None,
        temperature_classifier: Optional[object] = None,
        tool_result_deduplicator: Optional[object] = None,
    ):
        self._config = config or CONTEXT_ASSEMBLER_CONFIG
        self._session_ledger = session_ledger
        self._score_fn = score_fn
        self._conversation_controller = conversation_controller
        self._temperature_classifier = temperature_classifier
        # FEP-0023 P2: when present, dedups the assembled *view* (never source
        # history). Wired only when USE_TOOL_RESULT_DEDUP is enabled, so the
        # assembler itself stays flag-agnostic: presence == active.
        self._tool_result_deduplicator = tool_result_deduplicator
        # Per-turn raw-score cache (co-design review U1-5): the raw score_fn
        # result over the unchanged older-message prefix is recomputed every
        # turn; key = (identity of first older message, prefix length, query,
        # total content length). Append-only turns hit; replacement and
        # length-changing in-place edits miss; equal-length in-place rewrites
        # are invisible but bounded to selection ordering (budget accounting
        # reads fresh lengths). Only the RAW scores are cached — temperature
        # multipliers depend on current_turn and are applied fresh per call.
        self._score_cache_key: Optional[tuple] = None
        self._score_cache_value: Optional[list] = None

    @property
    def config(self) -> ContextAssemblerConfig:
        return self._config

    def _deduplicate_semantic(self, messages: list) -> list:
        """Remove semantically redundant messages (DCE-inspired).

        Uses content normalization + hashing for fast near-duplicate detection.
        Short messages (<50 chars) are never deduplicated. Keeps first occurrence.
        """
        if not messages:
            return []

        seen_hashes: set = set()
        result = []

        for msg in messages:
            content = getattr(msg, "content", "") or ""

            # Short messages are cheap -- always keep
            if len(content) < 50:
                result.append(msg)
                continue

            # Normalize: strip whitespace, collapse runs, lowercase
            normalized = " ".join(content.lower().split())
            content_hash = hashlib.md5(normalized.encode()).hexdigest()

            if content_hash in seen_hashes:
                continue  # Duplicate -- skip

            seen_hashes.add(content_hash)
            result.append(msg)

        return result

    @staticmethod
    def _detect_focus_phase(recent_messages: list) -> str:
        """Detect current activity phase from recent tool calls (DACS-inspired).

        Returns: 'exploration' | 'mutation' | 'execution' | 'mixed'
        """
        from collections import Counter

        tool_types = []
        for msg in recent_messages[-6:]:
            content = getattr(msg, "content", "") or ""
            content_lower = content.lower()
            if any(
                kw in content_lower for kw in ("read(", "read_file", "code_search", "grep", "ls(")
            ):
                tool_types.append("exploration")
            elif any(
                kw in content_lower
                for kw in (
                    "edit(",
                    "edit_files",
                    "edit result",
                    "write(",
                    "write_file",
                    "write result",
                    "create",
                )
            ):
                tool_types.append("mutation")
            elif any(
                kw in content_lower for kw in ("shell(", "execute_bash", "bash", "test", "git ")
            ):
                tool_types.append("execution")

        if not tool_types:
            return "mixed"
        most_common = Counter(tool_types).most_common(1)[0][0]
        return most_common

    def _apply_focus_scoring(
        self,
        messages: list,
        scores: list,
        focus_phase: str,
        predicted_phase: str = None,
    ) -> list:
        """Adjust message scores based on current AND predicted phase.

        Boosts active-phase messages (1.5x), compresses stale-phase (0.3x).
        With prediction: double-compresses messages irrelevant to BOTH
        current and predicted phases (0.2x) — AutonAgenticAI-inspired.
        """
        if focus_phase == "mixed":
            return list(scores)

        phase_keywords = {
            "exploration": ("read(", "read_file", "search", "grep", "ls("),
            "mutation": (
                "edit(",
                "edit_files",
                "edit result",
                "write(",
                "write_file",
                "write result",
                "create",
            ),
            "execution": ("shell(", "execute_bash", "bash", "test", "git "),
        }

        adjusted = []
        for msg, score in zip(messages, scores):
            content = getattr(msg, "content", "") or ""
            content_lower = content.lower()

            is_exploration = any(
                kw in content_lower for kw in ("read(", "read_file", "search", "grep", "ls(")
            )
            is_mutation = any(
                kw in content_lower
                for kw in (
                    "edit(",
                    "edit_files",
                    "edit result",
                    "write(",
                    "write_file",
                    "write result",
                    "create",
                )
            )
            is_execution = any(
                kw in content_lower for kw in ("shell(", "execute_bash", "bash", "test", "git ")
            )

            if focus_phase == "mutation" and is_exploration:
                adjusted.append(score * 0.3)
            elif focus_phase == "exploration" and is_mutation:
                adjusted.append(score * 0.3)
            elif focus_phase == "mutation" and is_mutation:
                adjusted.append(score * 1.5)
            elif focus_phase == "exploration" and is_exploration:
                adjusted.append(score * 1.5)
            elif focus_phase == "execution" and is_execution:
                adjusted.append(score * 1.5)
            else:
                adjusted.append(score)

        # Predictive pruning: double-compress messages irrelevant to both phases
        if predicted_phase and predicted_phase != focus_phase and predicted_phase != "mixed":
            current_kws = phase_keywords.get(focus_phase, ())
            predicted_kws = phase_keywords.get(predicted_phase, ())

            for i, msg in enumerate(messages):
                content_lower = getattr(msg, "content", "").lower()
                relevant_to_current = any(kw in content_lower for kw in current_kws)
                relevant_to_predicted = any(kw in content_lower for kw in predicted_kws)
                if not relevant_to_current and not relevant_to_predicted:
                    adjusted[i] *= 0.2  # Aggressive prune for doubly-irrelevant

        return adjusted

    def assemble(
        self,
        messages: List[Message],
        max_context_chars: int,
        current_query: Optional[str] = None,
    ) -> List[Message]:
        """Assemble messages for an LLM call within token budget.

        Thin wrapper over :meth:`_assemble_impl` that applies the FEP-0023 P2
        tool-result dedup *view stage* as the final step. Both operate only on
        freshly-built lists, so the caller's source-of-truth history is never
        modified.

        Args:
            messages: Full conversation history
            max_context_chars: Maximum context size in characters
            current_query: Current user query for relevance scoring

        Returns:
            New list of messages fitting within budget
        """
        result = self._assemble_impl(messages, max_context_chars, current_query)
        if self._tool_result_deduplicator is not None:
            dedup_fn = getattr(self._tool_result_deduplicator, "deduplicate_history_view", None)
            if dedup_fn is not None:
                try:
                    result, stubbed = dedup_fn(result)
                    if stubbed:
                        logger.debug(
                            "Tool-result dedup stubbed %d message(s) in assembled view", stubbed
                        )
                except Exception as exc:  # pragma: no cover - never break assembly
                    logger.debug("Tool-result dedup view stage skipped: %s", exc)
        return result

    def _assemble_impl(
        self,
        messages: List[Message],
        max_context_chars: int,
        current_query: Optional[str] = None,
    ) -> List[Message]:
        """Core assembly (budgeting, ledger injection, scoring). Returns a new list."""
        if not messages:
            return []

        # Graceful degradation: if no ledger and no score_fn, pass through
        if self._session_ledger is None and self._score_fn is None:
            return list(messages)

        # Budget allocation
        history_budget = int(max_context_chars * self._config.history_budget_pct)
        ledger_budget = int(max_context_chars * self._config.ledger_budget_pct)

        result: List[Message] = []
        chars_used = 0

        # 1. Keep system prompt (first message if role=system)
        start_idx = 0
        if messages and messages[0].role == "system":
            result.append(messages[0])
            chars_used += len(messages[0].content)
            start_idx = 1

        # 2. Inject ledger as early assistant message
        if self._session_ledger is not None:
            render_fn = getattr(self._session_ledger, "render", None)
            if render_fn:
                ledger_text = render_fn(max_chars=ledger_budget)
                if ledger_text:
                    from victor.agent.conversation.types import (
                        MESSAGE_SOURCE_METADATA_KEY,
                        MessageSource,
                    )

                    ledger_msg = Message(
                        role="assistant",
                        content=ledger_text,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.LEDGER_RENDER.value},
                    )
                    result.append(ledger_msg)
                    chars_used += len(ledger_text)

        # Filter remaining messages to exclude additional system messages.
        # This ensures byte-identical system prompts for provider KV caching.
        # Additional system-role nudges should have been redirected to the
        # user prefix or converted to 'user' role by the policy.
        remaining = [m for m in messages[start_idx:] if m.role != "system"]
        if not remaining:
            return result

        # 3. Identify recent turns to keep in full
        full_turn_count = self._config.full_turn_count
        # Count turns (a turn = user message + subsequent assistant/tool messages)
        turn_boundaries = []
        for i, msg in enumerate(remaining):
            if msg.role == "user":
                turn_boundaries.append(i)

        if len(turn_boundaries) <= full_turn_count:
            # All messages fit as recent turns
            for msg in remaining:
                result.append(msg)
                chars_used += len(msg.content)
            return result

        # Split: older messages vs recent turns
        recent_start = turn_boundaries[-full_turn_count]
        older_messages = remaining[:recent_start]
        recent_messages = remaining[recent_start:]

        # Calculate budget for older messages
        recent_chars = sum(len(m.content) for m in recent_messages)
        older_budget = max(0, history_budget - chars_used - recent_chars)

        # 4. Score and select older messages
        older_chars = 0
        if older_messages and self._score_fn and older_budget > 0:
            # Index map (co-design review U1-5): older_messages.index(msg)
            # was an O(n) scan with value-equality per selected message —
            # O(n^2) per turn on long histories. Keyed by id() because the
            # temperature multipliers below also key on object identity.
            index_by_id = {id(m): i for i, m in enumerate(older_messages)}
            # Raw-score cache (see __init__). The key is (first-msg identity,
            # prefix length, query) PLUS a cheap content fingerprint that
            # catches mid-prefix replacement and length-changing in-place
            # edits — the id/len pair alone could not (adversarial-review
            # finding: a replaced middle message kept the key stable and the
            # stale cached list then missed the index map with a KeyError).
            content_fingerprint = sum(len(m.content) for m in older_messages)
            cache_key = (
                id(older_messages[0]),
                len(older_messages),
                current_query,
                content_fingerprint,
            )
            cached = (
                self._score_cache_value
                if self._score_cache_key == cache_key and self._score_cache_value is not None
                else None
            )
            if cached is not None and all(id(m) in index_by_id for m, _ in cached):
                scored = cached
            else:
                scored = self._score_fn(older_messages, current_query)
                self._score_cache_key = cache_key
                self._score_cache_value = scored
            # scored should be list of (message, score) or similar
            if scored and isinstance(scored[0], tuple):
                # Apply temperature multipliers if classifier is available
                if self._temperature_classifier is not None:
                    try:
                        recent_tool_names: frozenset = frozenset()
                        classified = self._temperature_classifier.classify(
                            older_messages,
                            current_turn=len(turn_boundaries) - full_turn_count,
                            recent_tool_names=recent_tool_names,
                        )
                        multipliers = self._temperature_classifier.get_score_multipliers(classified)
                        if multipliers:
                            scored = [
                                (msg, score * multipliers.get(id(msg), 1.0))
                                for msg, score in scored
                            ]
                    except Exception as _te:
                        logger.debug(f"Context temperature classification skipped: {_te}")
                scored = list(scored)
                scored.sort(key=lambda x: x[1], reverse=True)
                selected_older = []
                older_chars = 0
                for msg, score in scored:
                    msg_len = len(msg.content)
                    if older_chars + msg_len <= older_budget:
                        # Foreign objects (a misbehaving score_fn returning
                        # messages outside the prefix) fall back to the
                        # legacy linear scan, preserving its ValueError
                        # semantics instead of raising KeyError
                        # (adversarial-review finding).
                        pos = index_by_id.get(id(msg))
                        if pos is None:
                            pos = older_messages.index(msg)
                        selected_older.append((msg, score, pos))
                        older_chars += msg_len
                # Sort by original order
                selected_older.sort(key=lambda x: x[2])
                for msg, _, _ in selected_older:
                    result.append(msg)
            else:
                # score_fn returned non-tuple, include what fits
                older_chars = 0
                for msg in older_messages:
                    if older_chars + len(msg.content) <= older_budget:
                        result.append(msg)
                        older_chars += len(msg.content)
        elif older_messages and older_budget > 0:
            # No score_fn: use Rust-accelerated fit_context() for smart selection
            try:
                from victor.processing.native.context_fitter import fit_context

                # Convert chars budget to token budget (approx 4 chars/token)
                token_budget = older_budget // 4
                msg_dicts = [
                    {"role": m.role, "content": m.content, "priority": 50} for m in older_messages
                ]
                fit_result = fit_context(msg_dicts, budget=token_budget, strategy="smart")
                for idx in fit_result.kept_indices:
                    result.append(older_messages[idx])
                    older_chars += len(older_messages[idx].content)
            except Exception:
                # Fallback: include what fits from the end (most recent older msgs)
                older_chars = 0
                for msg in reversed(older_messages):
                    if older_chars + len(msg.content) <= older_budget:
                        result.insert(len(result), msg)
                        older_chars += len(msg.content)

        # 5. Semantic augmentation: retrieve relevant compacted context
        if self._conversation_controller and current_query:
            retrieve_fn = getattr(self._conversation_controller, "retrieve_relevant_history", None)
            if retrieve_fn:
                try:
                    remaining_budget = older_budget - older_chars
                    if remaining_budget > 500:
                        relevant = retrieve_fn(query=current_query, limit=3)
                        semantic_chars = 0
                        for ctx_str in relevant:
                            if semantic_chars + len(ctx_str) <= remaining_budget:
                                result.append(
                                    Message(
                                        role="assistant",
                                        content=f"[Historical context: {ctx_str}]",
                                    )
                                )
                                semantic_chars += len(ctx_str)
                except Exception as e:
                    logger.debug(f"Semantic retrieval failed gracefully: {e}")

        # 6. Append recent turns (always kept)
        for msg in recent_messages:
            result.append(msg)

        return result
