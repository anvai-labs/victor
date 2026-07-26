"""GEPA v2 reflection/mutation service.

Wraps a provider with GEPA-specific prompts for the three core operations:
- reflect(): Analyze ASI execution traces, diagnose failure patterns
- mutate(): Generate improved prompt section from reflection
- merge(): Combine strengths of two Pareto-optimal candidates

Follows the DecisionService pattern: provider-agnostic, configurable
via settings, sync interface using a persistent background event loop.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any, Callable, Optional, Protocol, Tuple

from victor.framework.rl.mutator_rotation import is_rate_limit

logger = logging.getLogger(__name__)

# Asked for a replacement mutator after a rate limit: given the model that was
# refused and the error, return a ``(provider, model)`` to retry on, or None to
# give up. The service deliberately cannot build providers itself — whoever owns
# provider construction owns this callback.
MutatorFailover = Callable[[str, BaseException], Optional[Tuple[Any, str]]]

# Retry budget for a call that came back empty. The headroom is for thinking,
# not output — the mutate target is only ~1500 characters. Measured on
# deepseek-v4-pro with a realistic mutate prompt: 1000 tokens returns nothing at
# all, 4096 returns a truncated 992 chars, 8192 returns 1345, and 16384 returns
# 1485 — a full answer. Reasoning eats the budget before the rewrite starts, so
# anything less than this silently trades away the end of the prompt.
REASONING_TOKEN_BUDGET = 16384

# A budget that large takes a reasoning model two to three minutes. The normal
# 120s deadline would abort the retry before it answered, turning the escalation
# into a slower way to fail.
REASONING_TIMEOUT_S = 420.0


class _EmptyResponse(RuntimeError):
    """A call that succeeded and returned nothing. Not a throttle — see below."""


# ---------------------------------------------------------------------------
# Persistent background event loop (shared across all GEPAService instances)
# ---------------------------------------------------------------------------


class _BackgroundLoop:
    """Single asyncio event loop running in a daemon thread.

    Avoids the 'Event loop is closed' error caused by creating and closing
    a new loop per LLM call (which orphans httpx keep-alive connections).
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gepa-event-loop")
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any, timeout: Optional[float] = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


_GEPA_LOOP: Optional[_BackgroundLoop] = None
_GEPA_LOOP_LOCK = threading.Lock()


def _get_background_loop() -> _BackgroundLoop:
    global _GEPA_LOOP
    if _GEPA_LOOP is None:
        with _GEPA_LOOP_LOCK:
            if _GEPA_LOOP is None:
                _GEPA_LOOP = _BackgroundLoop()
    return _GEPA_LOOP


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

REFLECT_SYSTEM = """You are a prompt engineering expert analyzing execution traces \
for an AI coding agent. Your goal is to diagnose WHY the agent's current prompt \
guidance causes failures, and propose specific fixes.

You will receive:
1. Aggregated execution trace data (tool calls, reasoning, results, errors)
2. The current prompt section being evaluated

Produce exactly 3-5 specific, actionable bullet points. Each must:
- Reference a concrete failure pattern from the traces
- Propose a precise wording change to the prompt
- Explain the expected impact

Do NOT produce vague advice like "improve clarity". Be specific."""

MUTATE_SYSTEM = """You are rewriting a prompt section for an AI coding agent. \
You will receive the current prompt text and a reflection analyzing its failures.

Requirements:
- Address EVERY failure pattern from the reflection
- Keep the output under {max_chars} characters (HARD LIMIT)
- Be specific and actionable — no vague platitudes
- Preserve the section's core purpose
- Output ONLY the improved prompt text — no explanation, no preamble

Current length: {current_len} characters. Target: under {max_chars} characters."""

MERGE_SYSTEM = """You are combining two prompt section variants for an AI coding \
agent. Each variant excels at different task types.

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Create a merged version that preserves the strengths of both. Requirements:
- Keep under {max_chars} characters
- Identify what each does best and unify
- Output ONLY the merged text — no explanation"""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class GEPAServiceProtocol(Protocol):
    """Protocol for GEPA reflection and mutation LLM calls."""

    def reflect(self, traces_summary: str, section_name: str, current_text: str) -> str: ...

    def mutate(
        self,
        current_text: str,
        reflection: str,
        section_name: str,
        max_chars: int,
    ) -> str: ...

    def merge(
        self,
        candidate_a: str,
        candidate_b: str,
        section_name: str,
        max_chars: int,
    ) -> str: ...

    def get_tier(self) -> str: ...


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class GEPAService:
    """Wraps a provider with GEPA-specific prompts for reflect/mutate/merge.

    Sync interface — uses run_sync_in_thread internally since providers
    are async. Matches the existing GEPAStrategy calling convention.
    """

    def __init__(
        self,
        provider: Any,
        model: str,
        tier: str = "balanced",
        max_prompt_chars: int = 1500,
        timeout_s: float = 30.0,
        max_tokens: int = 1000,
        failover: Optional[MutatorFailover] = None,
        max_failovers: int = 3,
    ):
        self._provider = provider
        self._model = model
        self._tier = tier
        self._max_prompt_chars = max_prompt_chars
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._failover = failover
        self._max_failovers = max(0, max_failovers)

    def get_tier(self) -> str:
        return self._tier

    def reflect(self, traces_summary: str, section_name: str, current_text: str) -> str:
        """Analyze ASI traces and produce actionable reflection.

        The section is passed **in full**. It used to be truncated to 1000
        characters, which predates sections three times that size — four of the
        seven evolvable sections exceed the old cap (ASI 2934, GROUNDING_RULES
        1912, COMPLETION_GUIDANCE 1551, GROUNDING_RULES_EXTENDED 1066). Asked to
        diagnose text it could only partly read, the model reliably proposed
        nothing substantive, and the mutator downstream returned approximately
        its input: COMPLETION_GUIDANCE produced whitespace-only collapse on two
        independent evolution runs.
        """
        user_prompt = (
            f"Execution traces for section '{section_name}':\n\n"
            f"{traces_summary}\n\n"
            f"Current prompt section:\n{current_text}\n\n"
            f"Diagnose the failure patterns and propose specific fixes."
        )
        result = self._call_llm(REFLECT_SYSTEM, user_prompt)
        if result:
            return result
        return f"[Reflection unavailable — {self._tier} tier LLM call failed]"

    def mutate(
        self,
        current_text: str,
        reflection: str,
        section_name: str,
        max_chars: int = 0,
    ) -> str:
        """Generate improved prompt section from reflection."""
        limit = max_chars or self._max_prompt_chars
        system = MUTATE_SYSTEM.format(max_chars=limit, current_len=len(current_text))
        user_prompt = (
            f"Section: {section_name}\n\n"
            f"Current text:\n{current_text}\n\n"
            f"Reflection on failures:\n{reflection}\n\n"
            f"Generate the improved version (under {limit} characters):"
        )
        result = self._call_llm(system, user_prompt, max_tokens=self._max_tokens)
        if not result:
            # Returning the seed is the right fallback, but it must not be
            # mistaken for a mutation: whatever runs after this sees "new" text
            # identical to the input, and any reformatting it applies becomes
            # the candidate's entire diff.
            logger.warning(
                "GEPA mutate produced no candidate for '%s'; returning the prompt unchanged.",
                section_name,
            )
            return current_text

        original_len = len(result)
        from victor.framework.rl.prompt_hygiene import (
            evaluate_prompt_candidate,
            sanitize_prompt_candidate,
        )

        # Boundary-aware truncation + fence stripping + consecutive-line dedupe.
        # Prevents the corrupt mid-token / mid-sentence outputs seen in
        # LARGE_FILE_PAGINATION_GUIDANCE (44 chars) and INIT_SYNTHESIS_RULES.
        sanitized = sanitize_prompt_candidate(result, limit=limit, seed_text=current_text)
        if len(sanitized) < len(current_text) // 4:
            logger.warning(
                "GEPA mutate for '%s' collapsed to %d chars after sanitization; "
                "rejecting candidate.",
                section_name,
                len(sanitized),
            )
            return current_text
        if len(sanitized) != original_len:
            logger.info(
                "GEPA mutate output sanitized from %d to %d chars (truncated=%s)",
                original_len,
                len(sanitized),
                len(sanitized) < original_len,
            )

        # Structural hygiene gate. sanitize_prompt_candidate() already stripped
        # fences and collapsed duplicate lines, so the only remaining concerns
        # here are runaway growth and repetitive-garbage trigrams. Seed-
        # similarity and unsupported-addition violations are intentionally NOT
        # enforced on the mutation path: a mutation is expected to rewrite the
        # prompt, which legitimately has low overlap with the seed.
        report = evaluate_prompt_candidate(current_text, sanitized)
        structural = {"growth_exceeded", "repeated_trigrams"}
        triggered = structural & set(report.violations)
        if triggered:
            # Warning, not info: this discards a mutation the provider was paid
            # for and leaves the caller with text identical to its input, which
            # surfaces as a bare "no change" with no way to tell a rejected
            # candidate from a model that had nothing to offer.
            logger.warning(
                "GEPA rejected the candidate for %s on structural hygiene (%s); "
                "returning the prompt unchanged (%d chars offered, seed %d).",
                section_name,
                ",".join(sorted(triggered)),
                len(sanitized),
                len(current_text),
            )
            return current_text
        return sanitized

    def merge(
        self,
        candidate_a: str,
        candidate_b: str,
        section_name: str,
        max_chars: int = 0,
    ) -> str:
        """Combine strengths of two Pareto-optimal candidates."""
        limit = max_chars or self._max_prompt_chars
        system = MERGE_SYSTEM.format(
            candidate_a=candidate_a[:800],
            candidate_b=candidate_b[:800],
            max_chars=limit,
        )
        user_prompt = (
            f"Merge these two variants of '{section_name}' into one "
            f"that combines the best of both. Under {limit} characters."
        )
        result = self._call_llm(system, user_prompt, max_tokens=self._max_tokens)
        if result and len(result) > limit:
            result = result[:limit]
        return result or candidate_a  # Fallback to first candidate

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Call the provider, failing over to another one if this one throttles.

        The retry lives here rather than a layer up because this is the only
        place that knows the call failed. Returning None on a 429 nullifies the
        whole evolution — `mutate()` falls back to `current_text`, a downstream
        strategy reformats it, and the whitespace-only result is stored and
        reported as an evolved candidate — so the section is worth one more
        attempt on a provider that has not refused us.
        """
        budget = max_tokens or self._max_tokens
        deadline = self._timeout_s
        failovers_left = self._max_failovers
        escalations_left = 1

        while True:
            try:
                content = self._attempt_call(system_prompt, user_prompt, budget, deadline)
            except Exception as e:
                # Warning, not debug. This masqueraded as a strategy problem for
                # two full runs because "rate limited (429)" was only visible at
                # DEBUG.
                logger.warning(
                    "GEPA %s tier (%s) LLM call failed: %s",
                    self._tier,
                    self._model,
                    e,
                )
                # Ask even on the last allowed attempt: the lookup is what
                # reports the failure, and a shared rotation needs that to bench
                # the provider for later sections.
                replacement = self._request_failover(e)
                if replacement is None or failovers_left <= 0:
                    logger.warning("No mutator left to try — prompt will NOT be mutated.")
                    return None
                failovers_left -= 1
                self._provider, self._model = replacement
                logger.warning("Retrying the mutation on %s.", self._model)
                continue

            if content:
                return content

            if escalations_left > 0 and budget < REASONING_TOKEN_BUDGET:
                # A successful call that returns nothing is usually a reasoning
                # model that spent the whole budget thinking. Verified on
                # deepseek-v4-pro: the identical mutate prompt yields 0 chars at
                # 1000 tokens and a real rewrite at 4000. The old default was set
                # before reasoning models, so this looked like "the model had no
                # improvement to offer" for every one of them.
                escalations_left -= 1
                budget = REASONING_TOKEN_BUDGET
                # Extend the deadline with the budget. A 16k-token reasoning call
                # takes minutes; leaving the 120s timeout in place would abort the
                # retry before it answered and make the escalation a slower way to
                # fail.
                deadline = max(deadline, REASONING_TIMEOUT_S)
                logger.warning(
                    "GEPA %s tier (%s) returned no content; retrying with a %d-token "
                    "budget and a %.0fs deadline in case reasoning consumed it.",
                    self._tier,
                    self._model,
                    budget,
                    deadline,
                )
                continue

            # The budget was already generous and it still said nothing, so this
            # is about the model rather than the prompt. Measured on the same
            # mutate prompt: deepseek-v4-pro returns 0 characters where kimi-k3
            # returns a full 1624-character rewrite at the *original* budget. A
            # peer is worth one call before writing the section off.
            if failovers_left > 0:
                replacement = self._request_failover(
                    _EmptyResponse(f"{self._model} returned no content"),
                    only_if_throttled=False,
                )
                if replacement is not None:
                    failovers_left -= 1
                    self._provider, self._model = replacement
                    logger.warning("Still no content; trying %s instead.", self._model)
                    continue

            logger.warning(
                "GEPA %s tier (%s) returned no usable content; "
                "mutation will fall back to the unchanged prompt.",
                self._tier,
                self._model,
            )
            return None

    def _attempt_call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int],
        timeout_s: Optional[float] = None,
    ) -> Optional[str]:
        """One provider call. Raises on transport failure; None means no content."""
        from victor.providers.base import Message

        # Suppress thinking for Qwen models
        effective_user = user_prompt
        if "qwen" in self._model.lower():
            effective_user = f"/no_think\n{user_prompt}"

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=effective_user),
        ]
        loop = _get_background_loop()
        response = loop.run(
            self._provider.chat(
                messages=messages,
                model=self._model,
                max_tokens=max_tokens or self._max_tokens,
                temperature=0.7,
            ),
            timeout=timeout_s or self._timeout_s,
        )
        content = self._strip_thinking(response.content if response else "")
        if content and len(content) > 20:
            return content.strip()
        return None

    def _request_failover(
        self,
        error: BaseException,
        *,
        only_if_throttled: bool = True,
    ) -> Optional[Tuple[Any, str]]:
        """Ask for a replacement mutator.

        Transport *failures* only qualify when they are throttling: a malformed
        prompt, a bad key, or a bug in this file fails identically on every
        provider, so failing over on those would burn each one's quota to learn
        nothing and bench healthy providers on our own defect. An empty answer is
        the exception the callers opt into — it is model-specific, not shared.
        """
        if self._failover is None or (only_if_throttled and not is_rate_limit(error)):
            return None
        try:
            return self._failover(self._model, error)
        except Exception as exc:  # a broken failover must not mask the real error
            logger.warning("GEPA failover lookup failed: %s", exc)
            return None

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Strip <think> blocks and 'Thinking Process:' preamble."""
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if "Thinking Process:" in content:
            parts = re.split(r"\n(?=[A-Z]{3,}[:\s])", content)
            for part in reversed(parts):
                if "Thinking Process" not in part and len(part.strip()) > 50:
                    return part.strip()
        return content
