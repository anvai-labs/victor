# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Retry policy for empty model responses, independent of transport.

A reasoning model that spends its whole ``max_tokens`` budget on reasoning tokens
returns a well-formed response with empty ``content``. Retrying that with
identical parameters reproduces it exactly — the failure is deterministic, so a
retry loop that does not *change* anything is guaranteed to burn its full budget
and still fail.

This module owns the escalation decision so both the streaming chat path and the
planning path apply the same policy. It deliberately depends on nothing from the
orchestrator, the stream context, or a provider instance: callers pass in what
they observed and receive the parameters for the next attempt.

Background: session ``sandhi-cdfbc589`` (2026-07-26) lost ~100s to three
byte-identical 35s plan-generation attempts against glm-5.2, each returning
``length=0``. The streaming path had already learned this lesson (session
``modality-doc-review-fixes-b4e87728``) but kept the remedy inline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "EmptyResponseDiagnosis",
    "RetryParameters",
    "diagnose_empty_response",
    "next_retry_parameters",
    "DEFAULT_TEMPERATURE_LADDER",
    "MAX_ESCALATED_TOKENS",
]

#: Temperatures tried on successive retries. Varying temperature breaks
#: deterministic reproduction of the same empty completion.
DEFAULT_TEMPERATURE_LADDER: tuple[float, ...] = (0.7, 0.9)

#: Ceiling for escalated ``max_tokens``. Matches the streaming path's cap.
MAX_ESCALATED_TOKENS = 32768

#: Multiplier applied to ``max_tokens`` when reasoning exhaustion is diagnosed.
_REASONING_ESCALATION_FACTOR = 4


@dataclass(frozen=True)
class EmptyResponseDiagnosis:
    """Why a response came back without usable content.

    Attributes:
        reasoning_exhausted: The model produced reasoning tokens but no content,
            i.e. the token budget was consumed before the answer began.
        reasoning_chars: Observed reasoning characters, when known.
        stop_reason: Provider-reported stop reason, when known.
    """

    reasoning_exhausted: bool = False
    reasoning_chars: Optional[int] = None
    stop_reason: Optional[str] = None

    @property
    def summary(self) -> str:
        """Short human-readable cause, for logs and error messages."""
        if self.reasoning_exhausted:
            return (
                f"reasoning-token exhaustion (reasoning_chars={self.reasoning_chars}, "
                f"stop_reason={self.stop_reason})"
            )
        return f"empty content with no reasoning output (stop_reason={self.stop_reason})"


@dataclass(frozen=True)
class RetryParameters:
    """Parameters for the next attempt after an empty response.

    Attributes:
        max_tokens: Token budget for the retry.
        temperature: Temperature for the retry, or None to leave unchanged.
        reasoning_effort: Value to send as ``reasoning_effort``, or None to omit.
            Callers must only forward this when the provider accepts it — see
            ``BaseProvider.supports_reasoning_effort``.
        escalated: Whether anything actually changed versus the failed attempt.
            A retry with ``escalated=False`` will reproduce the same failure.
        reason: Human-readable explanation, suitable for logging.
    """

    max_tokens: int
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    escalated: bool = False
    reason: str = ""


def diagnose_empty_response(
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> EmptyResponseDiagnosis:
    """Classify an empty response from whatever the transport observed.

    Args:
        diagnostics: Transport-supplied signals. Recognised keys are
            ``reasoning_chars`` (int) and ``stop_reason`` (str). Any mapping is
            accepted so callers need not construct a specific type; unknown keys
            are ignored and a missing/empty mapping yields an undiagnosed result.

    Returns:
        The diagnosis. ``reasoning_exhausted`` is True only when the transport
        actually observed reasoning output — absence of evidence is not treated
        as evidence of exhaustion.
    """
    diagnostics = diagnostics or {}

    raw_chars = diagnostics.get("reasoning_chars")
    try:
        reasoning_chars = int(raw_chars) if raw_chars is not None else None
    except (TypeError, ValueError):
        reasoning_chars = None

    stop_reason = diagnostics.get("stop_reason")

    return EmptyResponseDiagnosis(
        reasoning_exhausted=bool(reasoning_chars),
        reasoning_chars=reasoning_chars,
        stop_reason=str(stop_reason) if stop_reason is not None else None,
    )


def next_retry_parameters(
    *,
    attempt: int,
    base_max_tokens: int,
    diagnosis: Optional[EmptyResponseDiagnosis] = None,
    temperature_ladder: Sequence[float] = DEFAULT_TEMPERATURE_LADDER,
    supports_reasoning_effort: bool = False,
    max_tokens_cap: int = MAX_ESCALATED_TOKENS,
) -> RetryParameters:
    """Compute parameters for retry number ``attempt`` after an empty response.

    Args:
        attempt: Zero-based retry index (0 = the first retry after the initial
            failure).
        base_max_tokens: ``max_tokens`` used by the attempt that came back empty.
        diagnosis: Result of :func:`diagnose_empty_response`, when available.
        temperature_ladder: Temperatures to walk across successive retries. The
            last entry is reused once the ladder is exhausted.
        supports_reasoning_effort: Whether the provider accepts a
            ``reasoning_effort`` parameter for this model. Only then is one
            returned — sending it to a provider that rejects it turns a
            recoverable empty response into a hard 400.
        max_tokens_cap: Upper bound for the escalated token budget.

    Returns:
        Parameters for the next attempt. ``escalated`` is False only when
        nothing could be changed, which callers should treat as "retrying is
        pointless" rather than as a normal retry.
    """
    diagnosis = diagnosis or EmptyResponseDiagnosis()
    reasons: list[str] = []

    max_tokens = int(base_max_tokens) if base_max_tokens else 0
    if diagnosis.reasoning_exhausted and max_tokens > 0:
        escalated_tokens = min(max_tokens * _REASONING_ESCALATION_FACTOR, max_tokens_cap)
        if escalated_tokens > max_tokens:
            reasons.append(f"max_tokens {max_tokens} -> {escalated_tokens}")
            max_tokens = escalated_tokens

    temperature: Optional[float] = None
    if temperature_ladder:
        index = min(attempt, len(temperature_ladder) - 1)
        temperature = float(temperature_ladder[index])
        # Only a *new* rung counts as escalation. Past the end of the ladder the
        # temperature repeats the previous attempt's value, so it changes nothing
        # and must not be reported as though it did.
        if attempt < len(temperature_ladder):
            reasons.append(f"temperature={temperature}")

    # Ask the model to spend less of the budget thinking. Only meaningful when
    # the diagnosis is reasoning exhaustion, and only safe where it is accepted.
    # Counts as escalation only on the first retry — after that it is already set
    # and re-sending it changes nothing.
    reasoning_effort: Optional[str] = None
    if diagnosis.reasoning_exhausted and supports_reasoning_effort:
        reasoning_effort = "low"
        if attempt == 0:
            reasons.append("reasoning_effort=low")

    reason = (
        f"empty response diagnosed as {diagnosis.summary}; retrying with " + ", ".join(reasons)
        if reasons
        else f"empty response diagnosed as {diagnosis.summary}; nothing left to escalate"
    )

    return RetryParameters(
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        escalated=bool(reasons),
        reason=reason,
    )
