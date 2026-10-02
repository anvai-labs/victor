"""Shared model controls: vocabulary, not a claim of universal provider support.

Keep effort separate from model identity, sampling temperature, token limits and
thinking visibility. Provider adapters own each model's supported subset and
native encoding; Sandhi's neutral request owns the wire representation.
"""

from typing import Literal, get_args

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
REASONING_EFFORT_VALUES = frozenset(get_args(ReasoningEffort))


def validate_reasoning_effort(value: object) -> None:
    """Reject malformed effort values while preserving None as unspecified."""
    if value is not None and (not isinstance(value, str) or value not in REASONING_EFFORT_VALUES):
        raise ValueError("reasoning_effort must be a supported effort label or None")
