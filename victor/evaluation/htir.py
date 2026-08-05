# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""HTIR — Harness Trace Intermediate Representation (EVR-5, ADR-012).

Normalizes the structured :class:`~victor.evaluation.agentic_harness.AgenticExecutionTrace`
into a canonical, harness-agnostic step sequence tagged along three axes:

- **Role** — who produced the step (assistant / tool / user / system).
- **Status** — the step's execution outcome (ok / failed / timeout / refused).
- **Artifact-effect** — whether the step produced a verifiable effect, reusing the *same*
  classification the runtime completion gate uses
  (:func:`~victor.framework.effect_gate.classify_tool_result`) so an evaluation trace and a
  live turn agree on what "an effect" is.

Every step is additionally tagged with an **ETCLOVG layer** (Execution / Tooling /
Context-Memory / Lifecycle-Orchestration / Observability / Verification / Governance — the
HarnessFix taxonomy, arXiv:2606.06324). This gives recovery and failure-attribution a
structured vocabulary — *which harness layer failed* — instead of blind, exception-typed retry
(ADR-012, prong 2).

Pure and deterministic: no LLM, no I/O, no runtime coupling. It consumes only the trace's
structured fields, so it is unit-testable standalone and safe to call from the acceptance
oracle, recovery attribution, or offline analysis alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall


class Role(str, Enum):
    """Who produced a normalized step."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepStatus(str, Enum):
    """Execution outcome of a normalized step (HTIR "ExecutionStatus" axis)."""

    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    PENDING = "pending"


class ArtifactEffect(str, Enum):
    """Verifiable-effect class of a step. Mirrors :class:`~victor.framework.effect_gate.EffectClass`
    plus an explicit ``NONE`` for steps that produced nothing verifiable."""

    NONE = "none"
    WORKSPACE_DELTA = "workspace_delta"
    VERIFIED_CHECK = "verified_check"
    GROUNDED_CLAIM = "grounded_claim"


class ETCLOVGLayer(str, Enum):
    """HarnessFix ETCLOVG failure-attribution layers (arXiv:2606.06324)."""

    EXECUTION = "execution"
    TOOLING = "tooling"
    CONTEXT_MEMORY = "context_memory"
    LIFECYCLE_ORCHESTRATION = "lifecycle_orchestration"
    OBSERVABILITY = "observability"
    VERIFICATION = "verification"
    GOVERNANCE = "governance"


# File-write / shell tools are Execution-layer even when they fail (a failed write reports NONE
# effect, but the failure belongs to Execution, not Tooling).
_EXECUTION_TOOLS = frozenset(
    {"write", "create_file", "edit", "replace_in_file", "patch", "shell", "run_command"}
)
# Read/retrieval tools ground a claim but mutate nothing — a Context-Memory-layer concern.
_CONTEXT_TOOLS = frozenset(
    {
        "read",
        "read_file",
        "cat",
        "grep",
        "search",
        "find",
        "ls",
        "glob",
        "list_files",
        "codebase_search",
        "semantic_search",
        "retrieve",
        "recall",
        "memory",
    }
)
# Planning / delegation / task-management tools drive the Lifecycle-Orchestration layer.
_ORCHESTRATION_TOOLS = frozenset(
    {"plan", "delegate", "handoff", "task", "todo", "create_task", "spawn_team", "team"}
)
# Governance: human-in-the-loop / policy / approval seams.
_GOVERNANCE_TOOLS = frozenset({"ask", "approval", "policy", "request_approval"})
# Observability: logging / metrics / reporting.
_OBSERVABILITY_TOOLS = frozenset({"log", "metric", "report", "trace"})

# First-line refusal markers, mirroring the conservative set in ``trajectory_eval`` — kept local so
# HTIR normalization has no import coupling to the scorer module.
_REFUSAL_PATTERNS = (
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i'm unable to",
    "i am unable to",
    "against policy",
    "cannot assist",
    "can't help with that",
)


@dataclass(frozen=True)
class HTIRStep:
    """One normalized step: role · status · artifact-effect · ETCLOVG layer."""

    index: int
    role: Role
    status: StepStatus
    effect: ArtifactEffect
    layer: ETCLOVGLayer
    tool_name: str = ""
    summary: str = ""

    @property
    def is_failure(self) -> bool:
        return self.status in (StepStatus.FAILED, StepStatus.TIMEOUT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role.value,
            "status": self.status.value,
            "effect": self.effect.value,
            "layer": self.layer.value,
            "tool_name": self.tool_name,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class HTIRTrace:
    """A trajectory normalized into HTIR steps, with layer-level failure attribution helpers."""

    task_id: str
    steps: tuple[HTIRStep, ...]
    session_id: str = ""
    benchmark: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> tuple[HTIRStep, ...]:
        return tuple(s for s in self.steps if s.is_failure)

    @property
    def effect_steps(self) -> tuple[HTIRStep, ...]:
        return tuple(s for s in self.steps if s.effect is not ArtifactEffect.NONE)

    def layers_touched(self) -> frozenset[ETCLOVGLayer]:
        return frozenset(s.layer for s in self.steps)

    def failures_by_layer(self) -> dict[ETCLOVGLayer, int]:
        """Count failed steps per ETCLOVG layer — the input to layer-targeted recovery (ADR-012)."""
        counts: dict[ETCLOVGLayer, int] = {}
        for step in self.failures:
            counts[step.layer] = counts.get(step.layer, 0) + 1
        return counts

    def dominant_failure_layer(self) -> Optional[ETCLOVGLayer]:
        """The layer with the most failures, or ``None`` when the trace has no failures."""
        counts = self.failures_by_layer()
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "benchmark": self.benchmark,
            "n_steps": len(self.steps),
            "failures_by_layer": {k.value: v for k, v in self.failures_by_layer().items()},
            "steps": [s.to_dict() for s in self.steps],
        }


def _artifact_effect(call: "EvalToolCall", turn_index: int) -> ArtifactEffect:
    """Classify a tool call's artifact-effect, reusing the runtime effect-gate rules."""
    from victor.framework.effect_gate import classify_tool_result

    evidence = classify_tool_result(
        {
            "tool_name": getattr(call, "name", ""),
            "args": getattr(call, "arguments", {}) or {},
            "success": getattr(call, "success", False),
        },
        turn_index,
    )
    if evidence is None:
        # A successful read/retrieval tool grounds a claim even when it mutates nothing.
        if (
            getattr(call, "success", False)
            and _canonical(getattr(call, "name", "")) in _CONTEXT_TOOLS
        ):
            return ArtifactEffect.GROUNDED_CLAIM
        return ArtifactEffect.NONE
    return ArtifactEffect(evidence.effect_class.value)


def _canonical(tool_name: str) -> str:
    from victor.tools.tool_names import get_canonical_name

    if not tool_name:
        return ""
    try:
        return get_canonical_name(tool_name)
    except Exception:  # noqa: BLE001 - unknown tools resolve to themselves
        return tool_name


def _infer_layer(canonical: str, effect: ArtifactEffect) -> ETCLOVGLayer:
    """Attribute a tool step to an ETCLOVG layer from its effect and canonical name.

    Effect wins first (a verified check is a Verification-layer step regardless of tool name); then
    name-based buckets; a step with no recognized bucket is a Tooling-layer step.
    """
    if effect is ArtifactEffect.VERIFIED_CHECK:
        return ETCLOVGLayer.VERIFICATION
    if effect is ArtifactEffect.WORKSPACE_DELTA:
        return ETCLOVGLayer.EXECUTION
    if canonical in _EXECUTION_TOOLS:
        return ETCLOVGLayer.EXECUTION
    if canonical in _GOVERNANCE_TOOLS:
        return ETCLOVGLayer.GOVERNANCE
    if canonical in _ORCHESTRATION_TOOLS:
        return ETCLOVGLayer.LIFECYCLE_ORCHESTRATION
    if canonical in _OBSERVABILITY_TOOLS:
        return ETCLOVGLayer.OBSERVABILITY
    if canonical in _CONTEXT_TOOLS:
        return ETCLOVGLayer.CONTEXT_MEMORY
    return ETCLOVGLayer.TOOLING


def attribute_failure_layer(tool_name: Optional[str]) -> Optional[ETCLOVGLayer]:
    """Attribute a single tool failure to its ETCLOVG layer (ADR-012 prong 2).

    The live-recovery counterpart to :meth:`HTIRTrace.failures_by_layer`: given the name of the
    tool that failed, return the harness layer to attribute the failure to, reusing the *same*
    inference as full-trace normalization (a failed write/shell is Execution, a failed read is
    Context-Memory, anything else is Tooling). Returns ``None`` when no tool name is available —
    e.g. a provider-level error with no tool in flight — so the caller falls back to its
    exception-type handling.
    """
    if not tool_name:
        return None
    # A failed tool produced no verifiable effect, so classify by name alone (effect = NONE).
    return _infer_layer(_canonical(tool_name), ArtifactEffect.NONE)


def _final_assistant_message(trace: "AgenticExecutionTrace") -> str:
    for msg in reversed(list(getattr(trace, "messages", []) or [])):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def _is_refusal(text: str) -> bool:
    first_line = text.strip().splitlines()[0].lower() if text.strip() else ""
    return any(p in first_line for p in _REFUSAL_PATTERNS)


def normalize(trace: "AgenticExecutionTrace") -> HTIRTrace:
    """Normalize an :class:`AgenticExecutionTrace` into an :class:`HTIRTrace`.

    Each tool call becomes a TOOL step (status from ``success``, effect + ETCLOVG layer inferred);
    a trailing assistant message becomes an ASSISTANT step (Governance/refused when it refuses,
    otherwise Lifecycle-Orchestration). File edits are represented by their originating write-tool
    calls, so they are not double-counted here.
    """
    steps: list[HTIRStep] = []
    for i, call in enumerate(getattr(trace, "tool_calls", []) or []):
        canonical = _canonical(getattr(call, "name", ""))
        success = bool(getattr(call, "success", False))
        effect = _artifact_effect(call, i)
        status = StepStatus.OK if success else StepStatus.FAILED
        layer = _infer_layer(canonical, effect)
        steps.append(
            HTIRStep(
                index=len(steps),
                role=Role.TOOL,
                status=status,
                effect=effect,
                layer=layer,
                tool_name=canonical,
                summary=f"{canonical} {'ok' if success else 'failed'}",
            )
        )

    final = _final_assistant_message(trace)
    if final:
        refused = _is_refusal(final)
        steps.append(
            HTIRStep(
                index=len(steps),
                role=Role.ASSISTANT,
                status=StepStatus.REFUSED if refused else StepStatus.OK,
                effect=ArtifactEffect.NONE,
                layer=ETCLOVGLayer.GOVERNANCE if refused else ETCLOVGLayer.LIFECYCLE_ORCHESTRATION,
                summary=final.strip().splitlines()[0][:120] if final.strip() else "",
            )
        )

    return HTIRTrace(
        task_id=getattr(trace, "task_id", ""),
        steps=tuple(steps),
        session_id=getattr(trace, "session_id", ""),
        benchmark=getattr(trace, "benchmark", ""),
        metadata={
            "turns": int(getattr(trace, "turns", 0) or 0),
            "tool_calls": len(getattr(trace, "tool_calls", []) or []),
        },
    )
