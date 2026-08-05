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

"""Effect-grounded completion gate (EVR-4, ADR-010, FEP-0008 Phase B).

Makes COMPLETE conditional on a **verifiable effect**: before ``AgenticLoop._evaluate`` returns
COMPLETE, the session must show either (a) a workspace state delta observed in ``tool_results``
(write/edit/create, or a non-readonly shell command), (b) a successfully executed verification
check (test/lint run), or — for Q&A / no-mutation tasks — (c) a grounded claim (read-tool
evidence or verified file references). A COMPLETE lacking an effect is downgraded to RETRY
(never FAIL) with reason ``completion-without-effect``.

Design decisions (recorded in ADR-010's revision history):

- **Session-scoped ledger, not turn-scoped**: a no-tool summary turn *after* effectful turns is a
  legitimate completion, so evidence accumulates across the loop run and resets with the loop's
  other per-run state (spin detector, criteria builder).
- **Bounded downgrades**: after ``max_downgrades`` (default 2) the gate annotates-and-allows
  (``effect_gate_exhausted``) instead of downgrading, so a too-strict effect detector can never
  trap the loop in a RETRY cycle.
- **Lenient Q&A grounding v1**: :class:`GroundedClaimChecker` accepts read-tool evidence or
  file references verified against the workspace; direct no-tool conversational answers pass.
  Strict claim extraction (``victor/tools/verification/``) is a later hardening step.
- **Opt-in, default off** per the flag-graduation policy — the gate is a strict no-op unless
  ``AgenticLoopConfig.enable_effect_gate`` is set (threaded from
  ``AgentSettings.effect_gated_completion`` / ``VICTOR_EFFECT_GATED_COMPLETION``).

Placement mirrors :mod:`victor.framework.rubric_completion` (EVR-3): completion machinery lives
in the framework layer, wired as a post-filter around the single EVALUATE seam so it applies to
*all* completion strategies (enhanced, rubric, hybrid, legacy) as a precondition.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.tools.base import AccessMode
from victor.tools.effective_access import resolve_effective_access
from victor.tools.tool_names import get_canonical_name

logger = logging.getLogger(__name__)

__all__ = [
    "EffectClass",
    "resolve_effect_gate_enabled",
    "EffectEvidence",
    "EffectLedger",
    "EffectGate",
    "EffectGateConfig",
    "GroundedClaimChecker",
    "classify_tool_result",
]


def resolve_effect_gate_enabled(settings: Any) -> bool:
    """Resolve the ADR-010 effect-gate flag: env override, then AgentSettings; default off.

    Shared by the buffered and streaming executors so both modes gate completion
    identically (ADR-012 parity).
    """
    env = os.environ.get("VICTOR_EFFECT_GATED_COMPLETION")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(getattr(getattr(settings, "agent", None), "effect_gated_completion", False))


class EffectClass(str, Enum):
    """The verifiable-effect classes ADR-010 recognizes."""

    WORKSPACE_DELTA = "workspace_delta"  # file write/edit/create or mutating shell command
    VERIFIED_CHECK = "verified_check"  # a test/lint/verification command ran successfully
    GROUNDED_CLAIM = "grounded_claim"  # Q&A answer grounded in tool evidence / real files


@dataclass(frozen=True)
class EffectEvidence:
    """One recorded piece of effect evidence from a turn's tool results."""

    effect_class: EffectClass
    tool_name: str  # canonical tool name
    artifact: Optional[str] = None  # file path (workspace delta) or command (verified check)
    turn_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect_class": self.effect_class.value,
            "tool_name": self.tool_name,
            "artifact": self.artifact,
            "turn_index": self.turn_index,
        }


# Canonical tool names whose success is a workspace delta (file created/modified). Alias-safe:
# inputs go through get_canonical_name() first (write_file→write, edit_file→edit, apply_patch→patch).
_FILE_WRITE_TOOLS = frozenset({"write", "create_file", "edit", "replace_in_file", "patch"})

# Shell-family tools: classified per invocation via resolve_effective_access + command inspection.
_SHELL_TOOLS = frozenset({"shell", "run_command"})

# The dedicated test-runner tool: a successful run is a verified check by construction.
_TEST_TOOLS = frozenset({"test"})

# Commands that constitute a verification check when they exit successfully.
_VERIFICATION_COMMAND_RE = re.compile(
    r"\b(pytest|unittest|tox|nox|ruff|mypy|pyright|flake8|pylint|eslint|tsc"
    r"|cargo\s+(test|check|clippy)|go\s+(test|vet)|npm\s+(test|run\s+lint)"
    r"|make\s+(test|lint|check)|black\s+--check|isort\s+--check)\b"
)

# First tokens of shell commands that are pure reads — a successful invocation grounds a claim
# but is NOT a workspace delta. Conservative allowlist; anything unrecognized stays a delta
# (the gate's job is to catch completion-without-ANY-effect, not to audit shell semantics).
_READONLY_COMMAND_HEADS = frozenset(
    {"ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "which", "wc", "echo", "stat", "du"}
)
_READONLY_GIT_SUBCOMMANDS = frozenset({"status", "diff", "log", "show", "branch", "blame"})


def _is_readonly_command(command: str) -> bool:
    tokens = command.split()
    if not tokens:
        return False
    head = tokens[0]
    if head in _READONLY_COMMAND_HEADS:
        return True
    if head == "git" and len(tokens) > 1 and tokens[1] in _READONLY_GIT_SUBCOMMANDS:
        return True
    return False


def classify_tool_result(
    result: Mapping[str, Any], turn_index: int = 0
) -> Optional[EffectEvidence]:
    """Classify one tool-result dict into an :class:`EffectEvidence`, or None.

    Accepts the ``TurnResult.tool_results`` dict shape (``tool_name``/``args``/``success``; the
    ``name``/``arguments`` spelling used by evaluation traces is also accepted). Names are
    resolved through :func:`get_canonical_name` so aliases classify identically, and shell
    invocations are narrowed per-invocation via :func:`resolve_effective_access` (a
    ``readonly=True`` shell run is a read, not a delta).

    Returns None for failed tools, read-only invocations, and tools with no effect semantics
    (reads still count as *grounding* evidence — the ledger tracks that separately).
    """
    tool_name = str(result.get("tool_name") or result.get("name") or "")
    if not tool_name:
        return None
    raw_args = result.get("args")
    if not isinstance(raw_args, Mapping):
        raw_args = result.get("arguments")
    args: Dict[str, Any] = dict(raw_args) if isinstance(raw_args, Mapping) else {}
    if not result.get("success"):
        return None

    canonical = get_canonical_name(tool_name)

    if canonical in _FILE_WRITE_TOOLS:
        artifact = args.get("file_path") or args.get("path") or None
        return EffectEvidence(
            EffectClass.WORKSPACE_DELTA,
            canonical,
            str(artifact) if artifact else None,
            turn_index,
        )

    if canonical in _TEST_TOOLS:
        artifact = args.get("cmd") or args.get("command") or args.get("path") or canonical
        return EffectEvidence(EffectClass.VERIFIED_CHECK, canonical, str(artifact), turn_index)

    if canonical in _SHELL_TOOLS:
        # Per-invocation narrowing: an enforced readonly shell run is a pure read.
        if resolve_effective_access(canonical, args) is AccessMode.READONLY:
            return None
        command = str(args.get("cmd") or args.get("command") or "")
        if _VERIFICATION_COMMAND_RE.search(command):
            return EffectEvidence(EffectClass.VERIFIED_CHECK, canonical, command[:200], turn_index)
        if _is_readonly_command(command):
            return None
        return EffectEvidence(EffectClass.WORKSPACE_DELTA, canonical, None, turn_index)

    return None


class EffectLedger:
    """Session-scoped accumulator of effect evidence fed from each turn's ``tool_results``.

    Session-scoped by design (ADR-010 decision 1): a no-tool summary turn after effectful turns
    is a legitimate completion, so evidence persists across turns and is cleared at the loop's
    per-run reset sites via :meth:`reset`.
    """

    def __init__(self) -> None:
        self._evidence: List[EffectEvidence] = []
        self._attempted_tool_calls = 0
        self._successful_tool_calls = 0

    def record_turn(self, action_result: Any, turn_index: int = 0) -> None:
        """Record effect evidence from one turn's ``TurnResult``-shaped action result."""
        tool_results = getattr(action_result, "tool_results", None) or []
        for result in tool_results:
            if not isinstance(result, Mapping):
                continue
            self._attempted_tool_calls += 1
            if result.get("success"):
                self._successful_tool_calls += 1
            evidence = classify_tool_result(result, turn_index)
            if evidence is not None:
                self._evidence.append(evidence)

    def candidate_effects(self) -> List[EffectEvidence]:
        """All recorded delta/check evidence (grounding reads are tracked separately)."""
        return list(self._evidence)

    def has_read_evidence(self) -> bool:
        """Whether any tool ran successfully this session (grounds a claim in tool feedback)."""
        return self._successful_tool_calls > 0

    @property
    def attempted_tool_calls(self) -> int:
        return self._attempted_tool_calls

    def summary(self) -> Dict[str, Any]:
        return {
            "evidence": [e.to_dict() for e in self._evidence],
            "attempted_tool_calls": self._attempted_tool_calls,
            "successful_tool_calls": self._successful_tool_calls,
        }

    def reset(self) -> None:
        self._evidence.clear()
        self._attempted_tool_calls = 0
        self._successful_tool_calls = 0


# File-reference tokens: path-ish strings with an extension (src/foo.py, ./a/b.md, Makefile-style
# names are out of scope for v1). Used by the lenient grounded-claim check.
_FILE_REFERENCE_RE = re.compile(r"(?<![\w./-])[\w./-]*\w+\.[A-Za-z][A-Za-z0-9]{0,9}(?![\w.])")


class GroundedClaimChecker:
    """Lenient v1 grounded-claim check for Q&A / no-mutation completions (ADR-010 decision 3).

    A claim is grounded when the session has read-tool evidence, or when the answer references
    at least one file that actually exists in the workspace. Strict claim extraction via
    ``victor/tools/verification/`` is deferred to a later hardening pass.
    """

    MAX_REFERENCES_CHECKED = 20

    def is_grounded(
        self,
        content: str,
        ledger: EffectLedger,
        workspace: Optional[Path],
    ) -> bool:
        if ledger.has_read_evidence():
            return True
        if workspace is not None and content:
            for match in _FILE_REFERENCE_RE.findall(content)[: self.MAX_REFERENCES_CHECKED]:
                candidate = Path(match)
                try:
                    resolved = candidate if candidate.is_absolute() else workspace / candidate
                    if resolved.exists():
                        return True
                except OSError:
                    continue
        return False


@dataclass
class EffectGateConfig:
    """Configuration for :class:`EffectGate`. Disabled by default (flag-graduation policy)."""

    enabled: bool = False
    max_downgrades: int = 2
    verify_artifacts: bool = True


class EffectGate:
    """Post-filter on the EVALUATE seam: COMPLETE requires a verifiable effect (ADR-010).

    Wired by ``AgenticLoop`` as ``record() → _evaluate_core() → apply()``. Strict no-op when
    disabled: :meth:`record` and :meth:`apply` early-return before any state mutation, so the
    flag-off behavior is byte-identical to the pre-gate loop (ADR-012 parity guarantee).
    """

    def __init__(
        self,
        config: Optional[EffectGateConfig] = None,
        workspace_resolver: Optional[Callable[[Dict[str, Any]], Optional[Path]]] = None,
    ) -> None:
        self.config = config or EffectGateConfig()
        self._workspace_resolver = workspace_resolver
        self.ledger = EffectLedger()
        self._claim_checker = GroundedClaimChecker()
        self._downgrades = 0
        self._turn_index = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def record(self, action_result: Any, state: Optional[Dict[str, Any]] = None) -> None:
        """Record one turn's tool results into the session ledger (no-op when disabled)."""
        if not self.enabled:
            return
        self._turn_index += 1
        try:
            self.ledger.record_turn(action_result, self._turn_index)
        except Exception:  # pragma: no cover — evidence recording must never break the loop
            logger.debug("[EffectGate] record failed", exc_info=True)

    def reset(self) -> None:
        """Reset session state (ledger + downgrade budget); called at the loop's reset sites."""
        if not self.enabled:
            return
        self.ledger.reset()
        self._downgrades = 0
        self._turn_index = 0

    async def apply(
        self,
        evaluation: EvaluationResult,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Gate a COMPLETE evaluation on effect evidence; pass everything else through.

        Downgrades an ungrounded COMPLETE to RETRY (never FAIL) with reason
        ``completion-without-effect``; after ``max_downgrades`` it annotates-and-allows.
        """
        if not self.enabled:
            return evaluation
        if not evaluation.should_complete:
            return evaluation

        # Team execution runs sub-loops whose effects are invisible to this ledger — bypass.
        if self._is_team_execution(action_result):
            evaluation.metadata["effect_gate"] = {"bypassed": "team_execution"}
            return evaluation

        expected = self._expected_effect_class(evaluation, action_result, state)
        satisfied_by = self._satisfying_evidence(expected, action_result, state)
        if satisfied_by is not None:
            evaluation.metadata["effect_gate"] = {
                "expected_effect_class": expected.value,
                "satisfied_by": satisfied_by,
            }
            return evaluation

        if self._downgrades >= max(0, int(self.config.max_downgrades)):
            # Downgrade budget exhausted: annotate-and-allow (ADR-010 decision 2) so a
            # too-strict effect detector can never trap the loop in a RETRY cycle.
            logger.info(
                "[EffectGate] completion-without-effect but downgrade budget exhausted "
                "(%d/%d) — allowing COMPLETE with annotation",
                self._downgrades,
                self.config.max_downgrades,
            )
            evaluation.metadata["effect_gate_exhausted"] = True
            evaluation.metadata["effect_gate"] = {
                "expected_effect_class": expected.value,
                **self.ledger.summary(),
            }
            return evaluation

        self._downgrades += 1
        logger.info(
            "[EffectGate] COMPLETE downgraded to RETRY: completion-without-effect "
            "(expected=%s, downgrade %d/%d)",
            expected.value,
            self._downgrades,
            self.config.max_downgrades,
        )
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=min(evaluation.score, 0.4),
            reason=(
                "completion-without-effect: COMPLETE claimed without a verifiable "
                f"{expected.value} effect (ADR-010)"
            ),
            metrics=dict(evaluation.metrics),
            metadata={
                **evaluation.metadata,
                "completion_without_effect": True,
                "expected_effect_class": expected.value,
                "effect_gate": self.ledger.summary(),
            },
        )

    # -- internals ---------------------------------------------------------------------------

    @staticmethod
    def _is_team_execution(action_result: Any) -> bool:
        response = getattr(action_result, "response", None)
        metadata = getattr(response, "metadata", None)
        return isinstance(metadata, dict) and metadata.get("execution_mode") == "team_execution"

    @staticmethod
    def _expected_effect_class(
        evaluation: EvaluationResult,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EffectClass:
        """Choose the effect class this completion must evidence.

        Q&A shortcuts, forced/deterministic completion signals, and non-mutation task types
        route through the lenient GROUNDED_CLAIM path; mutation task types require a
        WORKSPACE_DELTA (a VERIFIED_CHECK also satisfies — see ``_satisfying_evidence``).
        Unknown task types default lenient (GROUNDED_CLAIM) to avoid false blocks.
        """
        if getattr(action_result, "is_qa_response", False):
            return EffectClass.GROUNDED_CLAIM
        if evaluation.metadata.get("forced_task_completion") or evaluation.metadata.get(
            "successful_tool_progress"
        ):
            return EffectClass.GROUNDED_CLAIM
        task_type = str(state.get("task_type") or "").strip().lower().replace("-", "_")
        if task_type in _MUTATION_TASK_TYPES:
            return EffectClass.WORKSPACE_DELTA
        return EffectClass.GROUNDED_CLAIM

    def _satisfying_evidence(
        self,
        expected: EffectClass,
        action_result: Any,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return a summary of the evidence satisfying ``expected``, or None if unsatisfied."""
        verified = self._verified_effects(state)
        if verified:
            # A workspace delta or verified check satisfies ANY expected class — an effect
            # trivially grounds a claim too.
            return {"effects": [e.to_dict() for e in verified]}

        if expected is EffectClass.GROUNDED_CLAIM:
            if self.ledger.attempted_tool_calls == 0:
                # Direct no-tool conversational answer (Q&A shortcut) — nothing was claimed
                # to be done to the workspace; pass (lenient v1, ADR-010 decision 3).
                return {"grounding": "direct_answer_no_tools"}
            content = str(getattr(action_result, "content", "") or "")
            workspace = self._resolve_workspace(state)
            if self._claim_checker.is_grounded(content, self.ledger, workspace):
                return {"grounding": "read_evidence_or_file_references"}
        return None

    def _verified_effects(self, state: Dict[str, Any]) -> List[EffectEvidence]:
        """Delta/check evidence, with file artifacts stat-verified at gate time when possible.

        When the workspace is resolvable and ``verify_artifacts`` is on, a WORKSPACE_DELTA whose
        recorded file artifact no longer exists does not count. When the workspace cannot be
        resolved, evidence counts unverified (count-not-block).
        """
        effects = self.ledger.candidate_effects()
        if not effects or not self.config.verify_artifacts:
            return effects
        workspace = self._resolve_workspace(state)
        if workspace is None:
            return effects
        verified: List[EffectEvidence] = []
        for evidence in effects:
            if evidence.effect_class is EffectClass.WORKSPACE_DELTA and evidence.artifact:
                path = Path(evidence.artifact)
                try:
                    resolved = path if path.is_absolute() else workspace / path
                    if not resolved.exists():
                        logger.debug(
                            "[EffectGate] delta artifact %s not found at gate time — "
                            "not counting",
                            evidence.artifact,
                        )
                        continue
                except OSError:
                    continue
            verified.append(evidence)
        return verified

    def _resolve_workspace(self, state: Dict[str, Any]) -> Optional[Path]:
        if self._workspace_resolver is None:
            return None
        try:
            return self._workspace_resolver(state)
        except Exception:
            return None


# Task-type labels (TaskAnalyzer / perception vocabulary, mirrored from AgenticLoop's
# _map_to_task_type mutation buckets) for which a completion must evidence a workspace effect.
_MUTATION_TASK_TYPES = frozenset(
    {
        "action",
        "bug",
        "bug_fix",
        "ci_cd",
        "code_generation",
        "code_modification",
        "create",
        "create_simple",
        "debug",
        "debugging",
        "deploy",
        "deployment",
        "docker_compose",
        "dockerfile",
        "edit",
        "generation",
        "implement",
        "infrastructure",
        "issue_resolution",
        "kubernetes",
        "refactor",
        "setup",
        "terraform",
    }
)
