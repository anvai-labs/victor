# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Victor Agent Adapter for agentic benchmarks.

This module provides an adapter that connects Victor's AgentOrchestrator
to the AgenticBenchmarkRunner, enabling evaluation of Victor's agentic
capabilities on real-world coding tasks.

Features:
- Captures tool calls and file edits during execution
- Tracks multi-turn conversation metrics
- Generates patch diffs from file modifications
- Integrates with Victor's tool system

Usage:
    from victor.evaluation.agent_adapter import (
        VictorAgentAdapter,
        create_victor_agent_callback,
    )

    # Create adapter for a profile
    adapter = VictorAgentAdapter.from_profile("default", base_url="http://localhost:11434")

    # Use as callback for AgenticBenchmarkRunner
    callback = create_victor_agent_callback(adapter)
    result = await runner.run_task(task, callback, config)
"""

import asyncio
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from victor.config.settings import load_settings
from victor.framework.workspace import workspace_git_diff

# Use TYPE_CHECKING to avoid circular import at runtime
# Chain: orchestrator → code_correction_middleware → evaluation → agent_adapter → orchestrator
if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.framework.session_config import SessionConfig
else:
    # Deferred import at runtime - only when actually needed
    AgentOrchestrator = None  # Will be imported lazily in from_profile()
from victor.agent.task_completion import TaskCompletionDetector
from victor.evaluation.agentic_harness import (
    AgenticExecutionTrace,
    FileEdit,
    EvalToolCall,
)
from victor.evaluation.correction.metrics import (
    CorrectionMetricsCollector,
)
from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
from victor.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


# Tools a benchmark session must have enabled. The `fs` domain has been
# removed — `read`/`edit`/`write` are now first-class tools with named
# parameters. `ls`/`find` route to `shell`. The ``code`` tool subsumes code
# search/grep (the live code-intelligence surface).
_BENCHMARK_BASE_TOOLS = frozenset(
    {
        "read",
        "edit",
        "write",
        "code",
        "shell",
    }
)

# Canonical benchmark toolset (the *desired* set): base tools plus ``graph``
# (call-graph analysis: callers/callees/impact). ``graph`` is the optional
# victor-coding tool; whether it's actually registered is a runtime concern
# surfaced by get_benchmark_tool_readiness(), and prompt guidance only
# describes it when it resolves (see _graph_tool_available).
_BENCHMARK_TOOL_ALLOWLIST = _BENCHMARK_BASE_TOOLS | {"graph"}

# Backward-compat alias (existing callers/tests reference the constant).
BENCHMARK_TOOL_ALLOWLIST = _BENCHMARK_TOOL_ALLOWLIST

# Tool names that modify files — drives the adapter's files_modified tracking
# (and thus whether a task is credited with producing a patch). Derived from
# CORE tool metadata (AccessMode.WRITE on each @tool) via _file_modifying_tools()
# so it stays in sync with the tool declarations; falls back to a hardcoded set
# if the metadata registry isn't populated yet (import-time / early boot).
_FILE_MODIFYING_TOOLS_FALLBACK = frozenset(
    {"edit", "write", "file_write", "file_edit", "edit_file", "patch"}
)
_file_modifying_tools_cache: Optional[frozenset[str]] = None


def _file_modifying_tools() -> frozenset[str]:
    """File-modifying tool names, from core AccessMode.WRITE metadata.

    Single source of truth: queries ``ToolMetadataRegistry.get_tools_by_access_mode
    (WRITE)`` and UNIONS it with a static fallback so the known tools (edit/write)
    are always present (the registry may be empty before tools register) and any
    newly-declared WRITE tools are picked up automatically.
    """
    global _file_modifying_tools_cache
    if _file_modifying_tools_cache is not None:
        return _file_modifying_tools_cache
    names = _FILE_MODIFYING_TOOLS_FALLBACK
    try:
        from victor.tools.enums import AccessMode
        from victor.tools.metadata import ToolMetadataRegistry

        reg_names = frozenset(
            ToolMetadataRegistry.get_instance().get_tools_by_access_mode(AccessMode.WRITE)
        )
        names = names | reg_names  # superset: known tools + any new WRITE tools
    except Exception:
        pass
    _file_modifying_tools_cache = names
    return names


def _graph_tool_available() -> bool:
    """True if the optional victor-coding ``graph`` tool resolves.

    ``graph`` (call-graph analysis: callers/callees/impact) helps the agent
    locate bugs in large codebases like Django/astropy, so it is added to the
    benchmark tool set when available. Environments without victor-coding (or
    without a built graph store) degrade gracefully — graph is simply omitted.
    """
    try:
        from victor.core.utils.capability_loader import load_graph_tool_module

        load_graph_tool_module()
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class PromptOptimizationBinding:
    """Pin one exact prompt candidate into the live runtime for targeted evaluation."""

    section_name: str
    prompt_candidate_hash: str
    provider: Optional[str] = None


@dataclass
class AdapterConfig:
    """Configuration for the Victor agent adapter.

    Defaults are based on ACTION complexity from ComplexityBudget,
    which is appropriate for benchmark tasks that require multiple tool calls.
    """

    # Defaults optimized for SWE-bench with slow models (DeepSeek, Qwen, Mixtral)
    # ACTION: tool_budget=50, max_turns=20, timeout_seconds=1200
    max_turns: int = 20  # Maximum conversation turns (fewer, longer turns)
    tool_budget: int = 50  # Maximum tool calls (ACTION complexity)
    max_tool_calls: int = 50  # Alias for tool_budget (backwards compat)
    total_timeout: int = 1200  # Total task timeout: 20 minutes for slow models
    # Per-turn floor. Benchmark data showed agents over-explore and die
    # mid-turn-1 because a high floor (240s) left too few turns within the
    # task budget — at 1200s total, 240s/turn → 5 turns; 120s/turn → 10
    # turns. More turns = more chances to reach the edit phase. Cloud
    # providers (grok/zai) complete one orchestrator.chat() (incl. its
    # internal agentic loop) well under 120s.
    min_turn_timeout: int = 120  # 2 minutes per turn
    track_file_edits: bool = True
    track_diffs: bool = True
    working_dir: Optional[Path] = None

    # Correction metrics tracking
    track_corrections: bool = True  # Enable correction metrics collection
    keep_correction_attempts: bool = False  # Store individual correction records

    # Optional prompt candidate binding for targeted benchmark/eval runs.
    prompt_binding: Optional[PromptOptimizationBinding] = None

    # Closed-loop verify-and-retry gate (PR2): when a verify_fn is supplied to
    # execute_task, the loop runs the FAIL_TO_PASS tests after the agent claims
    # done; if they don't all pass, the failure output is fed back as the next
    # continuation message and the loop continues, up to this many verify
    # retries (then the partial result is accepted). 0 disables the gate.
    # DEFAULT OFF (0): validation showed 0 conversions for deepseek-chat at
    # ~3 containers/failing-task cost — opt-in via --verify-retries for
    # stronger models that can capitalize on a retry.
    max_verify_retries: int = 0

    @property
    def timeout_per_turn(self) -> int:
        """Calculate per-turn timeout with minimum enforcement (P2: Timeout Fix).

        Uses SafeTimeoutPolicy to ensure at least min_turn_timeout seconds per turn,
        even for slow models like DeepSeek.
        """
        from victor.evaluation.timeout_calculator import SafeTimeoutPolicy

        policy = SafeTimeoutPolicy(min_turn_timeout=self.min_turn_timeout)
        return policy.calculate(self.total_timeout, self.max_turns)


@dataclass(frozen=True)
class BenchmarkToolReadiness:
    """Live benchmark tool readiness for a session-scoped tool registry."""

    required_tools: tuple[str, ...]
    enabled_tools: tuple[str, ...]
    missing_tools: tuple[str, ...] = ()
    disabled_tools: tuple[str, ...] = ()
    # Tools that are reported when missing/disabled but do NOT fail readiness.
    # graph is demand-loaded (DEMAND_TOOL_SPECS, not bootstrap): it registers
    # during the agent's first turn when the prompt mentions it, so it is
    # legitimately absent at session-init readiness time. Treating it as
    # optional avoids a false abort without skipping it.
    optional_tools: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """Whether all REQUIRED tools are present and enabled.

        Optional tools (e.g. graph) are surfaced in missing/disabled but do
        not gate readiness — they are demand-loaded during the run.
        """
        optional = set(self.optional_tools)
        hard_missing = any(t not in optional for t in self.missing_tools)
        hard_disabled = any(t not in optional for t in self.disabled_tools)
        return not hard_missing and not hard_disabled


def _summarize_eval_tool_calls(tool_calls: List[EvalToolCall]) -> Dict[str, int]:
    """Summarize tool usage counts for benchmark telemetry."""
    counts = Counter(call.name for call in tool_calls)
    return {
        "read_calls": int(counts.get("read", 0)),
        "edit_calls": int(counts.get("edit", 0)),
        "write_calls": int(counts.get("write", 0)),
        "code_calls": int(counts.get("code", 0)),
        "shell_calls": int(counts.get("shell", 0)),
    }


class VictorAgentAdapter:
    """Adapter that connects Victor's orchestrator to agentic benchmarks.

    This adapter wraps the AgentOrchestrator and captures execution traces
    suitable for benchmark evaluation.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        config: Optional[AdapterConfig] = None,
    ):
        """Initialize the adapter.

        Args:
            orchestrator: Victor's AgentOrchestrator instance
            config: Adapter configuration
        """
        self.orchestrator = orchestrator
        self.config = config or AdapterConfig()

        # Disable prompt optimization (GEPA) for benchmark sessions — it
        # thrashes the KV cache, wastes API calls on reflection, and
        # invalidates the evaluation (evolving prompts while measuring them).
        # Prompt optimization is user-driven (CLI: `victor benchmark evolve`),
        # never automatic during evaluation.
        try:
            import os

            os.environ["VICTOR_PROMPT_OPTIMIZATION_ENABLED"] = "false"
        except Exception:
            pass

        # Execution tracking
        self._served_prompt_identities: Dict[tuple, Dict[str, Any]] = {}
        self._tool_calls: List[EvalToolCall] = []
        self._file_edits: List[FileEdit] = []
        # The ground-truth combined patch (git diff of the workspace after the
        # run) — the "completing effect" for a code-fix task. Captured in
        # execute_task and surfaced via get_conversation_trace so the execution
        # manifest (and downstream judge views) carry the actual applied diff,
        # not just the fallible per-edit capture in _file_edits.
        self._generated_patch: str = ""
        self._messages: List[Dict[str, str]] = []
        self._turns: int = 0
        self._file_snapshots: Dict[str, str] = {}  # path -> content before edit

        # Action-bias: exploration budget. Counts consecutive tool calls that
        # do NOT modify a file (read/search/grep/shell) since the last
        # successful edit/write. Drives escalating continuation nudges so the
        # agent doesn't over-explore (the #1 failure mode: 22/32 failures
        # were "no valid patch" — the agent read 10-18 files and never
        # edited). Reset to 0 on any successful file-modifying tool.
        self._exploration_calls: int = 0

        # Docker-agnostic "agent has acted" signal: True once any edit/write/
        # patch tool call succeeds this task. Used in place of the host-side
        # _has_file_modifications() (which is always False in docker eval —
        # edits land in-container, not on the host snapshots) for the action-
        # bias escalation branch and the test-pass force-complete.
        self._made_edit_tool_call: bool = False

        # Correlation spine for this task. Set in execute_task and stamped on
        # every decision (log_decision reads get_session_id()) so the per-task
        # execution manifest can join decisions to the task outcome (reward).
        self._task_session_id: str = ""

        # Circuit breaker: tracks consecutive same-error failures per tool.
        # After N failures (default 3), the tool is auto-disabled for the
        # rest of the task — prevents the model from wasting 20+ turns
        # retrying a persistently broken tool (e.g. graph import error).
        self._tool_failure_threshold: int = 3
        self._tool_failures: dict[str, str] = {}  # tool_name -> last error hash
        self._tool_failure_counts: dict[str, int] = {}  # consecutive count
        self._disabled_tools: set[str] = set()

        # Correction metrics collector
        self._metrics_collector: Optional[CorrectionMetricsCollector] = None
        if self.config.track_corrections:
            self._metrics_collector = CorrectionMetricsCollector(
                keep_attempts=self.config.keep_correction_attempts
            )

        # Task completion detector (uses framework's detection, not gaming code)
        self._completion_detector = TaskCompletionDetector()

        # Hook into tool execution via ToolRegistry hooks
        # The orchestrator's streaming handler uses self.tools.execute() directly,
        # bypassing the ToolPipeline. We must hook into the ToolRegistry instead.
        if hasattr(orchestrator, "tools") and orchestrator.tools:
            orchestrator.tools.register_before_hook(
                self._on_tool_start_hook, critical=False, name="AgentAdapter.tool_start"
            )
            orchestrator.tools.register_after_hook(
                self._on_tool_complete_hook,
                critical=False,
                name="AgentAdapter.tool_complete",
            )
            logger.info("[AgentAdapter] Registered ToolRegistry hooks for tool call tracking")
        else:
            logger.warning("[AgentAdapter] Could not register hooks - ToolRegistry not found")

        self._apply_prompt_binding()

    def _apply_prompt_binding(self) -> None:
        """Apply any explicit prompt binding to the live optimization injector.

        Targeted evaluations must bind the runtime candidate up front so the
        executed prompt content matches the prompt identity saved in artifacts.
        """
        binding = self.config.prompt_binding
        if binding is None:
            return

        injector = getattr(self.orchestrator, "_optimization_injector", None)
        if injector is None or not hasattr(injector, "bind_prompt_candidate"):
            raise RuntimeError(
                "Prompt optimization binding requires an orchestrator optimization injector"
            )

        provider = (
            binding.provider
            or getattr(self.orchestrator, "provider_name", None)
            or getattr(getattr(self.orchestrator, "provider", None), "name", None)
        )
        injector.bind_prompt_candidate(
            section_name=binding.section_name,
            prompt_candidate_hash=binding.prompt_candidate_hash,
            provider=str(provider or "").strip(),
            strict=True,
        )

    @staticmethod
    def _has_test_pass_signal(content: str) -> bool:
        """Detect whether the agent's response indicates tests passed.

        Checks for common test-pass language + the prompt's explicit completion
        phrase. Used by the test-pass→force-complete path to break the
        "retry forever" loop.
        """
        lower = (content or "").lower()
        test_pass_markers = [
            # Explicit completion phrase from the benchmark prompt.
            "fix is complete and verified",
            # Pytest/unittest output patterns.
            "all tests passed",
            "tests passed",
            "test passed",
            "0 failed",
            "no failures",
            "0 failed,",
            "passed, 0 failed",
            # Broader phrasings models actually use.
            "tests are passing",
            "all tests succeeded",
            "pytest passed",
            "test suite passed",
            "test passes",
            "tests succeed",
            "no test failures",
            "all passing",
            "verification successful",
            "fix works",
            "fix is working",
            "issue is resolved",
        ]
        return any(marker in lower for marker in test_pass_markers)

    def _sample_served_prompt_identities(self) -> None:
        """Fold the runtime's current per-turn served identities into the task record.

        ``_last_served_prompt_identities`` is overwritten every turn — correct for
        reward attribution, which only cares about the turn that just ended, but a
        task spans many turns. Sampling on each tool call (plus once at payload
        time) unions them without the runtime having to track a second lifetime.
        """
        intelligence = getattr(self.orchestrator, "runtime_intelligence", None)
        identities = getattr(intelligence, "_last_served_prompt_identities", None) or []
        for identity in identities:
            candidate_hash = getattr(identity, "prompt_candidate_hash", None)
            if not candidate_hash:
                continue
            section = getattr(identity, "prompt_section_name", None) or getattr(
                identity, "section_name", None
            )
            key = (section, getattr(identity, "provider", None), candidate_hash)
            if key not in self._served_prompt_identities:
                self._served_prompt_identities[key] = identity.to_metadata()

    def get_served_prompt_identities(self) -> List[Dict[str, Any]]:
        """Prompt candidates actually served during this task.

        Written into the evaluation artifact so ``seed_from_evaluations()`` can
        attribute the task's pass/fail to the candidates that produced it. Until
        this existed, only an explicit ``--prompt-candidate-hash`` A/B stamped
        identity, so ordinary benchmark runs — the overwhelming majority —
        carried ground-truth outcomes that no candidate could ever claim.

        Falls back to the configured binding when nothing was observed (a
        targeted run still knows what it bound).
        """
        self._sample_served_prompt_identities()
        if self._served_prompt_identities:
            return list(self._served_prompt_identities.values())

        binding = self.config.prompt_binding
        if binding is None:
            return []
        return [
            {
                "provider": binding.provider,
                "prompt_candidate_hash": binding.prompt_candidate_hash,
                "section_name": binding.section_name,
                "prompt_section_name": binding.section_name,
                "strategy_name": None,
                "source": "explicit_binding",
            }
        ]

    def get_conversation_trace(self) -> Dict[str, Any]:
        """Serialize the full execution trace for post-hoc analysis.

        Returns the conversation messages, tool calls (with args + durations),
        and file edits (with diffs) captured during the task. Written to the
        results JSON so the trace survives process exit.
        """
        return {
            "session_id": self._task_session_id,
            "task_id": self.config.working_dir.name if self.config.working_dir else "",
            "messages": [
                {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
                for m in self._messages[-50:]  # last 50 messages (bounded)
            ],
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": str(tc.arguments)[:500],
                    "duration_s": round(getattr(tc, "duration", 0) or 0, 2),
                }
                for tc in self._tool_calls[-100:]  # last 100 calls (bounded)
            ],
            "file_edits": [
                {
                    "path": getattr(e, "path", ""),
                    "diff": str(getattr(e, "diff", ""))[:2000],  # bounded
                }
                for e in self._file_edits[-20:]  # last 20 edits
            ],
            # Ground-truth combined diff (git diff of the workspace) — the
            # completing effect. Bounded; the definitive artifact for a
            # code-fix judge view when per-edit capture (_file_edits) is empty.
            "generated_patch": self._generated_patch[:20000],
            "turns": self._turns,
        }

    def _on_tool_start_hook(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Hook called by ToolRegistry before tool execution."""
        self._sample_served_prompt_identities()
        self._on_tool_start(tool_name, arguments)

    def _on_tool_complete_hook(self, result: Any) -> None:
        """Hook called by ToolRegistry after tool execution."""
        self._on_tool_complete(result)

    def _on_tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Track tool call start."""
        start_time = time.time()
        logger.info(f"[AgentAdapter] Tool started: {tool_name}")

        # Record tool call with start time for duration tracking
        self._tool_calls.append(
            EvalToolCall(
                name=tool_name,
                arguments=arguments,
                timestamp=start_time,
            )
        )
        # Store start time for duration calculation
        self._current_tool_start = start_time

        # Snapshot file before edit
        if self.config.track_file_edits and tool_name in (
            "file_write",
            "file_edit",
            "edit_file",
            "patch",
        ):
            path = arguments.get("path") or arguments.get("file_path", "")
            if path and self.config.working_dir:
                full_path = self.config.working_dir / path
                if full_path.exists():
                    try:
                        self._file_snapshots[path] = full_path.read_text()
                    except Exception:
                        pass

    def _on_tool_complete(self, result: Any) -> None:
        """Track tool call completion."""
        # Calculate duration
        end_time = time.time()
        duration_ms = 0
        if hasattr(self, "_current_tool_start"):
            duration_ms = int((end_time - self._current_tool_start) * 1000)

        # Update last tool call with result
        if self._tool_calls:
            tool_name = self._tool_calls[-1].name
            success = getattr(result, "success", True) if hasattr(result, "success") else True
            logger.info(
                f"[AgentAdapter] Tool completed: {tool_name} "
                f"(duration={duration_ms}ms, success={success})"
            )
            last_call = self._tool_calls[-1]
            if hasattr(result, "success"):
                last_call.success = result.success
                error = getattr(result, "error", None)
                output = getattr(result, "output", None)
                legacy_result = getattr(result, "result", None)
                rendered = legacy_result if result.success else error
                if rendered is None:
                    rendered = error if result.success else legacy_result
                if rendered is None:
                    rendered = output
                last_call.result = str(rendered) if rendered is not None else None
            elif hasattr(result, "tool_name"):
                last_call.success = getattr(result, "success", True)
                last_call.result = getattr(result, "result", None)

            # Record tool result for completion detection (framework integration)
            result_dict = {
                "success": getattr(result, "success", True),
                "path": last_call.arguments.get("path") or last_call.arguments.get("file_path"),
            }
            self._completion_detector.record_tool_result(last_call.name, result_dict)

            # Circuit breaker: if a tool fails repeatedly with the same error,
            # auto-disable it for the rest of this task to prevent turn waste
            # (e.g. graph tool with an import error → 24 wasted retries).
            if not success:
                self._record_tool_failure(
                    tool_name,
                    last_call.result or "unknown error",
                )

            # Action-bias: track exploration budget. A SUCCESSFUL file-modifying
            # call resets the counter (the agent took action); anything else
            # (read/search/grep/shell, or a FAILED edit) increments it. The
            # continuation-message logic escalates nudges as the counter grows
            # so the agent converges on editing instead of over-exploring.
            if success and tool_name in _file_modifying_tools():
                self._exploration_calls = 0
                # Docker-agnostic "agent acted" signal: latches True the first
                # time any edit/write/patch tool succeeds. Host-side
                # _has_file_modifications() is blind to in-container edits
                # (always False in docker eval), so escalation/force-complete
                # gate on THIS instead.
                self._made_edit_tool_call = True
            else:
                self._exploration_calls += 1

        # Track file edit after completion
        if self.config.track_file_edits and self._tool_calls:
            last_call = self._tool_calls[-1]
            if last_call.name in _file_modifying_tools():
                path = last_call.arguments.get("path") or last_call.arguments.get("file_path", "")
                if path and self.config.working_dir:
                    self._capture_file_edit(path, last_call.name)

    def _record_tool_failure(self, tool_name: str, error: str) -> None:
        """Circuit breaker: track consecutive failures and auto-disable.

        After N consecutive failures with the SAME error, the tool is removed
        from the orchestrator's enabled set for the rest of this task. This
        prevents the model from wasting 20+ turns on a persistently broken
        tool (e.g. graph import error on every call). Reset per task in
        reset().
        """
        # Log the full error text (truncated) on EVERY failure. The circuit
        # breaker below only hashes the error, so without this the failure
        # cause is invisible in logs — e.g. an edit returning an 8110-char
        # error payload would leave no trace of WHY it failed.
        logger.warning(
            "[AgentAdapter] Tool '%s' failed (turn %d, call #%d): %.500s",
            tool_name,
            self._turns,
            len(self._tool_calls),
            error or "(no error text)",
        )

        import hashlib

        error_hash = hashlib.md5(error[:200].encode()).hexdigest()
        if self._tool_failures.get(tool_name) == error_hash:
            self._tool_failure_counts[tool_name] = self._tool_failure_counts.get(tool_name, 1) + 1
        else:
            self._tool_failures[tool_name] = error_hash
            self._tool_failure_counts[tool_name] = 1

        count = self._tool_failure_counts[tool_name]
        if count >= self._tool_failure_threshold and tool_name not in self._disabled_tools:
            self._disabled_tools.add(tool_name)
            logger.warning(
                "[AgentAdapter] Circuit breaker: disabling '%s' after %d "
                "consecutive failures (error: %s). The tool will be unavailable "
                "for the rest of this task.",
                tool_name,
                count,
                error[:100],
            )
            try:
                current = set(getattr(self.orchestrator, "_enabled_tools", None) or set())
                current.discard(tool_name)
                self.orchestrator.set_enabled_tools(current)
                self.orchestrator.tools.disable_tool(tool_name)
            except Exception:
                pass

    def _capture_file_edit(self, path: str, action: str) -> None:
        """Capture file edit after tool completion."""
        if not self.config.working_dir:
            return

        # Only capture repo-relative files. Scratch files the agent writes
        # outside the workspace (e.g. /tmp test scripts) must not appear in the
        # patch — SWE-bench's ``git apply`` rejects them with "invalid path".
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts:
            return
        full_path = self.config.working_dir / path
        try:
            full_path.resolve().relative_to(self.config.working_dir.resolve())
        except (ValueError, OSError):
            return

        before_content = self._file_snapshots.get(path, "")
        after_content = ""

        if full_path.exists():
            try:
                after_content = full_path.read_text()
            except Exception:
                return

        # Determine action type
        if not before_content and after_content:
            edit_action = "create"
        elif before_content and not after_content:
            edit_action = "delete"
        else:
            edit_action = "modify"

        # Generate diff
        diff = ""
        if self.config.track_diffs and before_content != after_content:
            try:
                import difflib

                diff_lines = difflib.unified_diff(
                    before_content.splitlines(keepends=True),
                    after_content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
                diff = "".join(diff_lines)
            except Exception:
                pass

        self._file_edits.append(
            FileEdit(
                path=path,
                action=edit_action,
                before_content=before_content,
                after_content=after_content,
                diff=diff,
            )
        )

    @classmethod
    def benchmark_tool_allowlist(cls) -> frozenset[str]:
        """Return the canonical benchmark tool allowlist (the *desired* toolset).

        Always includes the base tools plus ``graph`` (call-graph analysis:
        callers/callees/impact) — graph helps the agent locate bugs in large
        codebases. This is the declarative desired set; whether ``graph`` is
        actually registered in a given session is a runtime concern surfaced
        by :meth:`get_benchmark_tool_readiness` (missing/disabled), and the
        prompt guidance only describes graph when it resolves. Environments
        without victor-coding enable a graph tool that simply isn't registered
        (harmless no-op in ``set_enabled_tools``).
        """
        return _BENCHMARK_TOOL_ALLOWLIST

    def get_benchmark_tool_readiness(
        self,
        required_tools: Optional[set[str]] = None,
    ) -> BenchmarkToolReadiness:
        """Inspect the live session tool registry for benchmark-critical tools.

        By default the hard requirement is the base tool set; ``graph`` is
        optional (demand-loaded during the run, so legitimately absent at
        session-init). Pass ``required_tools`` to override.
        """
        if required_tools is None:
            required = set(_BENCHMARK_BASE_TOOLS)
            optional = {"graph"}
        else:
            required = set(required_tools)
            optional = set()
        registry = getattr(self.orchestrator, "tools", None)
        required_sorted = tuple(sorted(required))

        if registry is None or not hasattr(registry, "list_tools"):
            return BenchmarkToolReadiness(
                required_tools=required_sorted,
                enabled_tools=(),
                missing_tools=required_sorted,
                optional_tools=tuple(sorted(optional)),
            )

        try:
            tools = registry.list_tools(only_enabled=False)
        except TypeError:
            tools = registry.list_tools(False)
        except Exception:
            tools = []

        registered = {
            name
            for tool in tools or []
            for name in [getattr(tool, "name", None)]
            if isinstance(name, str) and name
        }
        # Check the full desired set (base ∪ optional) for visibility, but
        # `ready` ignores optional tools (see BenchmarkToolReadiness.ready).
        checked = required | optional
        missing = sorted(checked - registered)

        enabled = []
        disabled = []
        for tool_name in sorted(checked & registered):
            is_enabled = True
            if hasattr(registry, "is_tool_enabled"):
                try:
                    is_enabled = bool(registry.is_tool_enabled(tool_name))
                except Exception:
                    is_enabled = False
            if is_enabled:
                enabled.append(tool_name)
            else:
                disabled.append(tool_name)

        return BenchmarkToolReadiness(
            required_tools=required_sorted,
            enabled_tools=tuple(enabled),
            missing_tools=tuple(missing),
            disabled_tools=tuple(disabled),
            optional_tools=tuple(sorted(optional)),
        )

    def get_partial_trace(self) -> Dict[str, Any]:
        """Get partial trace data for timeout scenarios.

        Returns a dict with current state that can be used to populate
        TaskResult even when the task didn't complete normally.
        """
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if hasattr(self.orchestrator, "get_token_usage"):
            usage = self.orchestrator.get_token_usage()
            token_usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
        # Fallback: read from _cumulative_token_usage directly if metrics returns 0
        if token_usage["total_tokens"] == 0:
            cum = getattr(self.orchestrator, "_cumulative_token_usage", {})
            if cum.get("total_tokens", 0) > 0:
                token_usage = {
                    "input_tokens": cum.get("prompt_tokens", 0),
                    "output_tokens": cum.get("completion_tokens", 0),
                    "total_tokens": cum.get("total_tokens", 0),
                }

        # Get extended token fields from cumulative dict
        cum = getattr(self.orchestrator, "_cumulative_token_usage", {})

        return {
            "code": "",  # No code generated on timeout
            "tokens_input": token_usage["input_tokens"],
            "tokens_output": token_usage["output_tokens"],
            "tokens_used": token_usage["total_tokens"],
            "cached_tokens": cum.get("cached_tokens", 0),
            "reasoning_tokens": cum.get("reasoning_tokens", 0),
            "cost_usd_micros": cum.get("cost_usd_micros", 0),
            "tool_calls": len(self._tool_calls),
            "turns": self._turns,
            "file_edits": len(self._file_edits),
            "files_modified": [getattr(e, "path", "") for e in self._file_edits[:10]],
            **_summarize_eval_tool_calls(self._tool_calls),
            "conversation_trace": self.get_conversation_trace(),
            "prompt_identities": self.get_served_prompt_identities(),
        }

    def reset(self) -> None:
        """Reset tracking state for new task."""
        self._tool_calls = []
        self._file_edits = []
        self._generated_patch = ""
        self._messages = []
        self._turns = 0
        self._file_snapshots = {}
        self._task_session_id = ""
        self._exploration_calls = 0
        self._made_edit_tool_call = False
        # Served prompt identities are per-task evidence; carrying them across
        # tasks would credit one task's candidates with another task's outcome.
        self._served_prompt_identities = {}
        # Reset circuit breaker for new task
        self._tool_failures = {}
        self._tool_failure_counts = {}
        self._disabled_tools = set()
        self.orchestrator.reset_conversation()

        # Clear code_search index cache for task isolation
        try:
            from victor.core.utils.capability_loader import load_code_search_module

            load_code_search_module().clear_index_cache()
        except ImportError:
            pass

        # Reset token usage for fresh tracking per task
        if hasattr(self.orchestrator, "reset_token_usage"):
            self.orchestrator.reset_token_usage()

        # Reset metrics collector for new task
        if self.config.track_corrections:
            self._metrics_collector = CorrectionMetricsCollector(
                keep_attempts=self.config.keep_correction_attempts
            )

        # Reset completion detector for new task
        self._completion_detector.reset()

        # Clear RL repo context between tasks
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            get_rl_coordinator().set_repo_context(None)
        except Exception:
            pass

        # Reset conversation state machine (clear transition history)
        state_mgr = getattr(
            self.orchestrator,
            "_conversation_state",
            getattr(self.orchestrator, "conversation_state", None),
        )
        if state_mgr and hasattr(state_mgr, "reset"):
            state_mgr.reset()

        # Reset embedding service to prevent stale connections
        try:
            from victor.storage.embeddings.service import EmbeddingService

            if EmbeddingService._instance is not None and hasattr(
                EmbeddingService._instance, "_shutdown"
            ):
                EmbeddingService._instance._shutdown = False
        except Exception:
            pass

        # Force GC to release held resources
        import gc

        gc.collect()

    async def execute_task(
        self,
        task: BenchmarkTask,
        workspace_dir: Path,
    ) -> AgenticExecutionTrace:
        """Execute an agentic task and return execution trace.

        Uses the framework's PromptEnrichmentService and TaskCompletionDetector
        rather than benchmark-specific gaming code. This tests Victor as users
        would actually use it.

        Args:
            task: The benchmark task to execute
            workspace_dir: Directory where task files are located

        Returns:
            AgenticExecutionTrace with tool calls, file edits, and messages
        """
        self.reset()
        self.config.working_dir = workspace_dir

        # Stamp a per-task session_id on the correlation spine. log_decision()
        # (called during the agent loop) reads get_session_id() and writes it on
        # every decision record, so the per-task execution manifest can join
        # decisions to this task's outcome (reward) for classifier training.
        # 1:1 with task_id — one fresh UUID per task.
        import uuid as _uuid

        from victor.core.context import set_session_id

        self._task_session_id = _uuid.uuid4().hex
        set_session_id(self._task_session_id)
        # Also stamp the orchestrator's persistent session id. The contextvar
        # above can be lost when decisions cross a thread/loop boundary
        # (decide_sync runs on a fresh thread); orchestrator.chat re-stamps the
        # contextvar from this attr at chat start so every decision the loop
        # logs carries the task's session_id (FEP-0012 spine integrity).
        self.orchestrator.active_session_id = self._task_session_id

        # CRITICAL: Set workspace BEFORE any orchestrator operations
        # This ensures tools like file read/write, grep, etc. operate on the benchmark
        # repo rather than Victor's own codebase. Uses framework method for proper
        # encapsulation of project context updates.
        self.orchestrator.set_workspace(workspace_dir)

        # Also chdir so tools that resolve path='.' against cwd work correctly.
        # Tools like code_search._literal_search use os.walk(path) where path='.'
        # resolves to cwd, not the framework's project root. This is a pragmatic
        # fix; the proper solution is for tools to use get_project_paths().
        import os

        os.chdir(workspace_dir)

        # Start benchmarks in READING stage — agent must explore before editing.
        # The stage machine will naturally progress to EXECUTION after the agent
        # reads enough files (MAX_READS_WITHOUT_EDIT) or uses edit/write tools.
        try:
            from victor.core.shared_types import ConversationStage

            state_mgr = getattr(
                self.orchestrator,
                "_conversation_state",
                getattr(self.orchestrator, "conversation_state", None),
            )
            if state_mgr and hasattr(state_mgr, "_transition_to"):
                state_mgr._transition_to(ConversationStage.READING, confidence=0.8)
                # Allow natural stage progression — do NOT lock MIN_TOOLS
        except Exception:
            pass

        # Restrict tools to a focused set for benchmark tasks.
        # The default semantic selector broadcasts 15+ tools which causes model
        # decision fatigue — models default to text responses instead of tool use.
        # A minimal set forces decisive tool use at each stage. ``graph`` is
        # added when the optional victor-coding graph tool resolves.
        is_deep_research = task.benchmark == BenchmarkType.DR3_EVAL
        if is_deep_research:
            benchmark_tools = {"read", "write", "shell", "web_search", "web_fetch"}
        else:
            benchmark_tools = set(self.benchmark_tool_allowlist())
        graph_enabled = not is_deep_research and "graph" in benchmark_tools
        # Register demand-only curated tools (e.g. graph) into the live ToolRegistry
        # before set_enabled_tools. graph is in DEMAND_TOOL_SPECS (not BOOTSTRAP),
        # so it's never registered at init — _union_curated_enabled() can't find it
        # in list_tools(only_enabled=False) → it never reaches the LLM schema. Load
        # via SharedToolRegistry + BatchRegistrar (robust — doesn't depend on
        # ensure_tool_registered, which lives on ToolRegistrar, not ToolRegistry).
        try:
            from victor.agent.shared_tool_registry import SharedToolRegistry

            _shared = SharedToolRegistry.get_instance()
            _registered = {t.name for t in self.orchestrator.tools.list_tools(only_enabled=False)}
            _missing = benchmark_tools - _registered
            if _missing:
                _new = _shared.get_tools_for_names(_missing)
                if _new:
                    from victor.tools.batch_registration import BatchRegistrar

                    BatchRegistrar(self.orchestrator.tools).register_batch(_new, fail_fast=False)
                    logger.info("Demand-registered curated tools: %s", sorted(_missing))
        except Exception as e:
            logger.debug("Demand-registration of curated tools skipped: %s", e)
        try:
            self.orchestrator.set_enabled_tools(benchmark_tools)
            logger.info(
                f"Benchmark tool restriction: {len(benchmark_tools)} tools enabled "
                f"({', '.join(sorted(benchmark_tools))})"
            )
        except Exception as e:
            logger.debug(f"Could not restrict tools for benchmark: {e}")

        # Source fix: sync the registry's per-tool _tool_enabled map with the
        # curated set. set_enabled_tools (tool_access_policy) sets the policy
        # filter + selector filter but does NOT update the registry's own
        # _tool_enabled map — so list_tools(only_enabled=True), used by
        # MULTIPLE tool-gathering paths (agentic loop ACT phase, semantic
        # selector, etc.), excludes curated-but-registered-disabled tools
        # (code/graph). Enabling them in the registry map makes ALL paths
        # see them, not just paths that check the policy filter. This is the
        # definitive fix for "code/graph not in the LLM schema" — consumer
        # fixes (#343/#345/#348 union) only cover select_tools; this covers
        # every path.
        try:
            for _name in benchmark_tools:
                self.orchestrator.tools.enable_tool(_name)
        except Exception:
            pass

        trace = AgenticExecutionTrace(
            task_id=task.task_id,
            start_time=time.time(),
            session_id=self._task_session_id,
        )

        # Inject task context into vertical context for framework enrichment
        # This is the proper architecture: hints flow through the enrichment pipeline
        self._inject_task_context(task, workspace_dir)

        # Analyze intent for completion detection (framework feature)
        task_description = task.issue_text or task.prompt
        self._completion_detector.analyze_intent(task_description)

        # Configure the orchestrator's detector to not prematurely stop.
        # The adapter's outer loop controls completion, not the orchestrator.
        orch_detector = getattr(self.orchestrator, "_task_completion_detector", None)
        # Resolve the decision service ONCE — inject into both the orchestrator's
        # detector AND the adapter's own completion detector. The adapter's
        # detector (_completion_detector) is what the per-turn loop actually
        # uses (should_stop); without the service it uses pure regex and never
        # logs the decision — so the FEP-0012 classifier gets 0 task_completion
        # samples. Mirrors the working stage_detection path (state_machine.py).
        edge_service = None
        try:
            from victor.agent.services.protocols.decision_service import (
                get_decision_service,
            )
            from victor.core import get_container

            edge_service = get_decision_service(get_container())
        except Exception:
            pass

        # Inject into the adapter's detector (the one the loop actually uses).
        if edge_service:
            self._completion_detector._decision_service = edge_service

        if orch_detector:
            if edge_service:
                orch_detector._decision_service = edge_service

            orch_detector.analyze_intent(task_description)

            # If regex couldn't classify (common for complex SWE-bench issues),
            # default to requiring file modifications before completion.
            if not orch_detector._state.expected_deliverables:
                from victor.agent.task_completion import DeliverableType

                orch_detector._state.expected_deliverables = [DeliverableType.FILE_MODIFIED]

            # High continuation budget — adapter controls stopping
            orch_detector._state.max_continuation_requests = 999

        # Determine task complexity with fallback chain:
        # 1. Explicit override from task (highest priority)
        # 2. Inference from task description
        # 3. Conservative default (MEDIUM)
        complexity_value = self._determine_complexity(task, task_description)
        self._completion_detector.configure_for_complexity(complexity_value)

        # Wrap task description with explicit instructions for benchmark tasks.
        # Raw issue text alone causes models to analyze instead of fix.
        # GEPA failure analysis (edit_mismatch category): the #1 failure mode is
        # old_str not matching file content exactly. Explicit guidance here reduces
        # edit rollbacks by ensuring the agent copies text verbatim from read output.
        graph_guidance = (
            "- To inspect CALL RELATIONSHIPS across modules: call `graph` with "
            "mode='callers'|'callees'|'impact' (e.g. graph(mode='callers', "
            "node='ClassName.method')). Use graph to inspect callers, callees, "
            "dependencies, and impact before editing cross-module code.\n"
            if graph_enabled
            else ""
        )
        graph_workflow = (
            "   - For cross-module bugs, use `graph` (callers/callees/impact) to "
            "trace how the target symbol is reached before editing.\n"
            if graph_enabled
            else ""
        )
        if is_deep_research:
            prompt = (
                "Produce a rigorous deep-research report that answers the request below.\n\n"
                "WORKSPACE: all user-provided attachments are under the workspace-local "
                "`attachments/` directory. Inspect every listed attachment. You may also "
                "use web research tools when available. Do not modify the attachments.\n\n"
                "ATTACHMENT TOOLS: `read` is text-only. For PDFs, use `pdftotext FILE -`; "
                "if a PDF is scanned, render a page with `pdftocairo` and pipe it to "
                "`tesseract stdin stdout`. For images, use `tesseract FILE stdout`. For "
                "video, use `ffprobe` plus `ffmpeg` to sample representative frames, piping "
                "frames to `tesseract stdin stdout`. Inspect content, not only file metadata, "
                "before moving on to web research.\n\n"
                "DELIVERABLE: write the complete final report to `report.md` with `write`. "
                "Support material claims with inline citations and include a source list. "
                "Clearly distinguish evidence, synthesis, and uncertainty. When the file is "
                "complete, state `The report is complete.`\n\n"
                f"REQUEST:\n{task_description}"
            )
        else:
            prompt = (
                "Fix the following issue by editing the source code in this repository.\n\n"
                "WORKSPACE: the repository source is at your current working directory "
                "(the workspace root). All source, tests, and files you need are UNDER "
                "it. NEVER search outside the workspace — do NOT look in host paths "
                "like `.venv`, `~/.victor`, `site-packages`, or `/Users/...`; those are "
                "blocked and won't help. Use the `code` tool (search/grep) to locate "
                "files within the workspace.\n\n"
                "IMPORTANT: You have separate tools — `read`, `edit`, `write`, `code`, "
                "and `shell`. Call each as its own tool with named parameters.\n\n"
                "BE EFFICIENT — you have a limited budget of tool calls and turns.\n"
                "Most fixes need only 2-3 reads/searches before you can edit. The most "
                "common failure mode is reading too many files and running out of time "
                "before editing. Act with conviction:\n"
                "- SEARCH once or twice (grep/code) to locate the right file.\n"
                "- READ only the relevant function/method (use offset/limit).\n"
                "- EDIT as soon as you understand the fix — do NOT read the whole file.\n"
                "- Make a best-effort fix rather than over-researching; you can iterate "
                "after seeing the test result.\n\n"
                "TOOL USAGE:\n"
                "- To READ a file: call `read(path='path/to/file')`\n"
                "- To EDIT a file: call `edit(ops=[{'type':'replace','path':'...','old_str':'...','new_str':'...'}])`\n"
                "- To WRITE a new file: call `write(path='...', content='...')`\n"
                "- To SEARCH code: call `code` with cmd='search <query>' or 'grep <pattern>'\n"
                "- To RUN commands (pip install, pytest, build): call `shell` with cmd='<command>'\n"
                + graph_guidance
                + "\nWORKFLOW:\n"
                "1. Use `code` (search or grep) to find relevant files in this repository\n"
                + graph_workflow
                + "2. Use `read` to examine the code (use offset/limit for large files)\n"
                "3. Use `edit` to apply your fix — COPY old_str EXACTLY from the read "
                "output, character-by-character. Do NOT type it from memory.\n"
                "4. If an edit fails, re-read the file and try again with the exact text.\n"
                "5. VERIFY: call `shell` with cmd='python -m pytest <test_file> -x'. "
                "Iterate until the test passes.\n"
                "6. When the test passes, state 'The fix is complete and verified.'\n\n"
                "CONSTRAINTS:\n"
                "- Stay within the workspace directory. NEVER run 'find /' or search "
                "outside the project.\n"
                "- The `edit` old_str must match file content EXACTLY (whitespace, "
                "quotes, line breaks). Always copy from the most recent read output.\n\n"
                f"ISSUE:\n{task_description}"
            )

        complete = False
        agentic_loop_iterations = 0
        last_agentic_loop_success: Optional[bool] = None
        agentic_loop_error_count = 0
        try:
            # Start heartbeat for silent hang detection
            import time as _time

            _heartbeat_start = _time.time()

            async def _heartbeat():
                while True:
                    await asyncio.sleep(30)
                    logger.info(
                        "[AgentAdapter] Heartbeat: turn=%d, tool_calls=%d, "
                        "edited=%s, exploration_calls=%d, elapsed=%.0fs",
                        self._turns,
                        len(self._tool_calls),
                        self._made_edit_tool_call,
                        self._exploration_calls,
                        _time.time() - _heartbeat_start,
                    )

            _heartbeat_task = asyncio.create_task(_heartbeat())

            # Execute agent loop
            while not complete and self._turns < self.config.max_turns:
                # Enforce total_timeout across the whole task (previously tracked
                # but NOT enforced): max_turns × per-turn loop iterations × slow
                # inference can run for hours on a task that never completes. This
                # is the outermost wall-clock bound — it stops with the partial
                # trace instead of running to the turn cap. (The per-turn
                # asyncio.wait_for below only bounds a single turn.)
                _task_elapsed = _time.time() - _heartbeat_start
                if _task_elapsed > self.config.total_timeout:
                    logger.warning(
                        "[AgentAdapter] total_timeout %ds exceeded at turn %d "
                        "(elapsed %.0fs) — stopping with partial result",
                        self.config.total_timeout,
                        self._turns,
                        _task_elapsed,
                    )
                    break
                self._turns += 1

                # Add user message. Turn 1 gets the full task prompt; later
                # turns get a continuation. If the agent has already edited a
                # file, let it continue freely; if it has NOT edited yet, the
                # message escalates with exploration depth (_exploration_calls
                # counts non-modifying calls since the last edit) to push the
                # agent toward editing before it runs out of turns. This is the
                # primary lever against the over-exploration failure mode.
                if self._turns == 1:
                    current_message = prompt
                elif self._made_edit_tool_call:
                    current_message = "Continue."
                elif is_deep_research:
                    n = self._exploration_calls
                    if n >= 8:
                        current_message = (
                            "You have enough evidence. Stop researching and write the complete "
                            "report to `report.md` now using `write`."
                        )
                    else:
                        current_message = (
                            "Continue the research, then write the complete final report to "
                            "`report.md` using `write`."
                        )
                else:
                    n = self._exploration_calls
                    if n >= 8:
                        current_message = (
                            "CRITICAL: You must edit NOW or this task fails. "
                            f"You have made {n} tool calls (reads/searches) "
                            "without editing anything. Your current "
                            "understanding of the bug is sufficient — do not "
                            "read more. Call `edit` right now with your best "
                            "fix, copying old_str from text you already read."
                        )
                    elif n >= 5:
                        current_message = (
                            "STOP reading. You have made "
                            f"{n} tool calls without editing. Apply your best "
                            "fix NOW using `edit(ops=[...])`. Use text you "
                            "already read as old_str — do not search or read "
                            "again first."
                        )
                    else:
                        current_message = (
                            "You have enough context to apply a fix. Use `edit` "
                            "now — pick the most likely location based on what "
                            "you have read so far and apply your best fix. You "
                            "can iterate after seeing the test result."
                        )
                self._messages.append({"role": "user", "content": current_message})

                # Get agent response
                logger.info(
                    "[AgentAdapter] Turn %d started (tool_calls=%d, "
                    "edited=%s, exploration_calls=%d)",
                    self._turns,
                    len(self._tool_calls),
                    self._made_edit_tool_call,
                    self._exploration_calls,
                )
                try:
                    response = await asyncio.wait_for(
                        self.orchestrator.chat(current_message),
                        timeout=self.config.timeout_per_turn,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[AgentAdapter] Turn {self._turns} timed out")
                    break
                except Exception as e:
                    logger.error("[AgentAdapter] Turn %d provider error: %s", self._turns, e)
                    # Provider error (disconnect, SSL, etc.) — don't retry,
                    # record what we have and move on
                    break

                assistant_content = response.content if response else ""
                response_metadata = dict(getattr(response, "metadata", None) or {})
                try:
                    agentic_loop_iterations += int(
                        response_metadata.get("agentic_loop_iterations", 0) or 0
                    )
                except (TypeError, ValueError):
                    pass
                if "agentic_loop_success" in response_metadata:
                    last_agentic_loop_success = bool(response_metadata["agentic_loop_success"])
                if response_metadata.get("agentic_loop_error"):
                    agentic_loop_error_count += 1

                # Capture and display reasoning/thinking if present
                reasoning = None
                if response and response.metadata:
                    reasoning = response.metadata.get("reasoning_content")
                if reasoning:
                    logger.info(
                        "[AgentAdapter] Model reasoning (%d chars):\n%s",
                        len(reasoning),
                        reasoning[:500] + ("..." if len(reasoning) > 500 else ""),
                    )

                # Log the actual response content for debugging
                logger.info(
                    "[AgentAdapter] Turn %d response (%d chars, %d tool calls):\n%s",
                    self._turns,
                    len(assistant_content),
                    len(response.tool_calls) if response and response.tool_calls else 0,
                    assistant_content[:300] + ("..." if len(assistant_content) > 300 else ""),
                )

                self._messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "reasoning": reasoning,
                    }
                )

                # Use framework's completion detection (not gaming code)
                self._completion_detector.analyze_response(assistant_content)

                # For benchmark tasks, ignore premature active signals (e.g. "summary:")
                # unless the agent has actually modified files. Without this guard,
                # models that include "Summary:" headers in their first response
                # trigger immediate termination before any work is done.
                if (
                    self._completion_detector._state.active_signal_detected
                    and not self._made_edit_tool_call
                ):
                    self._completion_detector._state.active_signal_detected = False
                    self._completion_detector._state.completion_signals.clear()

                # Log the task_completion decision DIRECTLY (not via
                # _decide_sync) with an explicit session_id override. The
                # contextvar may not propagate to the orchestrator's internal
                # decision-logging paths during orchestrator.chat(), so all
                # 1020 prior task_completion entries had session_id="" and
                # couldn't join to outcomes. The explicit override guarantees
                # the spine is stamped. This is the FEP-0012 classifier's key
                # signal — without it, validate says "task_completion did not
                # clear the bar" (0 samples).
                try:
                    from victor.agent.decisions.chain import log_decision

                    log_decision(
                        decision_type="task_completion",
                        context={
                            "response_tail": assistant_content[-500:],
                            "deliverable_count": len(
                                self._completion_detector._state.completed_deliverables
                            ),
                            "signal_count": len(
                                self._completion_detector._state.completion_signals
                            ),
                        },
                        result="incomplete" if not complete else "complete",
                        source="heuristic",
                        confidence=0.3,
                        session_id_override=self._task_session_id,
                    )
                except Exception:
                    pass  # never break the benchmark on logging

                complete = self._completion_detector.should_stop()

                # Test-pass → force-complete (Phase 2 item 7): if the agent has
                # edited files AND its response indicates tests passed, the fix
                # is verified — stop iterating. This breaks the "retry forever"
                # loop (62/62 retry in the analysis) by giving the completion
                # detector a concrete success signal it lacks on its own.
                if (
                    not complete
                    and self._made_edit_tool_call
                    and self._has_test_pass_signal(assistant_content)
                ):
                    logger.info(
                        "[AgentAdapter] Turn %d test-pass detected → force-complete",
                        self._turns,
                    )
                    complete = True

                logger.info(
                    "[AgentAdapter] Turn %d complete=%s (deliverables=%d, signals=%d, "
                    "active_signal=%s, edited=%s, test_pass=%s)",
                    self._turns,
                    complete,
                    len(self._completion_detector._state.completed_deliverables),
                    len(self._completion_detector._state.completion_signals),
                    self._completion_detector._state.active_signal_detected,
                    self._made_edit_tool_call,
                    complete and self._turns > 1,
                )

                # Check tool budget
                if len(self._tool_calls) >= self.config.tool_budget:
                    logger.warning(f"Tool budget exhausted: {len(self._tool_calls)}")
                    break

                # Update prompt for continuation
                prompt = "Continue."

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            trace.validation_errors["execution"] = str(e)
        finally:
            # Cancel heartbeat
            _heartbeat_task.cancel()
            try:
                await _heartbeat_task
            except asyncio.CancelledError:
                pass
            # Structured per-task outcome summary for failure diagnosis. Surfaces
            # whether the agent produced a patch (files_modified), which tools
            # failed and how often, and what got circuit-broken — the signals
            # previously missing when a task came back "failed".
            logger.warning(
                "[AgentAdapter] Task summary: turns=%d tool_calls=%d "
                "edited=%s files_modified=%d (%s) tool_failures=%s disabled_tools=%s",
                self._turns,
                len(self._tool_calls),
                self._made_edit_tool_call,
                len(self._file_edits),
                ", ".join(getattr(e, "path", "") for e in self._file_edits[:5]),
                dict(self._tool_failure_counts) or "none",
                sorted(self._disabled_tools) or "none",
            )

        # Populate trace
        trace.end_time = time.time()
        trace.turns = self._turns
        trace.messages = self._messages.copy()
        trace.tool_calls = self._tool_calls.copy()
        trace.file_edits = self._file_edits.copy()
        # Preserve the inner AgenticLoop's completion telemetry for end-to-end
        # completion-strategy A/Bs. ``turns`` is the adapter's outer continuation
        # count; this sum is the strategy-sensitive loop latency and
        # ``agentic_loop_success`` is the claim compared with verifier truth.
        trace.completion_signals.update(
            {
                "agentic_loop_iterations": agentic_loop_iterations,
                "agentic_loop_success": last_agentic_loop_success,
                "agentic_loop_error_count": agentic_loop_error_count,
                "outer_completion_claimed": complete,
                "outer_stop_reason": (
                    "complete"
                    if complete
                    else "max_turns" if self._turns >= self.config.max_turns else "stopped"
                ),
            }
        )
        logger.info(
            f"[AgentAdapter] Trace populated: {len(self._tool_calls)} tool calls, {self._turns} turns"
        )

        # Generate the patch from the GROUND TRUTH (git diff) — not the
        # adapter's fallible edit-capture. Stages all changes (including new
        # files) then diffs staged vs HEAD. Falls back to the edit-capture
        # (_generate_combined_patch) only if git is unavailable (non-git workspace).
        if self.config.working_dir:
            git_patch = await workspace_git_diff(Path(self.config.working_dir))
            trace.generated_patch = git_patch or self._generate_combined_patch()
        else:
            trace.generated_patch = self._generate_combined_patch()
        # Persist for get_conversation_trace() → execution manifest → judge views.
        self._generated_patch = trace.generated_patch or ""

        # A patch is the right artifact for SWE-bench, which applies it to a
        # fresh clone. Code-generation benchmarks are the opposite: MBPP and
        # HumanEval execute ``agent_output + test_code`` directly, so they need
        # Python source. Handing them a diff makes line 3 of solution.py read
        # "@@ -0,0 +1,27 @@" — a SyntaxError before a single test runs, which is
        # why MBPP scored 0 on all 135 real tasks it has ever been given.
        trace.generated_code = (
            self._capture_report_source() if is_deep_research else self._capture_solution_source()
        )

        # Populate correction metrics if tracking is enabled
        if self._metrics_collector:
            trace.correction_metrics = self._metrics_collector.metrics.to_dict()

        # Capture token usage from orchestrator (P1: Token Tracking Fix)
        if hasattr(self.orchestrator, "get_token_usage"):
            trace.token_usage = self.orchestrator.get_token_usage()
            # Fallback: read cumulative dict directly if metrics reports 0
            if trace.token_usage and trace.token_usage.total_tokens == 0:
                cum = getattr(self.orchestrator, "_cumulative_token_usage", {})
                if cum.get("total_tokens", 0) > 0:
                    from victor.evaluation.protocol import TokenUsage

                    trace.token_usage = TokenUsage(
                        input_tokens=cum.get("prompt_tokens", 0),
                        output_tokens=cum.get("completion_tokens", 0),
                        total_tokens=cum.get("total_tokens", 0),
                    )

        return trace

    def _capture_report_source(self) -> str:
        """Capture a DR3 report while its ephemeral workspace still exists."""
        if self.config.working_dir:
            report_path = Path(self.config.working_dir) / "report.md"
            if report_path.is_file():
                try:
                    report = report_path.read_text(errors="replace")
                    if report.strip():
                        return report
                except OSError:
                    pass

        for edit in reversed(self._file_edits):
            if Path(edit.path).suffix.lower() not in {".md", ".txt"}:
                continue
            if edit.after_content.strip():
                return edit.after_content

        for message in reversed(self._messages):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content") or "").strip()
            if content and not content.startswith("VICTOR_"):
                return content
        return ""

    def _capture_solution_source(self) -> str:
        """Concatenated source of the Python files the agent wrote.

        Built from the tracked edits rather than the workspace, so it survives
        the temp directory being cleaned up before the caller looks at it. Last
        write to a path wins — an agent that revises a file twice means the
        second version.

        Test files and conftest.py are excluded. The runner appends the
        benchmark's own tests to this string, so a stale copy of the agent's
        guess at them would shadow the real ones — and conftest.py is pytest
        plumbing that routinely contains ``from solution import f``. Concatenated
        into solution.py that line imports the module into itself, which cost 10
        of 48 tasks a circular-import failure in the first run that used this.

        Self-imports are stripped for the same reason even when they appear in a
        file that is otherwise solution source: valid in a separate module,
        fatal once everything becomes one file.
        """
        from pathlib import Path as _Path

        by_path: dict = {}
        for edit in self._file_edits:
            path = str(getattr(edit, "path", "") or "")
            if not path.endswith(".py"):
                continue
            name = _Path(path).name
            if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
                continue
            content = getattr(edit, "after_content", "") or ""
            if not content.strip() and self.config.working_dir:
                # after_content is best-effort; the workspace still exists here.
                try:
                    candidate = _Path(self.config.working_dir) / path
                    if candidate.is_file():
                        content = candidate.read_text(errors="replace")
                except Exception:
                    content = ""
            if content.strip():
                by_path[path] = self._strip_self_imports(content)

        return "\n\n".join(by_path[path] for path in sorted(by_path))

    @staticmethod
    def _strip_self_imports(source: str) -> str:
        """Drop ``import solution`` lines, which self-import once concatenated."""
        # Substitution rather than splitlines/join: the latter silently drops a
        # trailing newline, and this text gets concatenated with the benchmark's
        # tests.
        return re.sub(
            r"(?m)^[ \t]*(?:from[ \t]+solution[ \t]+import\b|import[ \t]+solution\b).*\n?",
            "",
            source,
        )

    def _determine_complexity(self, task: BenchmarkTask, task_description: str) -> str:
        """Determine task complexity using fallback chain.

        Priority order:
        1. Explicit override from task.complexity_override (caller knows best)
        2. Inference from task description using TaskComplexityService
        3. Conservative default: "medium"

        This design:
        - Keeps override optional (minimal config)
        - Allows max control when needed
        - Framework doesn't fight what caller wants

        Args:
            task: The benchmark task (may have complexity_override)
            task_description: Text to classify if no override

        Returns:
            Complexity level string: simple, medium, complex, generation, action, analysis
        """
        # Priority 1: Explicit override (caller knows best)
        if task.complexity_override:
            logger.info(f"Using explicit complexity override: {task.complexity_override}")
            return task.complexity_override

        # Priority 2: Inference from task description
        try:
            from victor.framework.task.complexity import TaskComplexityService

            service = TaskComplexityService()
            classification = service.classify(task_description)

            # Only use inference if confidence is reasonable
            if classification.confidence >= 0.5:
                logger.info(
                    f"Task classified as {classification.complexity.value} "
                    f"(budget: {classification.tool_budget}, confidence: {classification.confidence:.2f})"
                )
                return classification.complexity.value
            else:
                logger.info(
                    f"Low confidence classification ({classification.confidence:.2f}), "
                    f"using conservative default"
                )
        except Exception as e:
            logger.warning(f"Complexity inference failed: {e}")

        # Priority 3: Conservative default
        logger.info("Using conservative default complexity: medium")
        return "medium"

    def _inject_task_context(self, task: BenchmarkTask, workspace_dir: Path) -> None:
        """Inject task context into orchestrator's system prompt for enrichment.

        This replaces the gaming approach of _build_task_prompt() with proper
        framework integration. Context is appended to the system prompt using
        the orchestrator's native append_to_system_prompt() method.

        Args:
            task: The benchmark task
            workspace_dir: Working directory for the task
        """
        context_sections = []

        # PIN the task description at the TOP of system context
        # This ensures the model always knows what it's supposed to fix,
        # even as the conversation grows and the user message fades.
        task_desc = getattr(task, "issue_text", None) or task.prompt or task.description
        if task_desc:
            context_sections.append(
                f"## YOUR TASK (focus ONLY on this issue)\n"
                f"{task_desc[:2000]}\n\n"
                f"IMPORTANT: Fix ONLY the issue described above. "
                f"Do NOT fix other issues you notice in the code."
            )

        # Add working directory context
        context_sections.append(f"## Working Directory\nYou are working in: {workspace_dir}")

        # Add repository context if available
        if task.repo:
            repo_name = task.repo.replace("https://github.com/", "").replace(".git", "")
            context_sections.append(f"**Repository:** {repo_name}")

        # Add hints through the system prompt (framework integration)
        # These become part of the system context, benefiting all verticals
        if task.hints:
            hints_section = "## Hints\n" + "\n".join(f"- {hint}" for hint in task.hints)
            context_sections.append(hints_section)

        # Add context code if provided
        if task.context_code:
            code_section = f"## Context Code\n```python\n{task.context_code}\n```"
            context_sections.append(code_section)

        # Append all context to the orchestrator's system prompt
        if context_sections:
            combined_context = "\n\n".join(context_sections)
            self.orchestrator.append_to_system_prompt(combined_context)
            logger.debug(f"Injected task context: {len(context_sections)} sections")

    def _generate_combined_patch(self) -> str:
        """Generate combined unified diff from all file edits."""
        if not self._file_edits:
            return ""

        patches = []
        for edit in self._file_edits:
            if edit.diff:
                patches.append(edit.diff)

        return "\n".join(patches)

    @classmethod
    def from_profile(
        cls,
        profile: str = "default",
        base_url: Optional[str] = None,
        model_override: Optional[str] = None,
        timeout: int = 120,
        config: Optional[AdapterConfig] = None,
    ) -> "VictorAgentAdapter":
        """Create adapter from a Victor profile.

        Args:
            profile: Profile name from profiles.yaml
            base_url: Override base URL (e.g., for specific Ollama host)
            model_override: Override model from profile
            timeout: Request timeout
            config: Adapter configuration

        Returns:
            VictorAgentAdapter instance
        """
        settings = load_settings()
        profiles = settings.load_profiles()

        if profile not in profiles:
            raise ValueError(f"Profile '{profile}' not found. Available: {list(profiles.keys())}")

        profile_config = profiles[profile]

        # Get provider settings (ProfileConfig is a Pydantic model, use attribute access)
        provider_name = profile_config.provider
        model = model_override or profile_config.model

        # Handle base URL override
        if base_url:
            if provider_name == "ollama":
                os.environ["OLLAMA_HOST"] = base_url
            elif provider_name == "lmstudio":
                os.environ["LMSTUDIO_ENDPOINTS"] = base_url

        # Create provider using ProviderRegistry (class methods)
        # Note: api_key and base_url may be extra fields from profiles.yaml (ProfileConfig allows extra="allow")
        # Only pass base_url if explicitly set, otherwise let provider use its default
        provider_kwargs = {
            "settings": settings,
            "timeout": timeout,
        }

        # Get API key from profile, environment variable, or keyring
        api_key = getattr(profile_config, "api_key", None)
        # Resolve ${ENV_VAR} references
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var)

        # If still no API key, try keyring
        if not api_key:
            try:
                from victor.config.api_keys import get_api_key

                api_key = get_api_key(provider_name)
            except ImportError:
                pass

        if api_key:
            provider_kwargs["api_key"] = api_key

        # Profiles.yaml commonly uses `endpoint:` (ProfileConfig keeps it as an
        # extra field); honor it as the base-URL synonym — without this the
        # provider silently falls back to its localhost default (observed live:
        # a 96-task calibration run produced 96 empty-response failures because
        # the agent called localhost instead of the profile's Ollama host).
        effective_base_url = (
            base_url
            or getattr(profile_config, "base_url", None)
            or getattr(profile_config, "endpoint", None)
        )
        if effective_base_url:
            provider_kwargs["base_url"] = effective_base_url

        provider = ProviderRegistry.create(provider_name, **provider_kwargs)

        # Create orchestrator - lazy import to break circular dependency
        # Chain: orchestrator → code_correction_middleware → evaluation → agent_adapter → orchestrator
        from victor.agent.orchestrator import AgentOrchestrator as Orchestrator

        orchestrator = Orchestrator(
            settings=settings,
            provider=provider,
            model=model,
            temperature=profile_config.temperature,
            max_tokens=profile_config.max_tokens,
            provider_name=provider_name,
        )

        return cls(orchestrator, config)

    @classmethod
    async def create_from_session_config(
        cls,
        session_config: "SessionConfig",
        *,
        profile: Optional[str] = None,
        vertical: Optional[str] = None,
        config: Optional[AdapterConfig] = None,
        enable_observability: bool = True,
    ) -> "VictorAgentAdapter":
        """Create an adapter through the public framework Agent API.

        This is the preferred path for CLI/runtime callers that already have a
        normalized SessionConfig and want framework-owned provider/session setup.

        Args:
            vertical: Vertical name (e.g. "coding") so the vertical's
                capabilities — code_search/graph for coding — register via
                AgentFactory. Required for code-intelligence benchmarks.
        """
        from victor.framework.agent import Agent

        agent = await Agent.create(
            profile=profile or session_config.agent_profile,
            session_config=session_config,
            vertical=vertical,
            enable_observability=enable_observability,
        )
        return cls(agent.get_orchestrator(), config)


def create_victor_agent_callback(
    adapter: VictorAgentAdapter,
) -> Callable[[BenchmarkTask, Path], AgenticExecutionTrace]:
    """Create an agent callback for AgenticBenchmarkRunner.

    Args:
        adapter: VictorAgentAdapter instance

    Returns:
        Async callback function for benchmark runner
    """

    async def callback(task: BenchmarkTask, workspace_dir: Path) -> AgenticExecutionTrace:
        return await adapter.execute_task(task, workspace_dir)

    return callback


# Convenience function for quick setup
async def run_agentic_task(
    task: BenchmarkTask,
    workspace_dir: Path,
    profile: str = "default",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> AgenticExecutionTrace:
    """Run a single agentic task with Victor.

    Args:
        task: Benchmark task to execute
        workspace_dir: Working directory for task
        profile: Victor profile name
        base_url: Override base URL
        model: Override model

    Returns:
        AgenticExecutionTrace with results
    """
    adapter = VictorAgentAdapter.from_profile(
        profile=profile,
        base_url=base_url,
        model_override=model,
    )
    return await adapter.execute_task(task, workspace_dir)
