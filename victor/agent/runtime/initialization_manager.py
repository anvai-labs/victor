"""Initialization phase manager for orchestrator runtime boundaries.

Coordinates the 9 runtime initialization phases in the correct order,
providing structured results and error reporting for each phase.
Supports fail-fast for critical phases and dependency-based skipping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Set

logger = logging.getLogger(__name__)


class InitializationError(Exception):
    """Raised when a critical initialization phase fails."""

    def __init__(self, phase: str, error: str, completed_phases: List[str]):
        self.phase = phase
        self.error = error
        self.completed_phases = completed_phases
        super().__init__(f"Critical initialization phase '{phase}' failed: {error}")


@dataclass
class PhaseResult:
    """Result of a single initialization phase."""

    name: str
    success: bool = True
    duration_ms: float = 0.0
    components_created: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class InitializationResult:
    """Aggregate result of all initialization phases."""

    phases: List[PhaseResult] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return all(p.success for p in self.phases)

    @property
    def total_duration_ms(self) -> float:
        return sum(p.duration_ms for p in self.phases)

    @property
    def failed_phases(self) -> List[PhaseResult]:
        return [p for p in self.phases if not p.success]

    @property
    def skipped_phases(self) -> List[PhaseResult]:
        return [p for p in self.phases if p.skipped]


class PhaseGroup(Enum):
    """Init-phase groups aligned to orchestrator construction boundaries (FEP-0016).

    The 9 phases are structurally interleaved with component assembly, so they run
    as three grouped calls at their natural lifecycle points rather than a single
    ``run_all_phases``:

    - ``EARLY``    — phases 1-4, before component assembly (in ``orchestrator.__init__``).
    - ``ASSEMBLY`` — phases 5-6, after ``_context_compactor`` exists (in
      ``ComponentAssembler.assemble_intelligence``).
    - ``SERVICE``  — phases 7-9, after ``_checkpoint_manager`` and assembled
      components exist (in ``AgentRuntimeBootstrapper.prepare_components``).
    """

    EARLY = "early"
    ASSEMBLY = "assembly"
    SERVICE = "service"


@dataclass
class _PhaseSpec:
    """One initialization phase: its runner, expected components, and constraints."""

    name: str
    initializer: Callable[[], None]
    components: List[str]
    critical: bool
    dependencies: List[str]
    group: PhaseGroup


class InitializationPhaseManager:
    """Manages orchestrator runtime initialization phases.

    Wraps the 9 ``_initialize_*`` methods on the orchestrator to provide structured
    timing, error reporting, component tracking, fail-fast for critical phases, and
    dependency-based skipping.

    The phases are interleaved with component assembly (FEP-0016), so the orchestrator
    invokes :meth:`run_group` at three construction boundaries. Cross-group
    dependencies (e.g. ``resilience_runtime`` depends on ``provider_runtime``) are
    satisfied because the manager accumulates succeeded-phase state across the calls.

    Usage:
        manager = InitializationPhaseManager()
        manager.run_group(orchestrator, PhaseGroup.EARLY)
        # ... component assembly builds _context_compactor ...
        manager.run_group(orchestrator, PhaseGroup.ASSEMBLY)
        # ... prepare_components builds _checkpoint_manager ...
        manager.run_group(orchestrator, PhaseGroup.SERVICE)
    """

    def __init__(self) -> None:
        # Accumulated across run_group() calls so later groups see earlier successes.
        self._result = InitializationResult()
        self._succeeded_phases: Set[str] = set()

    def _phase_specs(self, orchestrator: Any) -> List[_PhaseSpec]:
        """The 9 initialization phases in dependency order, tagged with their
        construction-boundary group (FEP-0016). Phase bodies are unchanged."""
        return [
            _PhaseSpec(
                "provider_runtime",
                orchestrator._initialize_provider_runtime,
                ["provider_runtime"],
                True,  # critical
                [],
                PhaseGroup.EARLY,
            ),
            _PhaseSpec(
                "metrics_runtime",
                orchestrator._initialize_metrics_runtime,
                ["usage_logger", "streaming_metrics_collector", "metrics_coordinator"],
                False,
                [],
                PhaseGroup.EARLY,
            ),
            _PhaseSpec(
                "workflow_runtime",
                orchestrator._initialize_workflow_runtime,
                ["workflow_registry"],
                False,
                [],
                PhaseGroup.EARLY,
            ),
            _PhaseSpec(
                "memory_runtime",
                orchestrator._initialize_memory_runtime,
                ["memory_manager"],
                False,
                [],
                PhaseGroup.EARLY,
            ),
            _PhaseSpec(
                "resilience_runtime",
                lambda: orchestrator._initialize_resilience_runtime(
                    context_compactor=orchestrator._context_compactor,
                ),
                ["recovery_handler", "recovery_integration"],
                False,
                ["provider_runtime"],
                PhaseGroup.ASSEMBLY,
            ),
            _PhaseSpec(
                "coordination_runtime",
                orchestrator._initialize_coordination_runtime,
                ["recovery_coordinator", "chunk_generator", "tool_planner", "task_coordinator"],
                False,
                ["provider_runtime"],
                PhaseGroup.ASSEMBLY,
            ),
            _PhaseSpec(
                "interaction_runtime",
                orchestrator._initialize_interaction_runtime,
                ["chat_service", "session_service"],
                True,  # critical
                ["provider_runtime", "coordination_runtime"],
                PhaseGroup.SERVICE,
            ),
            _PhaseSpec(
                "services",
                orchestrator._initialize_services,
                ["chat_service", "tool_service", "session_service", "context_service"],
                False,
                ["interaction_runtime"],
                PhaseGroup.SERVICE,
            ),
            _PhaseSpec(
                "credit_runtime",
                orchestrator._initialize_credit_runtime,
                ["credit_tracking_service"],
                False,
                ["interaction_runtime"],
                PhaseGroup.SERVICE,
            ),
        ]

    def run_group(self, orchestrator: Any, group: PhaseGroup) -> InitializationResult:
        """Run the phases in ``group`` (FEP-0016) in dependency order.

        Accumulates into the manager's result / succeeded-phase state so a phase in a
        later group can depend on a phase from an earlier group's call. Returns the
        aggregate :class:`InitializationResult` so far.

        Raises:
            InitializationError: If a critical phase in this group fails or is skipped.
        """
        specs = [s for s in self._phase_specs(orchestrator) if s.group is group]
        self._run_specs(specs)
        return self._result

    def run_phase(self, orchestrator: Any, name: str) -> InitializationResult:
        """Run a single named init phase in place (FEP-0016 per-phase wiring).

        The 9 phases are finely interleaved with orchestrator construction, so rather
        than move them, the orchestrator calls ``run_phase`` at each phase's existing
        site. The manager accumulates result / succeeded-phase state across the calls
        (same instance), so a phase's declared dependencies resolve against the phases
        that already ran earlier in construction, and it gains the phase contract's
        criticality (fail-fast), dependency-skip, and per-phase timing.

        Args:
            orchestrator: The orchestrator being constructed.
            name: The phase name (must match a spec in :meth:`_phase_specs`).

        Raises:
            KeyError: If ``name`` is not a declared phase.
            InitializationError: If this phase is critical and fails or is skipped.
        """
        spec = next((s for s in self._phase_specs(orchestrator) if s.name == name), None)
        if spec is None:
            raise KeyError(f"unknown initialization phase: {name!r}")
        self._run_specs([spec])
        return self._result

    def run_all_phases(self, orchestrator: Any) -> InitializationResult:
        """Run all 9 phases in order in a single call, resetting state first.

        NOTE: in production the phases are interleaved with component assembly, so the
        orchestrator invokes :meth:`run_group` at three boundaries instead. This method
        remains for direct/unit use and assumes every phase's prerequisites already
        exist on ``orchestrator``.

        Raises:
            InitializationError: If a critical phase fails.
        """
        self._result = InitializationResult()
        self._succeeded_phases = set()
        self._run_specs(self._phase_specs(orchestrator))

        if self._result.all_succeeded:
            logger.debug(
                "All %d initialization phases completed in %.1fms",
                len(self._result.phases),
                self._result.total_duration_ms,
            )
        else:
            failed = ", ".join(p.name for p in self._result.failed_phases)
            logger.warning("Initialization failures in phases: %s", failed)

        return self._result

    def _run_specs(self, specs: List[_PhaseSpec]) -> None:
        """Run phase specs with dependency-skip, fail-fast, and timing, appending to
        ``self._result`` and tracking ``self._succeeded_phases``."""
        for spec in specs:
            missing_deps = [d for d in spec.dependencies if d not in self._succeeded_phases]
            if missing_deps:
                reason = f"skipped due to failed/skipped dependencies: {', '.join(missing_deps)}"
                self._result.phases.append(
                    PhaseResult(
                        name=spec.name,
                        success=False,
                        skipped=True,
                        skip_reason=reason,
                        error=reason,
                    )
                )
                logger.warning("Phase '%s' %s", spec.name, reason)
                if spec.critical:
                    raise InitializationError(
                        phase=spec.name,
                        error=reason,
                        completed_phases=sorted(self._succeeded_phases),
                    )
                continue

            phase_result = self._run_phase(spec.name, spec.initializer, spec.components)
            self._result.phases.append(phase_result)

            if phase_result.success:
                self._succeeded_phases.add(spec.name)
            elif spec.critical:
                raise InitializationError(
                    phase=spec.name,
                    error=phase_result.error or "unknown error",
                    completed_phases=sorted(self._succeeded_phases),
                )
            else:
                logger.warning(
                    "Non-critical phase '%s' failed: %s — "
                    "system will continue with degraded %s functionality",
                    spec.name,
                    phase_result.error,
                    spec.name.replace("_runtime", "").replace("_", " "),
                )

    def _run_phase(
        self,
        name: str,
        initializer: Callable[[], None],
        expected_components: List[str],
    ) -> PhaseResult:
        """Run a single initialization phase with timing and error handling."""
        start = time.monotonic()
        phase = PhaseResult(name=name)

        try:
            initializer()
            phase.components_created = expected_components
        except Exception as exc:
            phase.success = False
            phase.error = str(exc)
            logger.error("Initialization phase '%s' failed: %s", name, exc)

        phase.duration_ms = (time.monotonic() - start) * 1000
        return phase
