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

"""One structured caller run context (seam-duplication audit Cluster 2).

A consumer campaign needed four separate hand-patches (#1721 criteria
forwarding, #1723 budget binding, #1725 guided termination, #1728 budget
bind) because the caller's funding and completion declarations traveled as
hand-built config dicts assembled independently at every loop build site —
the buffered and streaming sites each re-derived
``executor``/``completion_strategy``/``enable_criteria_gate`` from loose
``run_context`` keys, and completion-critical tools were resolved twice
(pipeline config and the recovery seam).

``CallerRunContext`` is constructed ONCE at the SDK boundary
(``Agent.run``/``Agent.stream``) from ``(runtime_context_overrides,
context, settings)`` and threaded as one object:

- ``to_loop_config(**defaults)`` produces the ``AgenticLoopConfig``-compatible
  config dict (criteria-gate resolution, completion-tool names, declared
  executor/strategy). Site-resolved ``defaults`` (governance-checked
  executor, calibrated strategy, effect gate, per-turn auditor) win; declared
  fields only fill keys the site did not resolve. With nothing declared and
  the site's defaults passed in, the produced dict equals the hand-built
  literals byte-for-byte (dict equality) — the consolidation is
  parity-gated.
- ``to_runtime_overrides()`` produces the ``runtime_context_overrides`` dict
  (``tool_budget``/``iteration_budget`` keys) that feeds the run-scoped
  budget ledger funding seam (``runtime_overrides.extend_budget_funding``);
  merge derived plans through ``victor.core.runtime_context
  .merge_runtime_context_overrides`` (caller-bound keys win).
- ``extra`` holds unparsed context keys verbatim; loop state seeding still
  uses the caller's raw context dict, so arbitrary caller payloads (files,
  variables) reach the loop untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Context keys modeled as typed fields; every other context key passes
# through uninterpreted in ``extra``.
DECLARED_CONTEXT_KEYS = (
    "criteria",
    "complete_on_key",
    "completion_tool_names",
    "executor",
    "completion_strategy",
)

# Budget keys carried in runtime_context_overrides — the ledger funding seam
# (victor.core.runtime_context.CALLER_BOUND_OVERRIDE_KEYS).
BUDGET_OVERRIDE_KEYS = ("tool_budget", "iteration_budget")


def _coerce_optional_int(value: Any) -> Optional[int]:
    """Best-effort integer coercion; None passes through, junk becomes None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("CallerRunContext ignoring non-integer budget value: %r", value)
        return None


def _coerce_str_tuple(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(str(name) for name in value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(name) for name in value)
    return ()


@dataclass
class CallerRunContext:
    """The caller's funding and completion declarations for ONE run.

    All fields default empty/None: an empty context must produce the same
    config dicts and decisions as the pre-consolidation no-override path
    (byte-identical default, ADR-012).
    """

    tool_budget: Optional[int] = None
    iteration_budget: Optional[int] = None
    criteria: Optional[Dict[str, Any]] = None
    complete_on_key: Optional[str] = None
    completion_tool_names: Tuple[str, ...] = ()
    # AgenticLoopExecutor name ("standard"/"stategraph"/"auto") or enum.
    executor: Optional[Any] = None
    completion_strategy: Optional[str] = None
    # Unparsed context passthrough; loop state seeding uses the caller's raw
    # context dict, so this is carried for structured consumers, not spilled
    # into the loop config.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_loop_config(self, **defaults: Any) -> Dict[str, Any]:
        """Build the ``AgenticLoopConfig``-compatible config dict.

        Args:
            **defaults: The build site's settings-derived resolutions
                (``executor``, ``completion_strategy``, ``enable_effect_gate``,
                ``enable_per_turn_auditor``, ...). They win over declared
                fields so a caller declaration can never bypass the site's
                governance/calibration checks; declared fields fill keys the
                site did not resolve.

        Returns:
            Dict with ``enable_criteria_gate`` resolved from declared criteria
            (non-empty declaration = the caller's opt-in, FEP-0043),
            ``completion_tool_names`` only when declared, and every default.
        """
        config: Dict[str, Any] = dict(defaults)
        if self.executor is not None:
            config.setdefault("executor", self.executor)
        if self.completion_strategy is not None:
            config.setdefault("completion_strategy", self.completion_strategy)
        config["enable_criteria_gate"] = bool(self.criteria)
        if self.completion_tool_names:
            config.setdefault("completion_tool_names", tuple(self.completion_tool_names))
        return config

    def to_runtime_overrides(self) -> Dict[str, Any]:
        """Build the scoped ``runtime_context_overrides`` budget-key dict.

        Only the caller-bound budget keys (the ledger funding seam) — other
        override keys stay in the caller's original dict and merge on top via
        ``victor.core.runtime_context.merge_runtime_context_overrides``.
        """
        overrides: Dict[str, Any] = {}
        if self.tool_budget is not None:
            overrides["tool_budget"] = self.tool_budget
        if self.iteration_budget is not None:
            overrides["iteration_budget"] = self.iteration_budget
        return overrides

    def to_run_context(self) -> Dict[str, Any]:
        """Reconstruct the structured run-context dict for loop state seeding.

        Declared fields round-trip (an explicitly empty criteria declaration
        stays present so ``bool(criteria)`` parity holds); unparsed keys come
        back from ``extra`` verbatim. Empty when nothing was declared.
        """
        context: Dict[str, Any] = {}
        if self.criteria is not None:
            context["criteria"] = self.criteria
        if self.complete_on_key is not None:
            context["complete_on_key"] = self.complete_on_key
        if self.completion_tool_names:
            context["completion_tool_names"] = tuple(self.completion_tool_names)
        if self.executor is not None:
            context["executor"] = self.executor
        if self.completion_strategy is not None:
            context["completion_strategy"] = self.completion_strategy
        context.update(self.extra)
        return context

    @classmethod
    def from_overrides_and_context(
        cls,
        overrides: Optional[Mapping[str, Any]],
        context: Optional[Mapping[str, Any]],
        settings: Optional[Any] = None,
    ) -> "CallerRunContext":
        """Construct from the SDK seam's raw kwargs.

        Args:
            overrides: ``runtime_context_overrides`` — budget keys are
                extracted (int-coerced); other keys are the derived/plan
                surface and stay there.
            context: The structured run context (declared criteria,
                completion marker, completion-critical tools) — declared keys
                populate fields, everything else lands in ``extra`` verbatim.
            settings: Session settings; resolves the completion-critical tool
                names when the caller did not declare them here (one
                resolution shared by both consumption sites instead of two).
        """
        overrides_map = overrides if isinstance(overrides, Mapping) else {}
        context_map = context if isinstance(context, Mapping) else {}

        completion_tool_names: Tuple[str, ...] = ()
        declared_names = context_map.get("completion_tool_names")
        if declared_names is not None:
            completion_tool_names = _coerce_str_tuple(declared_names)
        else:
            completion_tool_names = _resolve_settings_completion_tools(settings)

        extra = {
            key: value for key, value in context_map.items() if key not in DECLARED_CONTEXT_KEYS
        }
        criteria = context_map.get("criteria")
        complete_on_key = context_map.get("complete_on_key")
        return cls(
            tool_budget=_coerce_optional_int(overrides_map.get("tool_budget")),
            iteration_budget=_coerce_optional_int(overrides_map.get("iteration_budget")),
            criteria=dict(criteria) if isinstance(criteria, Mapping) else criteria,
            complete_on_key=complete_on_key if isinstance(complete_on_key, str) else None,
            completion_tool_names=completion_tool_names,
            executor=context_map.get("executor"),
            completion_strategy=context_map.get("completion_strategy"),
            extra=extra,
        )


def _resolve_settings_completion_tools(settings: Optional[Any]) -> Tuple[str, ...]:
    """Resolve the settings-declared completion-critical tools, if any."""
    if settings is None:
        return ()
    declared = getattr(getattr(settings, "tools", None), "completion_tools", None)
    if not declared:
        return ()
    return _coerce_str_tuple(declared)


__all__ = [
    "BUDGET_OVERRIDE_KEYS",
    "CallerRunContext",
    "DECLARED_CONTEXT_KEYS",
]
