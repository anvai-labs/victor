# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Decision chain resolver — unified fallback chain for all decision points.

All decision sites in the agentic loop should use `get_decision_chain()`
to determine whether to use heuristic-first or LLM-first, ensuring
consistent behavior across stage detection, tool selection, task
completion, etc.

Usage:
    from victor.agent.decisions.chain import get_decision_chain, should_use_llm

    # Get the ordered chain for a decision type
    chain = get_decision_chain("stage_detection")  # ["heuristic", "llm"]

    # Quick check: should LLM be used for this decision type?
    if should_use_llm("tool_selection"):
        tools = edge_model_filter(tools)
"""

from victor.core.json_utils import json_dumps
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Cache to avoid repeated settings loads
_chain_cache: dict = {}


def get_decision_chain(decision_type: str) -> List[str]:
    """Get the fallback chain for a decision type.

    Reads from settings.pipeline.decision_chain (per-type override)
    or settings.pipeline.decision_chain_default (global default).

    Args:
        decision_type: DecisionType enum value (e.g., "stage_detection",
                       "tool_selection", "task_completion")

    Returns:
        Ordered list of strategies: ["heuristic", "llm"] or ["llm", "heuristic"]
    """
    if decision_type in _chain_cache:
        return _chain_cache[decision_type]

    try:
        from victor.config.settings import load_settings

        settings = load_settings()
        pipeline = getattr(settings, "pipeline", None)
        if pipeline:
            # Check per-type override first
            chain_overrides = getattr(pipeline, "decision_chain", {})
            if decision_type in chain_overrides:
                chain = chain_overrides[decision_type]
                _chain_cache[decision_type] = chain
                return chain

            # Fall back to global default
            chain = getattr(pipeline, "decision_chain_default", ["heuristic", "llm"])
            _chain_cache[decision_type] = chain
            return chain
    except Exception:
        pass

    default = ["heuristic", "llm"]
    _chain_cache[decision_type] = default
    return default


def should_use_llm(decision_type: str) -> bool:
    """Check if LLM is in the fallback chain for this decision type.

    Returns True if "llm" appears anywhere in the chain.
    Use this for simple gate checks (e.g., should we call edge model at all?).
    """
    return "llm" in get_decision_chain(decision_type)


def is_llm_primary(decision_type: str) -> bool:
    """Check if LLM is the PRIMARY strategy (first in chain).

    Returns True only when "llm" is the first strategy.
    Use this when you need to know if LLM should be tried BEFORE heuristics.
    """
    chain = get_decision_chain(decision_type)
    return len(chain) > 0 and chain[0] == "llm"


def invalidate_chain_cache() -> None:
    """Clear the chain cache (for testing or settings reload)."""
    _chain_cache.clear()


def log_decision(
    decision_type: str,
    context: dict,
    result: str,
    source: str,
    confidence: float = 0.0,
    *,
    model_version: Optional[str] = None,
    feature_spec_version: Optional[str] = None,
    feature_digest: Optional[str] = None,
    session_id_override: Optional[str] = None,
) -> None:
    """Append decision I/O to JSONL for fine-tuning / RL data collection.

    Logs every decision (heuristic, LLM, or local classifier) with input
    context and output, enabling future model training on real decision patterns.

    FEP-0012: stamps the correlation spine (``session_id``/``turn_id``/
    ``trace_id``) and a unique ``decision_id`` so each record can be joined to
    its eventual outcome (``rl_outcome`` / ``usage.jsonl``) for reward-weighted
    training. Optional provenance fields (``model_version``,
    ``feature_spec_version``, ``feature_digest``) are set by the local
    classifier service for reproducibility.

    ``session_id_override``: explicit session_id that takes precedence over
    the contextvar. Callers that know their session_id (e.g. the benchmark
    adapter, which sets it before orchestrator.chat() but the contextvar
    may not propagate to all internal decision-logging paths) should pass it
    to guarantee the spine is stamped correctly.

    Path: ~/.victor/logs/decisions.jsonl
    """
    import json
    import uuid
    from datetime import datetime
    from pathlib import Path

    try:
        from victor.core.context import get_session_id, get_trace_id, get_turn_id

        log_dir = Path.home() / ".victor" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "decisions.jsonl"

        # Use explicit override if provided; otherwise fall back to contextvar.
        stamped_session_id = (
            session_id_override if session_id_override is not None else get_session_id()
        )

        entry = {
            "ts": datetime.now().isoformat(),
            "decision_id": uuid.uuid4().hex,
            "type": decision_type,
            "input": context,
            "output": result,
            "source": source,
            "confidence": confidence,
            # Correlation spine (FEP-0012) — join key to outcomes.
            "session_id": stamped_session_id,
            "turn_id": get_turn_id(),
            "trace_id": get_trace_id(),
        }
        # Optional provenance, set by the local classifier service.
        if model_version is not None:
            entry["model_version"] = model_version
        if feature_spec_version is not None:
            entry["feature_spec_version"] = feature_spec_version
        if feature_digest is not None:
            entry["feature_digest"] = feature_digest

        with open(log_path, "a") as f:
            f.write(json_dumps(entry, default=str) + "\n")
    except Exception:
        pass  # Never fail on logging
