# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Resolve one executor choice into a concrete admission/dispatch snapshot.

AUTO preserves the legacy USE_STATEGRAPH_AGENTIC_LOOP flag. STANDARD and
STATEGRAPH are explicit selections and never consult process-wide flags.
"""

from __future__ import annotations

from victor.config.agentic_loop_settings import AgenticLoopExecutor as AgenticLoopExecutor


def resolve_agentic_loop_executor(
    executor: AgenticLoopExecutor | str = AgenticLoopExecutor.AUTO,
) -> AgenticLoopExecutor:
    """Return a concrete executor, consulting the legacy flag only for AUTO.

    Resolver errors propagate; silently switching execution modes would invalidate
    admission checks. Pass the returned selection through the run's owned config.
    """
    selected = AgenticLoopExecutor.parse(executor)
    if selected is not AgenticLoopExecutor.AUTO:
        return selected
    from victor.core.feature_flags import FeatureFlag, is_feature_enabled

    return (
        AgenticLoopExecutor.STATEGRAPH
        if is_feature_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)
        else AgenticLoopExecutor.STANDARD
    )


def use_stategraph_executor(
    executor: AgenticLoopExecutor | str = AgenticLoopExecutor.AUTO,
) -> bool:
    """Compatibility predicate over the same explicit executor resolver."""
    return resolve_agentic_loop_executor(executor) is AgenticLoopExecutor.STATEGRAPH
