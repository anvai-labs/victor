# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility checks for the deprecated intelligent-pipeline shim."""

from victor.agent.intelligent_pipeline import (
    IntelligentAgentPipeline,
    clear_intelligent_agent_cache,
    get_intelligent_agent,
)
from victor.agent.runtime_intelligence_pipeline import (
    RuntimeIntelligencePipeline,
    clear_runtime_intelligence_pipeline_cache,
    get_runtime_intelligence_pipeline,
)


def test_legacy_shim_reexports_canonical_runtime_intelligence_symbols() -> None:
    """Legacy module should remain a thin alias to the canonical implementation."""
    assert IntelligentAgentPipeline is RuntimeIntelligencePipeline
    assert get_intelligent_agent is get_runtime_intelligence_pipeline
    assert clear_intelligent_agent_cache is clear_runtime_intelligence_pipeline_cache
