# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Compatibility shim for the renamed runtime-intelligence pipeline module."""

from victor.agent.runtime_intelligence_pipeline import (
    PROVIDERS_WITH_REPETITION_ISSUES,
    PipelineStats,
    RequestContext,
    ResponseResult,
    RuntimeIntelligencePipeline,
    clear_runtime_intelligence_pipeline_cache,
    get_runtime_intelligence_pipeline,
)

# Backward-compatible exports for older imports.
IntelligentAgentPipeline = RuntimeIntelligencePipeline
get_intelligent_agent = get_runtime_intelligence_pipeline
clear_intelligent_agent_cache = clear_runtime_intelligence_pipeline_cache

__all__ = [
    "RuntimeIntelligencePipeline",
    "RequestContext",
    "ResponseResult",
    "PipelineStats",
    "get_runtime_intelligence_pipeline",
    "clear_runtime_intelligence_pipeline_cache",
    "PROVIDERS_WITH_REPETITION_ISSUES",
    "IntelligentAgentPipeline",
    "get_intelligent_agent",
    "clear_intelligent_agent_cache",
]
