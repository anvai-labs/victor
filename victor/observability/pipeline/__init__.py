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

"""Pipeline Analytics Module - CI/CD and coverage analysis for Victor.

This module provides comprehensive analytics for CI/CD pipelines and
code coverage tracking. It supports multiple platforms and formats:

CI/CD Platforms:
- GitHub Actions
- GitLab CI
- Jenkins (planned)
- CircleCI (planned)
- Azure DevOps (planned)
- Bitbucket Pipelines (planned)

Coverage Formats:
- Cobertura XML
- LCOV
- JaCoCo XML

Usage:
    from victor.observability.pipeline import PipelineManager

    manager = PipelineManager("/path/to/project")
    result = await manager.analyze_pipelines()

    print(f"Found {len(result.configs)} pipeline configs")
    print(f"Issues: {len(result.issues)}")
    for rec in result.recommendations:
        print(f"  - {rec}")
"""

from .analyzers import (
    CoberturaAnalyzer,
    GitHubActionsAnalyzer,
    GitLabCIAnalyzer,
    JaCoCoAnalyzer,
    LCOVAnalyzer,
    get_all_coverage_analyzers,
    get_all_pipeline_analyzers,
    get_coverage_analyzer,
    get_pipeline_analyzer,
)
from .manager import PipelineManager
from .protocol import (
    CoverageAnalyzerProtocol,
    CoverageMetrics,
    PipelineAnalysisResult,
    PipelineAnalyzerProtocol,
    PipelineConfig,
    PipelineIssue,
    PipelinePlatform,
    PipelineRun,
    PipelineStatus,
    PipelineStep,
    StepType,
)

__all__ = [
    # Manager
    "PipelineManager",
    # Protocols
    "PipelineAnalyzerProtocol",
    "CoverageAnalyzerProtocol",
    # Data classes
    "PipelinePlatform",
    "PipelineStatus",
    "StepType",
    "PipelineStep",
    "PipelineConfig",
    "PipelineRun",
    "PipelineIssue",
    "PipelineAnalysisResult",
    "CoverageMetrics",
    # Analyzers
    "GitHubActionsAnalyzer",
    "GitLabCIAnalyzer",
    "CoberturaAnalyzer",
    "LCOVAnalyzer",
    "JaCoCoAnalyzer",
    # Registry functions
    "get_pipeline_analyzer",
    "get_coverage_analyzer",
    "get_all_pipeline_analyzers",
    "get_all_coverage_analyzers",
]
