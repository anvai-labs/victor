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

"""Merge Conflict Resolution Module - Intelligent conflict handling.

This module provides AI-assisted merge conflict resolution including:
- Automatic conflict detection from git status
- Intelligent resolution strategies
- Import statement conflict handling
- Trivial conflict (whitespace) resolution
- Union strategy for additive changes

Usage:
    from victor.processing.merge import MergeManager

    manager = MergeManager("/path/to/repo")

    # Detect conflicts
    conflicts = await manager.detect_conflicts()

    # Analyze
    analysis = await manager.analyze_conflicts(conflicts)
    print(f"Auto-resolvable: {analysis.auto_resolvable}")

    # Attempt resolution
    resolutions = await manager.resolve_conflicts(auto_apply=True)
"""

from .manager import MergeManager
from .protocol import (
    ConflictAnalysis,
    ConflictComplexity,
    ConflictHunk,
    ConflictResolverProtocol,
    ConflictType,
    FileConflict,
    FileResolution,
    MergeAnalyzerProtocol,
    MergeContext,
    Resolution,
    ResolutionStrategy,
)
from .resolvers import (
    DefaultMergeAnalyzer,
    ImportResolver,
    TrivialResolver,
    UnionResolver,
    get_analyzer,
    get_resolvers,
)

__all__ = [
    # Manager
    "MergeManager",
    # Protocols
    "ConflictResolverProtocol",
    "MergeAnalyzerProtocol",
    # Data classes
    "ConflictType",
    "ResolutionStrategy",
    "ConflictComplexity",
    "ConflictHunk",
    "FileConflict",
    "Resolution",
    "FileResolution",
    "MergeContext",
    "ConflictAnalysis",
    # Resolvers
    "TrivialResolver",
    "UnionResolver",
    "ImportResolver",
    "DefaultMergeAnalyzer",
    # Registry functions
    "get_resolvers",
    "get_analyzer",
]
