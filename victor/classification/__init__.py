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

"""Unified classification module for task type and complexity detection.

This module consolidates pattern matching and classification logic that was
previously duplicated between TaskTypeClassifier and TaskComplexityService.

Architecture (SOLID compliant):
- pattern_registry.py: Single source of truth for all classification patterns
- protocols.py: Interfaces for classifier components (DIP compliance)
- nudge_engine.py: Post-classification edge case corrections (SRP)

Usage:
    from victor.classification import PATTERNS, ClassificationPattern, NudgeEngine
    from victor.classification.protocols import TaskClassifierProtocol
"""

from victor.classification.pattern_registry import (
    PATTERNS,
    ClassificationPattern,
    TaskType,
    TASK_TYPE_TO_COMPLEXITY,
    get_patterns_by_complexity,
    get_patterns_by_task_type,
    get_patterns_sorted_by_priority,
    match_first_pattern,
    match_all_patterns,
)
from victor.classification.nudge_engine import (
    NudgeEngine,
    NudgeRule,
    PatternMatcher,
    get_nudge_engine,
    get_pattern_matcher,
)
from victor.classification.protocols import (
    TaskTypeResult,
    SemanticClassifierProtocol,
    PatternMatcherProtocol,
    NudgeEngineProtocol,
)

__all__ = [
    # Pattern registry
    "PATTERNS",
    "ClassificationPattern",
    "TaskType",
    "TASK_TYPE_TO_COMPLEXITY",
    "get_patterns_by_complexity",
    "get_patterns_by_task_type",
    "get_patterns_sorted_by_priority",
    "match_first_pattern",
    "match_all_patterns",
    # Nudge engine
    "NudgeEngine",
    "NudgeRule",
    "PatternMatcher",
    "get_nudge_engine",
    "get_pattern_matcher",
    # Protocols
    "TaskTypeResult",
    "SemanticClassifierProtocol",
    "PatternMatcherProtocol",
    "NudgeEngineProtocol",
]
