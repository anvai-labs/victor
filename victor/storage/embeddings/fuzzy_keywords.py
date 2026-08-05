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

"""
Fuzzy matching keyword dictionaries for classification systems.

This module provides specialized keyword dictionaries for different classification
types (task, intent, tool selection, safety). Each dictionary maps keywords to
weights that reflect their importance in classification decisions.

These dictionaries are used with the fuzzy matching module to provide robust
classification that handles typos and spelling variations.

Usage:
    >>> from victor.storage.embeddings.fuzzy_keywords import TASK_CLASSIFICATION_KEYWORDS
    >>> from victor.storage.embeddings.fuzzy_matcher import match_keywords_cascading
    >>> matches, stats = match_keywords_cascading("analize framework", TASK_CLASSIFICATION_KEYWORDS)
    >>> matches
    {'analyze', 'framework'}
"""

from typing import Dict

# =============================================================================
# Task Classification Keywords
# =============================================================================

TASK_CLASSIFICATION_KEYWORDS: Dict[str, float] = {
    # Core analysis (highest weight for safety)
    "analyze": 1.5,
    "analysis": 1.5,
    "review": 1.4,
    "audit": 1.4,
    "examine": 1.3,
    "inspect": 1.3,
    "investigate": 1.3,
    # Structural keywords (framework/architecture)
    "structure": 1.2,
    "architecture": 1.2,
    "framework": 1.1,
    "design": 1.1,
    "system": 1.0,
    # Creation keywords
    "create": 1.3,
    "generate": 1.3,
    "write": 1.2,
    "make": 1.1,
    "implement": 1.2,
    "add": 1.0,
    # Edit keywords
    "edit": 1.2,
    "refactor": 1.3,
    "fix": 1.2,
    "modify": 1.2,
    "change": 1.0,
    "update": 1.0,
    # Search keywords
    "search": 1.3,
    "find": 1.2,
    "locate": 1.2,
    "grep": 1.3,
    "look": 0.8,
    "list": 0.7,
    # Execution keywords (safety-critical)
    "execute": 1.2,
    "run": 1.2,
    "deploy": 1.2,
    # Testing keywords
    "test": 1.2,
    "testing": 1.2,
    "coverage": 1.1,
}

# =============================================================================
# Intent Classification Keywords
# =============================================================================

INTENT_CLASSIFICATION_KEYWORDS: Dict[str, float] = {
    # Continuation signals
    "continue": 1.5,
    "next": 1.4,
    "proceed": 1.4,
    "read": 1.3,
    "check": 1.3,
    "examine": 1.3,
    "let me": 1.2,
    # Completion signals
    "complete": 1.5,
    "done": 1.4,
    "finish": 1.4,
    "summary": 1.3,
    "conclusion": 1.3,
    # Asking input signals
    "should": 1.3,
    "prefer": 1.3,
    "like": 1.2,
    "confirm": 1.3,
    "approve": 1.3,
}

# =============================================================================
# Tool Selection Keywords
# =============================================================================

TOOL_SELECTION_KEYWORDS: Dict[str, float] = {
    # File operations
    "read": 1.5,
    "write": 1.5,
    "edit": 1.4,
    # Search operations
    "search": 1.5,
    "find": 1.4,
    "grep": 1.5,
    "locate": 1.3,
    # Execution
    "run": 1.4,
    "execute": 1.4,
    "shell": 1.3,
    # Version control
    "commit": 1.4,
    "push": 1.4,
    "pull": 1.3,
    # Testing
    "test": 1.4,
    "pytest": 1.5,
}

# =============================================================================
# Safety-Critical Keywords
# =============================================================================

SAFETY_KEYWORDS: Dict[str, float] = {
    # Destructive operations (highest weight)
    "delete": 2.0,
    "remove": 1.8,
    "destroy": 2.0,
    "drop": 1.8,
    "truncate": 1.8,
    # System modifications
    "modify": 1.5,
    "change": 1.4,
    "alter": 1.5,
    # Data operations
    "overwrite": 1.8,
    "replace": 1.5,
}

# =============================================================================
# Additional Specialized Dictionaries
# =============================================================================

# Code analysis specific keywords
CODE_ANALYSIS_KEYWORDS: Dict[str, float] = {
    "refactor": 1.5,
    "optimize": 1.4,
    "improve": 1.3,
    "simplify": 1.3,
    "cleanup": 1.2,
    "reorganize": 1.2,
    "restructure": 1.3,
}

# Documentation keywords
DOCUMENTATION_KEYWORDS: Dict[str, float] = {
    "document": 1.5,
    "docs": 1.4,
    "readme": 1.4,
    "comment": 1.3,
    "explain": 1.3,
    "describe": 1.2,
}

# Debug keywords
DEBUG_KEYWORDS: Dict[str, float] = {
    "debug": 1.5,
    "fix": 1.4,
    "error": 1.3,
    "bug": 1.3,
    "issue": 1.2,
    "problem": 1.2,
    "broken": 1.2,
}

# Deployment keywords
DEPLOYMENT_KEYWORDS: Dict[str, float] = {
    "deploy": 1.5,
    "release": 1.4,
    "publish": 1.3,
    "ship": 1.3,
    "production": 1.2,
    "staging": 1.1,
}

# =============================================================================
# Export all dictionaries
# =============================================================================

__all__ = [
    "TASK_CLASSIFICATION_KEYWORDS",
    "INTENT_CLASSIFICATION_KEYWORDS",
    "TOOL_SELECTION_KEYWORDS",
    "SAFETY_KEYWORDS",
    "CODE_ANALYSIS_KEYWORDS",
    "DOCUMENTATION_KEYWORDS",
    "DEBUG_KEYWORDS",
    "DEPLOYMENT_KEYWORDS",
]
