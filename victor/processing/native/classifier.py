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

"""Task classification and keyword detection with native acceleration."""

from typing import Any, List, Tuple

from victor.processing.native._base import _NATIVE_AVAILABLE, _native
from victor.processing.native.observability import dispatch_with_observability


@dispatch_with_observability("classify_task")
def classify_task_native(text: str) -> Any:
    """Classify a task using native classifier.

    Args:
        text: User message to classify

    Returns:
        ClassificationResult with task type and confidence
    """
    if _NATIVE_AVAILABLE:
        return _native.classify_task(text)

    # Fallback to unified classifier
    from victor.agent.unified_classifier import classify_task

    return classify_task(text)


@dispatch_with_observability("has_action_keywords")
def has_action_keywords(text: str) -> bool:
    """Check if text contains action keywords.

    Args:
        text: Text to check

    Returns:
        True if action keywords are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_action_keywords(text)

    # Pure Python fallback
    action_keywords = [
        "execute",
        "apply",
        "run",
        "deploy",
        "build",
        "install",
        "start",
        "stop",
        "restart",
        "test",
        "commit",
        "push",
        "pull",
        "merge",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in action_keywords)


@dispatch_with_observability("has_analysis_keywords")
def has_analysis_keywords(text: str) -> bool:
    """Check if text contains analysis keywords.

    Args:
        text: Text to check

    Returns:
        True if analysis keywords are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_analysis_keywords(text)

    # Pure Python fallback
    analysis_keywords = [
        "analyze",
        "explore",
        "review",
        "understand",
        "explain",
        "describe",
        "investigate",
        "examine",
        "study",
        "assess",
        "evaluate",
        "summarize",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in analysis_keywords)


@dispatch_with_observability("has_generation_keywords")
def has_generation_keywords(text: str) -> bool:
    """Check if text contains generation keywords.

    Args:
        text: Text to check

    Returns:
        True if generation keywords are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_generation_keywords(text)

    # Pure Python fallback
    generation_keywords = [
        "create",
        "generate",
        "write",
        "implement",
        "add",
        "new",
        "scaffold",
        "initialize",
        "setup",
        "bootstrap",
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in generation_keywords)


@dispatch_with_observability("has_negation")
def has_negation(text: str) -> bool:
    """Check if text contains negation patterns.

    Args:
        text: Text to check

    Returns:
        True if negation patterns are present
    """
    if _NATIVE_AVAILABLE:
        return _native.has_negation(text)

    # Pure Python fallback
    negation_patterns = [
        "don't",
        "do not",
        "dont",
        "shouldn't",
        "should not",
        "wouldn't",
        "would not",
        "can't",
        "cannot",
        "not",
        "never",
        "without",
        "avoid",
        "skip",
        "ignore",
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in negation_patterns)


def find_all_keywords(text: str) -> List[Tuple[int, int, str, str]]:
    """Find all keyword matches in text.

    Args:
        text: Text to search

    Returns:
        List of (start, end, matched_text, category) tuples
    """
    if _NATIVE_AVAILABLE:
        return _native.find_all_keywords(text)

    # Pure Python fallback
    results = []
    text_lower = text.lower()

    keyword_categories = {
        "action": [
            "execute",
            "apply",
            "run",
            "deploy",
            "build",
            "install",
            "start",
            "stop",
            "restart",
            "test",
            "commit",
            "push",
        ],
        "analysis": [
            "analyze",
            "explore",
            "review",
            "understand",
            "explain",
            "describe",
            "investigate",
            "examine",
        ],
        "generation": [
            "create",
            "generate",
            "write",
            "implement",
            "add",
            "scaffold",
        ],
        "search": ["find", "search", "locate", "grep", "look for", "where is"],
        "edit": [
            "modify",
            "refactor",
            "fix",
            "update",
            "change",
            "edit",
            "rename",
        ],
    }

    for category, keywords in keyword_categories.items():
        for keyword in keywords:
            pos = 0
            while True:
                found = text_lower.find(keyword, pos)
                if found == -1:
                    break
                end = found + len(keyword)
                matched = text[found:end]
                results.append((found, end, matched, category))
                pos = found + 1

    results.sort(key=lambda x: x[0])
    return results


# Re-export native classes when available
if _NATIVE_AVAILABLE:
    NativeTaskClassifier = _native.TaskClassifier
    NativeClassificationResult = _native.ClassificationResult
    NativeTaskType = _native.TaskType
else:
    NativeTaskClassifier = None
    NativeClassificationResult = None
    NativeTaskType = None
