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

"""Shared embedding infrastructure for Victor.

This module provides a unified embedding service that supports:
- Tool selection (static, ~65 items)
- Intent classification (static, ~20 items)
- Code search (dynamic, via LanceDB - handled separately)

Design:
- Single EmbeddingService singleton to load the model once
- StaticEmbeddingCollection for pickle/numpy-backed small collections
- Cosine similarity for semantic matching

All imports are lazy to avoid pulling in optional embedding dependencies
at `import victor` time.
"""


def __getattr__(name: str):
    """Lazy-load submodule symbols on first access."""

    if name == "EmbeddingService":
        from victor.storage.embeddings.service import EmbeddingService

        return EmbeddingService
    if name == "StaticEmbeddingCollection":
        from victor.storage.embeddings.collections import StaticEmbeddingCollection

        return StaticEmbeddingCollection
    if name == "IntentClassifier":
        from victor.storage.embeddings.intent_classifier import IntentClassifier

        return IntentClassifier
    if name in (
        "QuestionType",
        "QuestionTypeClassifier",
        "QuestionClassificationResult",
        "classify_question",
        "should_auto_continue",
    ):
        import victor.storage.embeddings.question_classifier as question_classifier

        return getattr(question_classifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EmbeddingService",
    "StaticEmbeddingCollection",
    "IntentClassifier",
    "QuestionType",
    "QuestionTypeClassifier",
    "QuestionClassificationResult",
    "classify_question",
    "should_auto_continue",
]
