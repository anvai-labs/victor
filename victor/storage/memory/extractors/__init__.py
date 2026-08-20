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

"""Entity extraction pipeline for Victor memory system.

Provides extractors for different entity types:
- CodeEntityExtractor: Functions, classes, modules from code
- TextEntityExtractor: People, organizations, concepts from text
- CompositeExtractor: Combines multiple extractors

Example:
    extractor = CompositeExtractor([
        CodeEntityExtractor(),
        TextEntityExtractor(),
    ])
    entities = await extractor.extract(text, source="conversation")
"""

from victor.storage.memory.extractors.base import (
    EntityExtractor,
    ExtractionResult,
)
from victor.storage.memory.extractors.code_extractor import CodeEntityExtractor
from victor.storage.memory.extractors.text_extractor import TextEntityExtractor
from victor.storage.memory.extractors.composite import CompositeExtractor

# Tree-sitter extractor (optional, requires tree-sitter)
try:
    from victor.storage.memory.extractors.tree_sitter_extractor import (
        TreeSitterEntityExtractor,
        TreeSitterFileExtractor,
    )

    _HAS_TREE_SITTER = True
except ImportError:
    TreeSitterEntityExtractor = None  # type: ignore
    TreeSitterFileExtractor = None  # type: ignore
    _HAS_TREE_SITTER = False

__all__ = [
    "EntityExtractor",
    "ExtractionResult",
    "CodeEntityExtractor",
    "TextEntityExtractor",
    "CompositeExtractor",
    "TreeSitterEntityExtractor",
    "TreeSitterFileExtractor",
    "has_tree_sitter",
    "create_extractor",
]


def has_tree_sitter() -> bool:
    """Check if Tree-sitter support is available."""
    return _HAS_TREE_SITTER


def create_extractor(
    use_tree_sitter: bool = True,
    include_text: bool = True,
    include_code: bool = True,
) -> CompositeExtractor:
    """Create an entity extractor with configurable backends.

    Args:
        use_tree_sitter: Use Tree-sitter for accurate AST parsing (if available)
        include_text: Include text entity extraction (orgs, technologies)
        include_code: Include regex-based code extraction (fallback)

    Returns:
        Configured CompositeExtractor
    """
    extractors = []

    # Add Tree-sitter extractor if available and requested
    if use_tree_sitter and _HAS_TREE_SITTER:
        extractors.append(TreeSitterEntityExtractor())
    elif include_code:
        # Fallback to regex-based code extraction
        extractors.append(CodeEntityExtractor())

    # Add text extractor
    if include_text:
        extractors.append(TextEntityExtractor())

    return CompositeExtractor(
        extractors=extractors,
        parallel=True,
        dedup_strategy="merge",
    )
