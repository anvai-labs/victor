# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Victor Contrib - Shared optional packages for verticals.

This package contains reusable components that are shared across multiple
verticals but are not part of the core framework. These packages provide
ready-made functionality that verticals can consume to avoid code duplication.

Architecture:
- Core (victor/): Essential framework (Agent, StateGraph, Tools, Events)
- Contrib (victor/contrib/): Shared optional packages for verticals
- Verticals (victor-xyz/): Domain-specific implementations

Contrib Packages:
- safety: Enhanced safety extensions for all verticals
- conversation: Enhanced conversation management for all verticals
- workflows: YAML workflow escape hatch base classes
- mode_config: Mode configuration provider base classes
- testing: Common testing utilities for verticals

Usage:
    # In a vertical (e.g., victor-devops)
    from victor.contrib.safety import BaseSafetyExtension
    from victor.contrib.conversation import BaseConversationManager

    class DevOpsSafetyExtension(BaseSafetyExtension):
        def get_vertical_name(self) -> str:
            return "devops"

        def get_vertical_rules(self) -> List[SafetyPattern]:
            # DevOps-specific safety rules
            return [...]
"""

__version__ = "0.1.0"

# Import main contrib packages for convenience
from victor.contrib import (
    codebase,  # Codebase analysis utilities
    editing,  # File editing utilities
    embeddings,  # Text embeddings
    languages,  # Language registry
    lsp,  # LSP integration
    parsing,  # Tree-sitter parsing
    prompts,  # Task type hints
    safety,
    conversation,
    vectorstores,  # Vector storage
    workflows,
    mode_config,
)


def __getattr__(name: str):
    # 'testing' hard-imports pytest (VerticalTestCase et al.), which production
    # installs don't ship — an eager import here fails the entire capabilities
    # bootstrap phase. Resolve it lazily so only test-time consumers pay for it.
    if name == "testing":
        import importlib

        return importlib.import_module("victor.contrib.testing")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core packages
    "codebase",
    "editing",
    "embeddings",
    "languages",
    "lsp",
    "parsing",
    "prompts",
    "vectorstores",
    # Existing packages
    "safety",
    "conversation",
    "workflows",
    "mode_config",
    "testing",
]
