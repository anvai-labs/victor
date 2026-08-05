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

"""Multi-language support infrastructure for Victor.

This module provides a pluggable language support system enabling:
- Language detection from files and content
- Language-specific syntax analysis
- Test runner integration per language
- Build system integration
- Formatter and linter integration

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Language Manager                           │
    │  (Discovers plugins, routes requests to language handlers)  │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   Language Registry                          │
    │  (Maps extensions/names to language plugins)                │
    └─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │  Python   │       │ JavaScript│       │   Rust    │
    │  Plugin   │       │  Plugin   │       │  Plugin   │
    └───────────┘       └───────────┘       └───────────┘

Each language plugin provides:
- File extension mappings
- Syntax configuration
- Test runner commands
- Build/run commands
- Formatter/linter commands
- Language server configuration
"""

from victor_coding.languages.base import (
    LanguagePlugin,
    LanguageCapabilities,
    TestRunner,
    BuildSystem,
    Formatter,
    Linter,
    LanguageConfig,
)
from victor_coding.languages.registry import LanguageRegistry, get_language_registry
from victor_coding.languages.manager import LanguageManager
from victor_coding.languages.tiers import (
    LanguageTier,
    TierConfig,
    LANGUAGE_TIERS,
    get_tier,
    get_languages_by_tier,
    is_lsp_recommended,
    has_native_ast,
    get_tier_summary,
)

__all__ = [
    # Base types
    "LanguagePlugin",
    "LanguageCapabilities",
    "TestRunner",
    "BuildSystem",
    "Formatter",
    "Linter",
    "LanguageConfig",
    # Registry and manager
    "LanguageRegistry",
    "get_language_registry",
    "LanguageManager",
    # Tier system
    "LanguageTier",
    "TierConfig",
    "LANGUAGE_TIERS",
    "get_tier",
    "get_languages_by_tier",
    "is_lsp_recommended",
    "has_native_ast",
    "get_tier_summary",
]
