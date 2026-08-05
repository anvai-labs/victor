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

"""Language tier definitions for optimal parsing strategy selection.

Tiers determine which parsing strategies (AST, Tree-sitter, LSP) are
available and recommended for each language. This enables the unified
symbol extractor to select the best approach per language.

Tier System:
    - Tier 1: Full AST + Tree-sitter + LSP (Python, TypeScript, JavaScript)
    - Tier 2: Tree-sitter + LSP (Go, Rust, Java, C/C++)
    - Tier 3: Tree-sitter only (Ruby, PHP, and others)

Usage:
    from victor_coding.languages.tiers import get_tier, LanguageTier

    config = get_tier("python")
    if config.tier == LanguageTier.TIER_1 and config.has_native_ast:
        # Use Python AST for deep analysis
        pass
    elif config.has_lsp:
        # Use LSP for type information
        pass
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class LanguageTier(Enum):
    """Language support tiers based on parsing capabilities.

    Higher tiers have more parsing options available.
    """

    TIER_1 = 1  # Full AST + Tree-sitter + LSP (best support)
    TIER_2 = 2  # Tree-sitter + LSP (good support)
    TIER_3 = 3  # Tree-sitter only (basic support)


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a language tier.

    Attributes:
        tier: The tier level
        has_native_ast: Whether native AST parsing is available (Python only)
        has_tree_sitter: Whether tree-sitter grammar is available
        has_lsp: Whether LSP support is configured
        lsp_required: Whether LSP should be auto-started for best results
    """

    tier: LanguageTier
    has_native_ast: bool
    has_tree_sitter: bool
    has_lsp: bool
    lsp_required: bool  # True if LSP should be auto-started


# Language tier registry
# Maps language identifiers to their tier configurations
LANGUAGE_TIERS: Dict[str, TierConfig] = {
    # ==========================================================================
    # Tier 1: Full AST + Tree-sitter + LSP
    # Best support with native AST parsing for deep analysis
    # ==========================================================================
    "python": TierConfig(
        tier=LanguageTier.TIER_1,
        has_native_ast=True,  # Python's ast module
        has_tree_sitter=True,
        has_lsp=True,  # pylsp, pyright
        lsp_required=False,  # Native AST is sufficient
    ),
    "typescript": TierConfig(
        tier=LanguageTier.TIER_1,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,  # tsserver
        lsp_required=True,  # LSP needed for type info
    ),
    "javascript": TierConfig(
        tier=LanguageTier.TIER_1,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,  # tsserver
        lsp_required=True,
    ),
    "tsx": TierConfig(
        tier=LanguageTier.TIER_1,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,
        lsp_required=True,
    ),
    "jsx": TierConfig(
        tier=LanguageTier.TIER_1,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,
        lsp_required=True,
    ),
    # ==========================================================================
    # Tier 2: Tree-sitter + LSP
    # Good support with LSP for semantic analysis
    # ==========================================================================
    "go": TierConfig(
        tier=LanguageTier.TIER_2,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,  # gopls
        lsp_required=False,
    ),
    "rust": TierConfig(
        tier=LanguageTier.TIER_2,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,  # rust-analyzer
        lsp_required=False,
    ),
    "java": TierConfig(
        tier=LanguageTier.TIER_2,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,  # jdtls
        lsp_required=False,
    ),
    "c": TierConfig(
        tier=LanguageTier.TIER_2,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,  # clangd
        lsp_required=False,
    ),
    "cpp": TierConfig(
        tier=LanguageTier.TIER_2,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=True,  # clangd
        lsp_required=False,
    ),
    # ==========================================================================
    # Tier 3: Tree-sitter only
    # Basic support with syntax-based analysis
    # ==========================================================================
    "ruby": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,  # solargraph available but not configured
        lsp_required=False,
    ),
    "php": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "csharp": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,  # OmniSharp available
        lsp_required=False,
    ),
    "swift": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "kotlin": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "scala": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,  # metals available
        lsp_required=False,
    ),
    "lua": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "perl": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "r": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "julia": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "elixir": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "haskell": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,  # hls available
        lsp_required=False,
    ),
    "ocaml": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "bash": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "sql": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "yaml": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "json": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "toml": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "html": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "css": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "markdown": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "vue": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "svelte": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "zig": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,  # zls available
        lsp_required=False,
    ),
    "nim": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "dart": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "clojure": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "erlang": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "terraform": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "dockerfile": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
    "make": TierConfig(
        tier=LanguageTier.TIER_3,
        has_native_ast=False,
        has_tree_sitter=True,
        has_lsp=False,
        lsp_required=False,
    ),
}

# Default tier for unknown languages
_DEFAULT_TIER = TierConfig(
    tier=LanguageTier.TIER_3,
    has_native_ast=False,
    has_tree_sitter=True,  # Attempt tree-sitter for unknown languages
    has_lsp=False,
    lsp_required=False,
)


def get_tier(language: str) -> TierConfig:
    """Get tier configuration for a language.

    Args:
        language: Language identifier (case-insensitive)

    Returns:
        TierConfig for the language, or default Tier 3 config for unknown
    """
    return LANGUAGE_TIERS.get(language.lower(), _DEFAULT_TIER)


def get_languages_by_tier(tier: LanguageTier) -> List[str]:
    """Get all languages in a specific tier.

    Args:
        tier: The tier to filter by

    Returns:
        List of language identifiers in that tier
    """
    return [lang for lang, config in LANGUAGE_TIERS.items() if config.tier == tier]


def is_lsp_recommended(language: str) -> bool:
    """Check if LSP is recommended for a language.

    Args:
        language: Language identifier

    Returns:
        True if LSP enrichment is beneficial for this language
    """
    config = get_tier(language)
    return config.has_lsp and config.tier in (LanguageTier.TIER_1, LanguageTier.TIER_2)


def has_native_ast(language: str) -> bool:
    """Check if native AST parsing is available for a language.

    Args:
        language: Language identifier

    Returns:
        True if native AST parsing is available (currently only Python)
    """
    return get_tier(language).has_native_ast


def get_tier_summary() -> Dict[str, List[str]]:
    """Get a summary of languages by tier.

    Returns:
        Dict mapping tier name to list of languages
    """
    return {
        "tier_1": get_languages_by_tier(LanguageTier.TIER_1),
        "tier_2": get_languages_by_tier(LanguageTier.TIER_2),
        "tier_3": get_languages_by_tier(LanguageTier.TIER_3),
    }
