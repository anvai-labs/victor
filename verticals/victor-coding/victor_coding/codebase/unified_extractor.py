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

"""Unified symbol extraction facade with tier-aware strategy selection.

Provides a single interface for symbol extraction that automatically
selects the optimal parsing strategy based on language tier:

- Tier 1 (Python, TS/JS): Native AST + Tree-sitter + LSP enrichment
- Tier 2 (Go, Rust, Java, C/C++): Tree-sitter + LSP enrichment
- Tier 3 (All others): Tree-sitter only

Usage:
    from victor_coding.codebase.unified_extractor import UnifiedSymbolExtractor

    extractor = UnifiedSymbolExtractor(
        tree_sitter=ts_extractor,
        lsp_service=lsp_pool,
    )
    symbols = await extractor.extract_symbols(Path("main.py"))

    for symbol in symbols:
        print(f"{symbol.name}: {symbol.symbol_type} @ line {symbol.line_number}")
        if symbol.return_type:
            print(f"  Returns: {symbol.return_type}")
        if symbol.parameters:
            print(f"  Params: {', '.join(symbol.parameters)}")

SOLID Principles Applied:
    - SRP: Facade delegates to specialized extractors
    - OCP: New tiers/languages add configs, not code changes
    - LSP: All extractors produce compatible EnrichedSymbol
    - ISP: LSPServiceProtocol is minimal interface
    - DIP: Depends on protocols, not concrete implementations
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from victor_coding.languages.tiers import LanguageTier, get_tier
from victor_contracts.utils.ast_helpers import extract_symbols

if TYPE_CHECKING:
    from victor_coding.codebase.tree_sitter_extractor import (
        ExtractedSymbol,
        TreeSitterExtractor,
    )
    from victor_contracts.lsp_runtime import LSPServiceProtocol

logger = logging.getLogger(__name__)


@dataclass
class EnrichedSymbol:
    """Symbol with tier-appropriate enrichment.

    Base fields (all tiers):
        - name, symbol_type, file_path, line_number, end_line
        - signature, docstring, parent_symbol

    Tier 1/2 enrichments (from LSP or native AST):
        - return_type, parameters, visibility
        - is_async, decorators

    Metadata:
        - source_tier: Which tier was used for extraction
    """

    name: str
    symbol_type: str  # class, function, method, variable, etc.
    file_path: str
    line_number: int
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None
    # Tier 1/2 enrichments (from LSP or native AST)
    return_type: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    visibility: Optional[str] = None  # public, private, protected
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    # Source tier for debugging/analytics
    source_tier: Optional[LanguageTier] = None


class UnifiedSymbolExtractor:
    """Unified facade for symbol extraction across all language tiers.

    Automatically selects the optimal extraction strategy based on
    language tier configuration. This centralizes extraction logic
    for use by both the indexer and LSP tools.

    Attributes:
        tree_sitter: TreeSitterExtractor for syntax analysis
        lsp_service: Optional LSP service for type enrichment
        enable_lsp_enrichment: Whether to use LSP (can be disabled)
    """

    def __init__(
        self,
        tree_sitter: Optional["TreeSitterExtractor"] = None,
        lsp_service: Optional["LSPServiceProtocol"] = None,
        enable_lsp_enrichment: bool = True,
    ):
        """Initialize the unified extractor.

        Args:
            tree_sitter: Tree-sitter extractor instance
            lsp_service: LSP service for type enrichment (optional)
            enable_lsp_enrichment: Whether to use LSP for enrichment
        """
        self._ts = tree_sitter
        self._lsp = lsp_service if enable_lsp_enrichment else None
        self._enable_lsp = enable_lsp_enrichment

    def set_tree_sitter(self, tree_sitter: "TreeSitterExtractor") -> None:
        """Set or update the tree-sitter extractor.

        Args:
            tree_sitter: TreeSitterExtractor instance
        """
        self._ts = tree_sitter

    def set_lsp_service(self, lsp_service: Optional["LSPServiceProtocol"]) -> None:
        """Set or update the LSP service.

        Args:
            lsp_service: LSP service instance or None to disable
        """
        if self._enable_lsp:
            self._lsp = lsp_service

    async def extract_symbols(
        self,
        file_path: Path,
        language: Optional[str] = None,
        content: Optional[str] = None,
    ) -> List[EnrichedSymbol]:
        """Extract symbols using optimal strategy for language tier.

        This is the main entry point. It:
        1. Detects language if not provided
        2. Gets tier configuration
        3. Extracts symbols via tree-sitter (all tiers)
        4. Enriches with native AST for Tier 1 Python
        5. Enriches with LSP for Tier 1/2 if available

        Args:
            file_path: Path to the source file
            language: Language identifier (auto-detected if not provided)
            content: File content (read from disk if not provided)

        Returns:
            List of enriched symbols with tier-appropriate information
        """
        if not self._ts:
            logger.warning("TreeSitterExtractor not available, returning empty symbols")
            return []

        # Detect language if not provided
        lang = language or self._ts.detect_language(file_path)
        if not lang:
            logger.debug(f"Could not detect language for {file_path}")
            return []

        tier_config = get_tier(lang)
        logger.debug(
            f"Extracting symbols from {file_path.name} "
            f"(language={lang}, tier={tier_config.tier.name})"
        )

        # Read content if not provided
        if content is None:
            try:
                content = file_path.read_text()
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return []

        # Base extraction via tree-sitter (all tiers)
        try:
            ts_symbols = self._ts.extract_symbols(file_path, lang)
            enriched = [self._convert_ts_symbol(s, file_path, tier_config.tier) for s in ts_symbols]
        except Exception as e:
            logger.warning(f"Tree-sitter extraction failed for {file_path}: {e}")
            enriched = []

        if not enriched:
            return enriched

        # Tier 1: Add native AST enrichment for Python
        if tier_config.tier == LanguageTier.TIER_1 and tier_config.has_native_ast:
            enriched = self._enrich_with_native_ast(content, enriched, lang)

        # Tier 1 & 2: Add LSP enrichment if available
        if (
            tier_config.tier in (LanguageTier.TIER_1, LanguageTier.TIER_2)
            and self._lsp
            and tier_config.has_lsp
        ):
            enriched = await self._enrich_with_lsp(file_path, content, enriched, lang)

        return enriched

    def extract_symbols_sync(
        self,
        file_path: Path,
        language: Optional[str] = None,
        content: Optional[str] = None,
    ) -> List[EnrichedSymbol]:
        """Synchronous symbol extraction (tree-sitter + native AST only).

        Use this for non-async contexts. LSP enrichment is skipped.

        Args:
            file_path: Path to the source file
            language: Language identifier (auto-detected if not provided)
            content: File content (read from disk if not provided)

        Returns:
            List of enriched symbols (without LSP enrichment)
        """
        if not self._ts:
            return []

        lang = language or self._ts.detect_language(file_path)
        if not lang:
            return []

        tier_config = get_tier(lang)

        if content is None:
            try:
                content = file_path.read_text()
            except Exception:
                return []

        try:
            ts_symbols = self._ts.extract_symbols(file_path, lang)
            enriched = [self._convert_ts_symbol(s, file_path, tier_config.tier) for s in ts_symbols]
        except Exception:
            return []

        # Tier 1: Add native AST enrichment for Python
        if tier_config.tier == LanguageTier.TIER_1 and tier_config.has_native_ast:
            enriched = self._enrich_with_native_ast(content, enriched, lang)

        return enriched

    def _convert_ts_symbol(
        self,
        ts_symbol: "ExtractedSymbol",
        file_path: Path,
        tier: LanguageTier,
    ) -> EnrichedSymbol:
        """Convert tree-sitter symbol to EnrichedSymbol."""
        return EnrichedSymbol(
            name=ts_symbol.name,
            symbol_type=ts_symbol.type,
            file_path=str(file_path),
            line_number=ts_symbol.line_number,
            end_line=ts_symbol.end_line,
            parent_symbol=ts_symbol.parent_symbol,
            source_tier=tier,
        )

    def _enrich_with_native_ast(
        self,
        content: str,
        symbols: List[EnrichedSymbol],
        language: str,
    ) -> List[EnrichedSymbol]:
        """Enrich symbols with native AST info (Python only currently).

        Python's ast module provides:
        - Return type annotations
        - Parameter names and types
        - Async function detection
        - Decorator list
        """
        if language != "python":
            return symbols

        try:
            tree = ast.parse(content)
            ast_syms = extract_symbols(tree, enrich=True)
            lookup = {(s.name, s.line_number): s for s in ast_syms}

            for sym in symbols:
                key = (sym.name, sym.line_number)
                if key in lookup:
                    info = lookup[key]
                    sym.return_type = info.return_type
                    sym.parameters = info.parameters
                    sym.is_async = info.is_async
                    sym.decorators = info.decorators
                    sym.docstring = info.docstring
        except SyntaxError as e:
            logger.debug(f"Python AST parse failed: {e}")
        except Exception as e:
            logger.warning(f"Python AST enrichment failed: {e}")

        return symbols

    async def _enrich_with_lsp(
        self,
        file_path: Path,
        content: str,
        symbols: List[EnrichedSymbol],
        language: str,
    ) -> List[EnrichedSymbol]:
        """Enrich symbols with LSP hover info for types.

        LSP provides:
        - Type signatures for functions/methods
        - Return types for languages without native AST
        - Documentation strings
        """
        if not self._lsp:
            return symbols

        try:
            # Open document in LSP
            await self._lsp.open_document(str(file_path), content)

            for sym in symbols:
                try:
                    # Get hover info at symbol definition
                    hover = await self._lsp.get_hover(
                        str(file_path),
                        sym.line_number - 1,  # 0-indexed
                        len(sym.name) // 2,  # Middle of symbol name
                    )
                    if hover and hover.contents:
                        # Parse type/signature from hover
                        parsed = self._parse_hover_contents(hover.contents)
                        if parsed.get("signature"):
                            sym.signature = parsed["signature"]
                        if parsed.get("return_type") and not sym.return_type:
                            sym.return_type = parsed["return_type"]
                        if parsed.get("docstring") and not sym.docstring:
                            sym.docstring = parsed["docstring"]
                except Exception as e:
                    logger.debug(f"LSP hover failed for {sym.name}: {e}")

            self._lsp.close_document(str(file_path))
        except Exception as e:
            logger.debug(f"LSP enrichment failed for {file_path}: {e}")

        return symbols

    def _parse_hover_contents(self, contents: str) -> Dict[str, Optional[str]]:
        """Parse signature and type info from LSP hover contents.

        LSP hover format varies by language server:
        - Python (pyright): "def foo(x: int) -> str"
        - TypeScript (tsserver): "(method) Class.foo(x: number): string"
        - Go (gopls): "func (r *Receiver) Method(x int) string"
        """
        result: Dict[str, Optional[str]] = {
            "signature": None,
            "return_type": None,
            "docstring": None,
        }

        lines = contents.strip().split("\n")
        in_docstring = False
        docstring_lines = []

        for line in lines:
            line = line.strip()

            # Skip markdown code fences
            if line.startswith("```"):
                continue
            if line.startswith("---"):
                in_docstring = True
                continue

            if in_docstring:
                docstring_lines.append(line)
                continue

            # Look for function signatures
            if "(" in line and ")" in line and not result["signature"]:
                result["signature"] = line
                # Extract return type if present
                if " -> " in line:
                    result["return_type"] = line.split(" -> ")[-1].strip()
                elif "): " in line:
                    # TypeScript style: ): ReturnType
                    result["return_type"] = line.split("): ")[-1].strip()
                continue

            # Look for type annotations (variable types)
            if ":" in line and not line.startswith("#") and not result["signature"]:
                result["signature"] = line

        if docstring_lines:
            result["docstring"] = "\n".join(docstring_lines)

        return result
