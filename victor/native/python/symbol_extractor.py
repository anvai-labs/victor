# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Native symbol protocol projection backed by victor-codegraph v2.

This module intentionally contains no second semantic parser. Raw identifier scanning
remains a separate protocol operation and uses one small lexical expression.
"""

from __future__ import annotations

import re
import sys
from typing import Any, List, Optional

from victor.core.codegraph_adapter import parse_code, project_python_imports
from victor.native.observability import InstrumentedAccelerator
from victor.native.protocols import Symbol, SymbolType

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FUNCTION_TYPES = {"FUNCTION", "METHOD", "CONSTRUCTOR"}
_CLASS_TYPES = {"CLASS", "INTERFACE", "TRAIT", "STRUCT", "ENUM"}


class PythonSymbolExtractor(InstrumentedAccelerator):
    """Compatibility implementation of ``SymbolExtractorProtocol``.

    The historic name is retained for callers selecting the Python backend, while
    semantic extraction itself is owned by the shared CodeGraph parser.
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"

    def get_version(self) -> Optional[str]:
        return self._version

    def extract_functions(self, source: str, lang: str) -> List[Symbol]:
        """Extract callable symbols through the canonical parser."""
        with self._timed_call("extract_functions", lang=lang):
            return self._project_symbols(source, lang, _FUNCTION_TYPES)

    def extract_classes(self, source: str, lang: str) -> List[Symbol]:
        """Extract class-like symbols through the canonical parser."""
        with self._timed_call("extract_classes", lang=lang):
            return self._project_symbols(source, lang, _CLASS_TYPES)

    def extract_imports(self, source: str, lang: str) -> List[str]:
        """Extract imported module names from CodeGraph's parse result."""
        with self._timed_call("extract_imports", lang=lang):
            parsed = parse_code(source, file_path="<memory>", language=lang)
            if parsed is None:
                return []
            if parsed.language == "python":
                return sorted({item.module for item in project_python_imports(parsed.imports)})
            return sorted(set(parsed.imports))

    def extract_references(self, source: str) -> List[str]:
        """Extract lexical identifier references; this is not semantic parsing."""
        with self._timed_call("reference_extraction"):
            return IDENTIFIER_PATTERN.findall(source)

    def is_stdlib_module(self, name: str) -> bool:
        """Check a top-level module against Python's runtime-owned stdlib set."""
        with self._timed_call("stdlib_check"):
            top_level = name.split(".", 1)[0]
            return top_level in sys.stdlib_module_names or top_level == "typing_extensions"

    def _project_symbols(
        self,
        source: str,
        language: str,
        accepted_types: set[str],
    ) -> List[Symbol]:
        parsed = parse_code(source, file_path="<memory>", language=language)
        if parsed is None:
            return []
        return [
            self._to_native_symbol(symbol)
            for symbol in parsed.symbols
            if symbol.symbol_type.name in accepted_types
        ]

    @staticmethod
    def _to_native_symbol(symbol: Any) -> Symbol:
        kind = symbol.symbol_type.name
        native_type = (
            SymbolType.FUNCTION
            if kind == "FUNCTION"
            else SymbolType.METHOD if kind in {"METHOD", "CONSTRUCTOR"} else SymbolType.CLASS
        )
        modifiers = list(symbol.modifiers)
        decorators = tuple(item[1:] for item in modifiers if item.startswith("@"))
        visibility = "private" if "private" in modifiers else "public"
        if kind in _FUNCTION_TYPES:
            prefix = "async def" if "async" in modifiers else "def"
            signature = f"{prefix} {symbol.signature or symbol.simple_name + '()'}"
        else:
            bases = next(
                (
                    item[8:-1]
                    for item in modifiers
                    if item.startswith("extends(") and item.endswith(")")
                ),
                "",
            )
            signature = f"class {symbol.simple_name}{f'({bases})' if bases else ''}:"
        return Symbol(
            name=symbol.simple_name,
            type=native_type,
            line=symbol.location.start_line,
            end_line=symbol.location.end_line,
            signature=signature,
            docstring=symbol.documentation or "",
            decorators=decorators,
            parent=symbol.scope_chain[-1] if symbol.scope_chain else None,
            visibility=visibility,
        )


__all__ = ["PythonSymbolExtractor"]
