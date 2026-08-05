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

"""LSP Write Enhancer for intelligent code writing.

This module provides LSP-enhanced write functionality:
- Syntax validation before writing
- Auto-formatting using language-specific formatters
- Diagnostic information (errors, warnings, hints)
- Type checking integration

Usage:
    from victor.tools.lsp_write_enhancer import write_with_lsp

    result = await write_with_lsp(
        path="src/main.py",
        content="def hello():\\n    print('Hello')\\n",
        validate=True,
        format=True,
    )
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.core.verticals.import_resolver import module_import_candidates

if TYPE_CHECKING:
    from victor.framework.protocols import LanguageRegistryProtocol, LSPPoolProtocol

logger = logging.getLogger(__name__)


class DiagnosticSeverity(Enum):
    """LSP diagnostic severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class Diagnostic:
    """A diagnostic message from LSP."""

    line: int
    column: int
    severity: DiagnosticSeverity
    message: str
    code: Optional[str] = None
    source: Optional[str] = None


@dataclass
class WriteResult:
    """Result of an LSP-enhanced write operation."""

    success: bool
    path: str
    original_content: Optional[str] = None
    written_content: Optional[str] = None
    formatted: bool = False
    validated: bool = False
    diagnostics: List[Diagnostic] = field(default_factory=list)
    formatter_used: Optional[str] = None
    error: Optional[str] = None

    @property
    def has_errors(self) -> bool:
        """Check if any error-level diagnostics exist."""
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    @property
    def has_warnings(self) -> bool:
        """Check if any warning-level diagnostics exist."""
        return any(d.severity == DiagnosticSeverity.WARNING for d in self.diagnostics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "path": self.path,
            "formatted": self.formatted,
            "validated": self.validated,
            "diagnostics": [
                {
                    "line": d.line,
                    "column": d.column,
                    "severity": d.severity.value,
                    "message": d.message,
                    "code": d.code,
                    "source": d.source,
                }
                for d in self.diagnostics
            ],
            "formatter_used": self.formatter_used,
            "error": self.error,
            "summary": {
                "total_diagnostics": len(self.diagnostics),
                "errors": sum(
                    1 for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR
                ),
                "warnings": sum(
                    1 for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING
                ),
                "info": sum(1 for d in self.diagnostics if d.severity == DiagnosticSeverity.INFO),
                "hints": sum(1 for d in self.diagnostics if d.severity == DiagnosticSeverity.HINT),
            },
        }


class LSPWriteEnhancer:
    """Enhances write operations with LSP validation and formatting."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        lsp_pool: Optional["LSPPoolProtocol"] = None,
        language_registry: Optional["LanguageRegistryProtocol"] = None,
    ):
        """Initialize the LSP write enhancer.

        Args:
            workspace_root: Root directory for LSP operations
            lsp_pool: Optional pre-configured LSP pool (avoids vertical dependency)
            language_registry: Optional pre-configured language registry
        """
        self._workspace_root = workspace_root or str(Path.cwd())
        self._lsp_pool: Optional["LSPPoolProtocol"] = lsp_pool
        self._language_registry: Optional["LanguageRegistryProtocol"] = language_registry
        # Use in-process black library by default (faster, no subprocess overhead)
        self._use_thread_formatter = True

    async def _get_lsp_pool(self) -> "LSPPoolProtocol":
        """Get or create LSP connection pool.

        Uses injected pool if available, otherwise falls back to coding vertical.
        """
        if self._lsp_pool is None:
            lsp_pool_cls = self._load_optional_symbol(
                "victor.coding.lsp.manager",
                "LSPConnectionPool",
            )
            if lsp_pool_cls is None:
                raise ImportError(
                    "LSP support requires the coding vertical package. "
                    "Install 'victor-coding' or inject an LSPPoolProtocol instance."
                )
            self._lsp_pool = lsp_pool_cls(self._workspace_root)
        return self._lsp_pool

    def _get_language_registry(self) -> "LanguageRegistryProtocol":
        """Get or create language registry.

        Uses injected registry if available, otherwise falls back to coding vertical.
        """
        if self._language_registry is None:
            get_registry = self._load_optional_symbol(
                "victor.coding.languages.registry",
                "get_language_registry",
            )
            if get_registry is None:
                raise ImportError(
                    "Language registry requires the coding vertical package. "
                    "Install 'victor-coding' or inject a LanguageRegistryProtocol instance."
                )
            self._language_registry = get_registry()
        return self._language_registry

    def _load_optional_symbol(self, module_path: str, symbol_name: str) -> Optional[Any]:
        """Load symbol from the first importable compatibility module path."""
        for candidate in module_import_candidates(module_path):
            try:
                module = importlib.import_module(candidate)
                return getattr(module, symbol_name)
            except Exception:
                continue
        return None

    async def validate_with_lsp(self, path: str, content: str) -> List[Diagnostic]:
        """Validate code content using LSP.

        Args:
            path: File path (for language detection)
            content: File content to validate

        Returns:
            List of diagnostics
        """
        pool = await self._get_lsp_pool()

        # Open document with content
        uri = pool._path_to_uri(path)
        pool.open_document(uri, content)

        # Wait a moment for LSP to process
        await asyncio.sleep(0.1)

        # Get diagnostics
        diagnostics = pool.get_diagnostics(path)

        # Close document
        pool.close_document(path)

        return [
            Diagnostic(
                line=d["line"],
                column=d.get("column", 0),
                severity=DiagnosticSeverity(d["severity"]),
                message=d["message"],
                code=d.get("code"),
                source=d.get("source"),
            )
            for d in diagnostics
        ]

    async def format_with_formatter(self, path: str, content: str) -> tuple[str, Optional[str]]:
        """Format code using language-specific formatter.

        Args:
            path: File path (for language detection)
            content: Content to format

        Returns:
            Tuple of (formatted_content, formatter_name)
        """
        registry = self._get_language_registry()
        file_path = Path(path)

        # Detect language
        language = registry.detect_language(file_path)
        if not language:
            return content, None

        # Get language plugin
        try:
            plugin = registry.get(language)
        except KeyError:
            return content, None

        # Get formatter
        project_root = Path(self._workspace_root)
        formatter = plugin.get_formatter(project_root) if hasattr(plugin, "get_formatter") else None

        if not formatter:
            return content, None

        # Try in-process formatting first (faster, no subprocess overhead)
        if formatter.name == "black" and self._use_thread_formatter:
            try:
                # Offload to thread to avoid blocking event loop
                import asyncio

                formatted = await asyncio.to_thread(
                    self._format_with_black_library, content, file_path
                )
                if formatted is not None:
                    return formatted, "black (in-process)"
            except Exception as e:
                logger.debug(f"In-process black failed, falling to subprocess: {e}")

        # Fallback: subprocess-based formatting
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=file_path.suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        try:
            cmd = formatter.command.copy()
            cmd.append(tmp_path)

            # Use asyncio subprocess to avoid blocking event loop
            import asyncio

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._workspace_root,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            if proc.returncode == 0:
                with open(tmp_path, "r") as f:
                    formatted = f.read()
                return formatted, formatter.name
            else:
                logger.warning(
                    f"Formatter {formatter.name} failed: {stderr.decode('utf-8', errors='ignore')}"
                )
                return content, None

        except asyncio.TimeoutError:
            logger.warning(f"Formatter {formatter.name} timed out")
            return content, None
        except FileNotFoundError:
            # Formatter not installed
            logger.debug(f"Formatter {formatter.name} not found")
            return content, None
        except Exception as e:
            logger.warning(f"Formatting failed: {e}")
            return content, None
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _format_with_black_library(self, content: str, file_path: Path) -> Optional[str]:
        """Format Python code using black as a library (no subprocess).

        Much faster than subprocess — avoids process spawn overhead.
        Returns formatted content or None if black unavailable/fails.
        """
        if file_path.suffix != ".py":
            return None
        try:
            import black

            mode = black.Mode(line_length=100, target_versions={black.TargetVersion.PY310})
            return black.format_str(content, mode=mode)
        except ImportError:
            return None
        except black.NothingChanged:
            return content
        except Exception:
            return None

    async def write_with_lsp(
        self,
        path: str,
        content: str,
        *,
        validate: bool = True,
        format_code: bool = True,
        write: bool = True,
    ) -> WriteResult:
        """Write file with LSP validation and formatting.

        Args:
            path: File path to write
            content: Content to write
            validate: Whether to validate with LSP
            format_code: Whether to format with language formatter
            write: Whether to actually write the file (False for dry-run)

        Returns:
            WriteResult with diagnostics and formatting info
        """
        result = WriteResult(success=False, path=path)

        # Format if requested
        if format_code:
            formatted_content, formatter_name = await self.format_with_formatter(path, content)
            result.formatted = formatter_name is not None
            result.formatter_used = formatter_name
            content_to_write = formatted_content
        else:
            content_to_write = content

        result.written_content = content_to_write

        # Validate if requested
        if validate:
            try:
                diagnostics = await self.validate_with_lsp(path, content_to_write)
                result.diagnostics = diagnostics
                result.validated = True
            except Exception as e:
                logger.warning(f"LSP validation failed: {e}")
                result.validized = False

        # Write file if requested and no errors
        if write:
            if result.has_errors:
                result.success = False
                result.error = "File has errors - not written"
            else:
                try:
                    file_path = Path(path).expanduser().resolve()
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save original if exists
                    if file_path.exists():
                        result.original_content = file_path.read_text()

                    # Write new content
                    file_path.write_text(content_to_write, encoding="utf-8")
                    result.success = True
                except Exception as e:
                    result.success = False
                    result.error = str(e)
        else:
            # Dry run - report would have been written
            result.success = not result.has_errors

        return result


# Global enhancer instance
_enhancer: Optional[LSPWriteEnhancer] = None


def get_write_enhancer(workspace_root: Optional[str] = None) -> LSPWriteEnhancer:
    """Get the global LSP write enhancer.

    Args:
        workspace_root: Optional workspace root

    Returns:
        LSPWriteEnhancer instance
    """
    global _enhancer
    if _enhancer is None or workspace_root is not None:
        _enhancer = LSPWriteEnhancer(workspace_root)
    return _enhancer


async def write_with_lsp(
    path: str,
    content: str,
    *,
    validate: bool = True,
    format_code: bool = True,
    write: bool = True,
    workspace_root: Optional[str] = None,
) -> WriteResult:
    """Convenience function to write with LSP enhancement.

    Args:
        path: File path to write
        content: Content to write
        validate: Whether to validate with LSP
        format_code: Whether to format with language formatter
        write: Whether to actually write the file
        workspace_root: Optional workspace root

    Returns:
        WriteResult with diagnostics and formatting info

    Example:
        result = await write_with_lsp(
            "src/main.py",
            "def hello(): print('hi')",
            validate=True,
            format_code=True,
        )

        if result.has_errors:
            for d in result.diagnostics:
                print(f"{d.line}:{d.column} - {d.message}")
    """
    enhancer = get_write_enhancer(workspace_root)
    return await enhancer.write_with_lsp(
        path, content, validate=validate, format_code=format_code, write=write
    )


__all__ = [
    "DiagnosticSeverity",
    "Diagnostic",
    "WriteResult",
    "LSPWriteEnhancer",
    "write_with_lsp",
    "get_write_enhancer",
]
