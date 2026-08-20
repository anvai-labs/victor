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

"""Context command parser for @-prefixed context references.

This module parses and resolves @-prefixed context commands similar to
Claude Code and other AI coding assistants:

- @url:<url> - Fetch and include URL content
- @file:<path> - Include file content
- @folder:<path> - Include folder structure and key files
- @problems - Include workspace diagnostics/errors

SOLID Principles Applied:
- Single Responsibility: Each resolver handles one type of context
- Open/Closed: New resolvers can be added without modifying existing code
- Liskov Substitution: All resolvers are interchangeable
- Interface Segregation: Resolver interface is minimal
- Dependency Inversion: Parser depends on abstractions, not concrete resolvers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import re
import logging
import asyncio
import aiohttp
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """Represents a resolved context item."""

    source_type: str  # "url", "file", "folder", "problems"
    source: str  # Original reference (URL, path, etc.)
    content: str  # Resolved content
    tokens: int = 0  # Token count (set after resolution)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class ParsedCommand:
    """A parsed @-command from user input."""

    command_type: str  # "url", "file", "folder", "problems"
    argument: str  # The argument after the colon
    original_text: str  # Original text that was matched
    start_pos: int  # Position in original input
    end_pos: int  # End position in original input


class ContextResolver(ABC):
    """Abstract resolver for context commands."""

    @property
    @abstractmethod
    def command_type(self) -> str:
        """The command type this resolver handles."""
        pass

    @abstractmethod
    async def resolve(self, argument: str, context: Dict[str, Any]) -> ContextItem:
        """
        Resolve the command and return context item.

        Args:
            argument: The argument after @command:
            context: Additional context (working_dir, etc.)

        Returns:
            ContextItem with resolved content
        """
        pass


class URLResolver(ContextResolver):
    """Resolves @url:<url> commands by fetching URL content."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 100000,
        user_agent: str = "Victor/1.0 (AI Coding Assistant)",
    ) -> None:
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent

    @property
    def command_type(self) -> str:
        return "url"

    async def resolve(self, argument: str, context: Dict[str, Any]) -> ContextItem:
        url = argument.strip()

        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme:
                url = f"https://{url}"
            elif parsed.scheme not in ("http", "https"):
                return ContextItem(
                    source_type="url",
                    source=url,
                    content="",
                    success=False,
                    error=f"Unsupported URL scheme: {parsed.scheme}",
                )
        except Exception as e:
            return ContextItem(
                source_type="url",
                source=url,
                content="",
                success=False,
                error=f"Invalid URL: {e}",
            )

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": self.user_agent}
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        return ContextItem(
                            source_type="url",
                            source=url,
                            content="",
                            success=False,
                            error=f"HTTP {response.status}: {response.reason}",
                        )

                    content_type = response.headers.get("Content-Type", "")
                    content = await response.text()

                    # Truncate if too long
                    if len(content) > self.max_content_length:
                        content = content[: self.max_content_length]
                        content += (
                            f"\n\n[Content truncated at {self.max_content_length} characters]"
                        )

                    return ContextItem(
                        source_type="url",
                        source=url,
                        content=content,
                        metadata={
                            "content_type": content_type,
                            "status": response.status,
                            "truncated": len(content) >= self.max_content_length,
                        },
                    )

        except asyncio.TimeoutError:
            return ContextItem(
                source_type="url",
                source=url,
                content="",
                success=False,
                error=f"Request timed out after {self.timeout}s",
            )
        except Exception as e:
            return ContextItem(
                source_type="url",
                source=url,
                content="",
                success=False,
                error=f"Failed to fetch URL: {e}",
            )


class FileResolver(ContextResolver):
    """Resolves @file:<path> commands by reading file content."""

    def __init__(
        self,
        max_file_size: int = 500000,
        allowed_extensions: Optional[Set[str]] = None,
    ) -> None:
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions

    @property
    def command_type(self) -> str:
        return "file"

    async def resolve(self, argument: str, context: Dict[str, Any]) -> ContextItem:
        working_dir = Path(context.get("working_directory", "."))
        file_path_str = argument.strip()

        # Resolve path
        if file_path_str.startswith("/"):
            file_path = Path(file_path_str)
        else:
            file_path = working_dir / file_path_str

        file_path = file_path.resolve()

        # Security: Ensure path is within working directory or allowed
        try:
            file_path.relative_to(working_dir.resolve())
        except ValueError:
            # Allow absolute paths but log warning
            logger.warning(f"File path {file_path} is outside working directory")

        if not file_path.exists():
            return ContextItem(
                source_type="file",
                source=str(file_path),
                content="",
                success=False,
                error=f"File not found: {file_path}",
            )

        if not file_path.is_file():
            return ContextItem(
                source_type="file",
                source=str(file_path),
                content="",
                success=False,
                error=f"Not a file: {file_path}",
            )

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            return ContextItem(
                source_type="file",
                source=str(file_path),
                content="",
                success=False,
                error=f"File too large: {file_size} bytes (max {self.max_file_size})",
            )

        # Check extension if restricted
        if self.allowed_extensions:
            ext = file_path.suffix.lower()
            if ext not in self.allowed_extensions:
                return ContextItem(
                    source_type="file",
                    source=str(file_path),
                    content="",
                    success=False,
                    error=f"File extension not allowed: {ext}",
                )

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return ContextItem(
                source_type="file",
                source=str(file_path),
                content=content,
                metadata={
                    "size": file_size,
                    "extension": file_path.suffix,
                    "name": file_path.name,
                },
            )
        except Exception as e:
            return ContextItem(
                source_type="file",
                source=str(file_path),
                content="",
                success=False,
                error=f"Failed to read file: {e}",
            )


class FolderResolver(ContextResolver):
    """Resolves @folder:<path> commands by listing folder structure."""

    def __init__(
        self,
        max_depth: int = 3,
        max_files: int = 100,
        include_file_previews: bool = True,
        preview_lines: int = 10,
        ignore_patterns: Optional[List[str]] = None,
    ) -> None:
        self.max_depth = max_depth
        self.max_files = max_files
        self.include_file_previews = include_file_previews
        self.preview_lines = preview_lines
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            ".pytest_cache",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
        ]

    @property
    def command_type(self) -> str:
        return "folder"

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        name = path.name
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True
        return False

    def _build_tree(
        self, path: Path, prefix: str = "", depth: int = 0, file_count: List[int] = None
    ) -> str:
        """Build a tree representation of the folder."""
        if file_count is None:
            file_count = [0]

        if depth > self.max_depth or file_count[0] >= self.max_files:
            return ""

        lines = []

        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return f"{prefix}[Permission denied]\n"

        for i, entry in enumerate(entries):
            if self._should_ignore(entry):
                continue

            if file_count[0] >= self.max_files:
                lines.append(f"{prefix}... (truncated, {self.max_files} files limit)\n")
                break

            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/\n")
                subtree = self._build_tree(entry, prefix + extension, depth + 1, file_count)
                lines.append(subtree)
            else:
                file_count[0] += 1
                size = entry.stat().st_size
                lines.append(f"{prefix}{connector}{entry.name} ({size} bytes)\n")

        return "".join(lines)

    async def resolve(self, argument: str, context: Dict[str, Any]) -> ContextItem:
        working_dir = Path(context.get("working_directory", "."))
        folder_path_str = argument.strip()

        # Resolve path
        if folder_path_str.startswith("/"):
            folder_path = Path(folder_path_str)
        else:
            folder_path = working_dir / folder_path_str

        folder_path = folder_path.resolve()

        if not folder_path.exists():
            return ContextItem(
                source_type="folder",
                source=str(folder_path),
                content="",
                success=False,
                error=f"Folder not found: {folder_path}",
            )

        if not folder_path.is_dir():
            return ContextItem(
                source_type="folder",
                source=str(folder_path),
                content="",
                success=False,
                error=f"Not a folder: {folder_path}",
            )

        try:
            tree = f"{folder_path.name}/\n"
            tree += self._build_tree(folder_path)

            # Add key file previews if enabled
            previews = ""
            if self.include_file_previews:
                key_files = [
                    "README.md",
                    "README.rst",
                    "pyproject.toml",
                    "package.json",
                ]
                for key_file in key_files:
                    key_path = folder_path / key_file
                    if key_path.exists() and key_path.is_file():
                        try:
                            content = key_path.read_text(encoding="utf-8", errors="replace")
                            lines = content.split("\n")[: self.preview_lines]
                            preview = "\n".join(lines)
                            if len(content.split("\n")) > self.preview_lines:
                                preview += f"\n... ({len(content.split(chr(10)))} total lines)"
                            previews += f"\n\n--- {key_file} ---\n{preview}"
                        except Exception:
                            pass

            content = f"Folder structure:\n{tree}"
            if previews:
                content += f"\n\nKey files:{previews}"

            return ContextItem(
                source_type="folder",
                source=str(folder_path),
                content=content,
                metadata={
                    "path": str(folder_path),
                    "has_previews": bool(previews),
                },
            )

        except Exception as e:
            return ContextItem(
                source_type="folder",
                source=str(folder_path),
                content="",
                success=False,
                error=f"Failed to read folder: {e}",
            )


class ProblemsResolver(ContextResolver):
    """Resolves @problems command to include workspace diagnostics."""

    @property
    def command_type(self) -> str:
        return "problems"

    async def resolve(self, argument: str, context: Dict[str, Any]) -> ContextItem:
        # Get diagnostics from LSP or workspace
        diagnostics = context.get("diagnostics", [])
        workspace_errors = context.get("workspace_errors", [])

        if not diagnostics and not workspace_errors:
            return ContextItem(
                source_type="problems",
                source="workspace",
                content="No problems found in workspace.",
                metadata={"count": 0},
            )

        lines = ["Workspace Problems:\n"]

        # Format diagnostics
        for diag in diagnostics:
            severity = diag.get("severity", "info")
            file = diag.get("file", "unknown")
            line = diag.get("line", 0)
            message = diag.get("message", "")
            lines.append(f"[{severity.upper()}] {file}:{line}: {message}")

        # Format workspace errors
        for error in workspace_errors:
            lines.append(f"[ERROR] {error}")

        content = "\n".join(lines)
        return ContextItem(
            source_type="problems",
            source="workspace",
            content=content,
            metadata={
                "count": len(diagnostics) + len(workspace_errors),
                "diagnostics": len(diagnostics),
                "errors": len(workspace_errors),
            },
        )


class ContextCommandParser:
    """Parses and resolves @-prefixed context commands in user input."""

    # Pattern to match @command:argument or @command (for commands without args)
    COMMAND_PATTERN = re.compile(
        r"@(url|file|folder|problems)(?::(\S+))?",
        re.IGNORECASE,
    )

    def __init__(self, resolvers: Optional[Dict[str, ContextResolver]] = None) -> None:
        """
        Initialize the parser with resolvers.

        Args:
            resolvers: Map of command type to resolver. If None, uses defaults.
        """
        self._resolvers: Dict[str, ContextResolver] = {}

        if resolvers:
            self._resolvers = resolvers
        else:
            # Register default resolvers
            self.register_resolver(URLResolver())
            self.register_resolver(FileResolver())
            self.register_resolver(FolderResolver())
            self.register_resolver(ProblemsResolver())

    def register_resolver(self, resolver: ContextResolver) -> None:
        """Register a context resolver."""
        self._resolvers[resolver.command_type.lower()] = resolver

    def parse(self, text: str) -> List[ParsedCommand]:
        """
        Parse @-commands from text.

        Args:
            text: User input text

        Returns:
            List of parsed commands
        """
        commands = []
        for match in self.COMMAND_PATTERN.finditer(text):
            command_type = match.group(1).lower()
            argument = match.group(2) or ""

            commands.append(
                ParsedCommand(
                    command_type=command_type,
                    argument=argument,
                    original_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        return commands

    def strip_commands(self, text: str) -> str:
        """
        Remove @-commands from text, returning clean user message.

        Args:
            text: User input with commands

        Returns:
            Text with commands removed
        """
        return self.COMMAND_PATTERN.sub("", text).strip()

    async def resolve(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, List[ContextItem]]:
        """
        Parse and resolve all @-commands in text.

        Args:
            text: User input with commands
            context: Additional context (working_directory, etc.)

        Returns:
            Tuple of (cleaned_text, list of resolved context items)
        """
        context = context or {}
        commands = self.parse(text)
        items = []

        for cmd in commands:
            resolver = self._resolvers.get(cmd.command_type)
            if resolver:
                try:
                    item = await resolver.resolve(cmd.argument, context)
                    items.append(item)
                except Exception as e:
                    logger.error(f"Failed to resolve {cmd.command_type}: {e}")
                    items.append(
                        ContextItem(
                            source_type=cmd.command_type,
                            source=cmd.argument,
                            content="",
                            success=False,
                            error=str(e),
                        )
                    )
            else:
                logger.warning(f"No resolver for command type: {cmd.command_type}")

        cleaned_text = self.strip_commands(text)
        return cleaned_text, items


# Factory function
def create_default_parser() -> ContextCommandParser:
    """Create a parser with default resolvers."""
    return ContextCommandParser()


# Convenience function
async def resolve_context_commands(
    text: str, working_directory: str = "."
) -> tuple[str, List[ContextItem]]:
    """
    Convenience function to parse and resolve context commands.

    Args:
        text: User input with @-commands
        working_directory: Working directory for file resolution

    Returns:
        Tuple of (cleaned_text, list of resolved context items)
    """
    parser = create_default_parser()
    return await parser.resolve(text, {"working_directory": working_directory})
