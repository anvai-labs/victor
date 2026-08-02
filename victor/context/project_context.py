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

"""Project-context loader with shared instruction discovery.

Victor prefers native instruction files such as `.victor/init.md`, but it also
imports compatibility surfaces like `AGENTS.md` and `CLAUDE.md` through the
shared discovery module. The current workspace directory is walked upward
toward the filesystem root, nearer files are treated as more specific, and
Victor-native files take precedence within a single directory.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.config.settings import (
    VICTOR_DIR_NAME,
    VICTOR_CONTEXT_FILE,
    get_project_paths,
)
from victor.context.instruction_discovery import (
    InstructionFile,
    build_instruction_signature,
    discover_instruction_files,
)

# Cache for ProjectContext instances and their content
# Key: (root_path, instruction signature)
_context_cache: Dict[
    Tuple[str, Tuple[Tuple[str, float, int], ...]],
    Tuple[Optional[Path], Tuple[InstructionFile, ...], str, str, Dict[str, str]],
] = {}
_cache_lock = threading.Lock()

logger = logging.getLogger(__name__)

# Context file location: .victor/init.md (configurable via settings.py)
# No legacy locations - clean integration for future code
CONTEXT_FILE_PATH = f"{VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE}"


class ProjectContext:
    """Loads and manages project-specific context from discovered instruction files."""

    def __init__(self, root_path: Optional[str] = None):
        """Initialize project context loader.

        Args:
            root_path: Root directory to search for context files.
                      Defaults to current working directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._context_file: Optional[Path] = None
        self._content: Optional[str] = None
        self._primary_content: Optional[str] = None
        self._parsed_sections: Dict[str, str] = {}
        self._instruction_files: List[InstructionFile] = []

    def find_context_file(self) -> Optional[Path]:
        """Find the canonical primary instruction file for this workspace.

        Returns:
            Path to context file if found, None otherwise.
        """
        discovered = discover_instruction_files(self.root_path)
        if not discovered:
            return None
        return discovered[0].path

    def load(self, force_reload: bool = False) -> bool:
        """Load project context from discovered instruction files with signature caching.

        Args:
            force_reload: Force re-reading files even if the current signature is cached.

        Returns:
            True if context was loaded successfully, False otherwise.
        """
        instruction_files = discover_instruction_files(self.root_path)
        if not instruction_files:
            logger.debug("No project context file found")
            return False

        self._context_file = instruction_files[0].path

        try:
            root_key = str(self.root_path)
            signature = build_instruction_signature(instruction_files)
            cache_key = (root_key, signature)

            with _cache_lock:
                if not force_reload and cache_key in _context_cache:
                    (
                        self._context_file,
                        cached_files,
                        self._primary_content,
                        self._content,
                        self._parsed_sections,
                    ) = _context_cache[cache_key]
                    self._instruction_files = list(cached_files)
                    logger.debug("Using cached project context for %s", self._context_file)
                    return True

                self._instruction_files = list(instruction_files)
                self._primary_content = self._instruction_files[0].content
                self._content = self._join_instruction_content(self._instruction_files)
                self._parse_sections(self._primary_content)

                # Update cache (clear old entries for same root)
                keys_to_remove = [k for k in _context_cache if k[0] == root_key]
                for k in keys_to_remove:
                    del _context_cache[k]
                _context_cache[cache_key] = (
                    self._context_file,
                    tuple(self._instruction_files),
                    self._primary_content,
                    self._content,
                    self._parsed_sections.copy(),
                )

            logger.info(f"Loaded project context from {self._context_file}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load project context: {e}")
            return False

    @staticmethod
    def clear_cache() -> None:
        """Clear the project context cache (for testing or hot-reload)."""
        with _cache_lock:
            _context_cache.clear()
        logger.debug("Cleared project context cache")

    @staticmethod
    def _join_instruction_content(instruction_files: List[InstructionFile]) -> str:
        """Join discovered instruction content for compatibility consumers."""
        return "\n\n".join(
            file.content.strip() for file in instruction_files if file.content.strip()
        ).strip()

    def _parse_sections(self, content: Optional[str] = None) -> None:
        """Parse markdown content into sections by headers."""
        source = content if content is not None else self._content
        if not source:
            return

        self._parsed_sections = {}
        current_section = "overview"
        current_content: List[str] = []

        for line in source.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_content:
                    self._parsed_sections[current_section] = "\n".join(current_content).strip()
                # Start new section
                current_section = line[3:].strip().lower().replace(" ", "_")
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            self._parsed_sections[current_section] = "\n".join(current_content).strip()

    @property
    def content(self) -> str:
        """Get the full context content."""
        return self._content or ""

    @property
    def context_file(self) -> Optional[Path]:
        """Get the path to the loaded context file."""
        return self._context_file

    @property
    def instruction_files(self) -> List[InstructionFile]:
        """Get the ordered instruction files used for this project context."""
        return list(self._instruction_files)

    def get_context_signature(self) -> Tuple[Tuple[str, float, int], ...]:
        """Return a live signature for prompt invalidation and cache refresh."""
        live_signature: List[Tuple[str, float, int]] = []
        for item in self._instruction_files:
            try:
                stat = item.path.stat()
                live_signature.append((str(item.path), stat.st_mtime, stat.st_size))
            except OSError:
                live_signature.append((str(item.path), item.mtime, item.size))
        return tuple(live_signature)

    def get_section(self, section_name: str) -> str:
        """Get a specific section from the context.

        Args:
            section_name: Name of the section (case-insensitive, spaces to underscores)

        Returns:
            Section content, or empty string if not found.
        """
        key = section_name.lower().replace(" ", "_")
        return self._parsed_sections.get(key, "")

    def get_system_prompt_addition(self) -> str:
        """Get context formatted for inclusion in system prompt.

        Returns:
            Formatted context string for system prompt injection.
        """
        if not self._content:
            return ""

        loaded_files = "\n".join(
            f"- {self._display_path(item.path)} ({item.scope}, source={item.source_type})"
            for item in self._instruction_files
        )
        rendered_files = "\n\n".join(
            (
                f"## {self._display_path(item.path)}"
                f" (scope: {item.scope}, source: {item.source_type})\n{item.content}"
            )
            for item in self._instruction_files
        )

        return f"""
<project-context>
Victor loaded project instructions by walking upward from the current working directory.
Nearer files are more specific. Within the same directory, Victor-native files load before
AGENTS/CLAUDE compatibility files.

Loaded instruction files:
{loaded_files}

{rendered_files}
</project-context>
"""

    def _display_path(self, path: Path) -> str:
        """Render instruction paths relative to the workspace when possible."""
        try:
            return str(path.relative_to(self.root_path.resolve()))
        except ValueError:
            return str(path)

    @classmethod
    def auto_generate(cls, root_path: str) -> Optional[Path]:
        """Auto-generate a compact init.md for a project if not present.

        Creates a lightweight (~500 token) context file from:
        - README.md first 1500 chars
        - Top-level directory listing (max 20)
        - Detected build system

        Idempotent: skips if init.md already exists.

        Args:
            root_path: Project root directory

        Returns:
            Path to created init.md, or None if already exists or failed
        """
        root = Path(root_path)
        victor_dir = root / VICTOR_DIR_NAME
        init_file = victor_dir / VICTOR_CONTEXT_FILE

        if init_file.exists():
            logger.debug(f"init.md already exists at {init_file}")
            return None

        sections = []
        project_name = root.name

        # 1. Extract README summary (first 1500 chars of first text paragraph)
        readme_summary = ""
        for readme_name in ["README.md", "README.rst", "README.txt", "readme.md"]:
            readme_path = root / readme_name
            if readme_path.exists():
                try:
                    raw = readme_path.read_text(encoding="utf-8")[:3000]
                    for para in raw.split("\n\n"):
                        stripped = para.strip()
                        if not stripped:
                            continue
                        if stripped.startswith(("#", "![", "<", "[!", "---", "```")):
                            continue
                        if stripped.startswith("[") and stripped.endswith(")"):
                            continue
                        readme_summary = stripped[:1500]
                        break
                except Exception:
                    pass
                break

        sections.append(f"# {project_name}\n")
        if readme_summary:
            sections.append(readme_summary + "\n")

        # 2. Top-level directory listing
        try:
            dirs = sorted(
                d.name
                for d in root.iterdir()
                if d.is_dir() and not d.name.startswith((".", "_", "venv", "env", "node_modules"))
            )[:20]
            if dirs:
                sections.append("## Structure\n")
                sections.append("```")
                sections.append("\n".join(f"{d}/" for d in dirs))
                sections.append("```\n")
        except OSError:
            pass

        # 3. Detect build system
        build_hints = []
        if (root / "pyproject.toml").exists():
            build_hints.append("Python (pyproject.toml)")
        elif (root / "setup.py").exists():
            build_hints.append("Python (setup.py)")
        if (root / "package.json").exists():
            build_hints.append("Node.js")
        if (root / "Cargo.toml").exists():
            build_hints.append("Rust")
        if (root / "go.mod").exists():
            build_hints.append("Go")

        if build_hints:
            sections.append(f"**Build**: {', '.join(build_hints)}\n")

        content = "\n".join(sections)

        # Write file
        try:
            victor_dir.mkdir(parents=True, exist_ok=True)
            init_file.write_text(content, encoding="utf-8")
            logger.info(f"Auto-generated init.md at {init_file} ({len(content)} chars)")
            return init_file
        except Exception as e:
            logger.warning(f"Failed to auto-generate init.md: {e}")
            return None

    def get_package_layout_hint(self) -> str:
        """Extract package layout hints from context.

        Returns:
            Package layout guidance if specified in context.
        """
        # Look for package layout or directory structure sections
        for section_name in [
            "package_layout",
            "directory_structure",
            "project_structure",
            "architecture",
        ]:
            content = self.get_section(section_name)
            if content:
                return content
        return ""


def generate_victor_md(root_path: Optional[str] = None) -> str:
    """Generate project context file by analyzing the codebase.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.

    Returns:
        Generated markdown content for init.md
    """
    root = Path(root_path) if root_path else Path.cwd()

    sections = []

    # Header - use configurable file name
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    project_name = root.name
    sections.append("## Project Overview\n")

    # Try to detect project type and description
    readme_files = ["README.md", "README.rst", "README.txt", "readme.md"]
    readme_content = ""
    for readme in readme_files:
        readme_path = root / readme
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8")
                # Extract first meaningful paragraph (skip headers, images, HTML, badges)
                lines = content.split("\n\n")
                for para in lines:
                    stripped = para.strip()
                    # Skip empty, headers, images, HTML tags, badges, links-only lines
                    if not stripped:
                        continue
                    if stripped.startswith(("#", "![", "<", "[!", "---", "```")):
                        continue
                    if stripped.startswith("[") and stripped.endswith(")"):
                        continue  # Skip badge/link lines like [License](url)
                    # Found a text paragraph
                    readme_content = stripped[:500]
                    break
            except Exception:
                pass
            break

    if readme_content:
        sections.append(f"**{project_name}**: {readme_content}\n")
    else:
        sections.append(f"**{project_name}**: [Add project description here]\n")

    # Detect package layout
    sections.append("## Package Layout\n")
    layout_hints = []

    # Find Python package directories at root level (directories with __init__.py)
    def is_python_package(path: Path) -> bool:
        return path.is_dir() and (path / "__init__.py").exists()

    root_packages = [
        d.name
        for d in root.iterdir()
        if is_python_package(d) and not d.name.startswith((".", "_", "venv", "test"))
    ]

    # Check for common Python project structures
    has_src = (root / "src").is_dir()
    expected_pkg = project_name.replace("-", "_")

    if root_packages:
        # Found Python packages at root level - these are likely the active code
        main_pkg = root_packages[0]  # Take first package found
        layout_hints.append(f"- **Active code**: `{main_pkg}/` (main package)")
        if has_src:
            layout_hints.append("- **Legacy/Deprecated**: `src/` (DO NOT USE)")
    elif has_src:
        # Only src/ exists, check if it has nested packages
        src_packages = [d.name for d in (root / "src").iterdir() if is_python_package(d)]
        if src_packages:
            layout_hints.append(f"- **Source code**: `src/{src_packages[0]}/` (src layout)")
        else:
            layout_hints.append("- **Source code**: `src/` (src layout)")
    elif (root / expected_pkg).is_dir():
        layout_hints.append(f"- **Main package**: `{expected_pkg}/`")

    # Check for tests
    if (root / "tests").is_dir():
        layout_hints.append("- **Tests**: `tests/`")
    elif (root / "test").is_dir():
        layout_hints.append("- **Tests**: `test/`")

    # Check for docs
    if (root / "docs").is_dir():
        layout_hints.append("- **Documentation**: `docs/`")

    if layout_hints:
        sections.append("\n".join(layout_hints) + "\n")
    else:
        sections.append("[Add package layout description here]\n")

    # Common Commands
    sections.append("## Common Commands\n")
    commands = []

    # Detect build system
    if (root / "pyproject.toml").exists():
        commands.append("```bash")
        commands.append("# Install with dev dependencies")
        commands.append('pip install -e ".[dev]"')
        commands.append("")
        commands.append("# Run tests")
        commands.append("pytest")
        commands.append("")
        commands.append("# Format code")
        commands.append("black .")
        commands.append("```")
    elif (root / "setup.py").exists():
        commands.append("```bash")
        commands.append("pip install -e .")
        commands.append("pytest")
        commands.append("```")
    elif (root / "package.json").exists():
        commands.append("```bash")
        commands.append("npm install")
        commands.append("npm test")
        commands.append("npm run build")
        commands.append("```")
    elif (root / "Cargo.toml").exists():
        commands.append("```bash")
        commands.append("cargo build")
        commands.append("cargo test")
        commands.append("```")
    elif (root / "go.mod").exists():
        commands.append("```bash")
        commands.append("go build ./...")
        commands.append("go test ./...")
        commands.append("```")
    else:
        commands.append("[Add common commands here]")

    sections.append("\n".join(commands) + "\n")

    # Architecture (placeholder)
    sections.append("## Architecture\n")
    sections.append("[Add architecture overview here]\n")

    # Important Notes
    sections.append("## Important Notes\n")
    sections.append("- [Add project-specific notes and conventions here]\n")

    return "\n".join(sections)


def init_victor_md(root_path: Optional[str] = None, force: bool = False) -> Optional[Path]:
    """Initialize project context file in .victor/init.md.

    Creates the file at the configured location (default: .victor/init.md).
    Location is configurable via settings.py (VICTOR_DIR_NAME, VICTOR_CONTEXT_FILE).

    Args:
        root_path: Root directory to create file in. Defaults to current directory.
        force: If True, overwrite existing file.

    Returns:
        Path to created file, or None if file exists and force=False.
    """
    root = Path(root_path) if root_path else Path.cwd()

    # Use settings-driven path
    paths = get_project_paths(root)
    target_file = paths.project_context_file

    # Ensure .victor directory exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if target_file.exists() and not force:
        logger.warning(f"{VICTOR_CONTEXT_FILE} already exists at {target_file}")
        return None

    content = generate_victor_md(root_path)

    try:
        target_file.write_text(content, encoding="utf-8")
        logger.info(f"Created {VICTOR_CONTEXT_FILE} at {target_file}")
        return target_file
    except Exception as e:
        logger.error(f"Failed to create {VICTOR_CONTEXT_FILE}: {e}")
        return None


async def generate_victor_md_with_graph(
    root_path: Optional[str] = None,
    task: Optional[str] = None,
    max_symbols: Optional[int] = None,
) -> str:
    """Generate project context file with graph-based enhancements.

    This async version includes graph-aware context when available:
    - Relevant symbols from graph traversal
    - Symbol dependencies
    - Data flow information
    - Similar code patterns

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        task: Optional task description for context relevance. If None,
              uses a generic "understand this codebase" task.
        max_symbols: Maximum number of symbols to include in graph context.
                     If None, uses settings.graph.init_max_symbols.

    Returns:
        Generated markdown content for init.md with graph enhancements
    """
    from victor.config.settings import get_settings

    root = Path(root_path) if root_path else Path.cwd()
    settings = get_settings()

    # Start with standard content
    content = generate_victor_md(root_path)
    sections = [content]

    # Check if graph-enhanced context is enabled
    graph_enabled = getattr(settings, "graph", None)
    if graph_enabled is None:
        logger.debug("Graph settings not available, skipping graph context")
        return content

    if not getattr(graph_enabled, "enable_graph_rag", False):
        logger.debug("Graph RAG not enabled, skipping graph context")
        return content

    if not getattr(graph_enabled, "enable_graph_context_in_init", True):
        logger.debug("Graph context in init.md disabled via settings")
        return content

    # Use setting for max_symbols if not provided
    if max_symbols is None:
        max_symbols = getattr(graph_enabled, "init_max_symbols", 50)

    # Check feature flag
    try:
        from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

        if not get_feature_flag_manager().is_enabled(FeatureFlag.USE_GRAPH_ENHANCED_CONTEXT):
            logger.debug("USE_GRAPH_ENHANCED_CONTEXT feature flag disabled")
            return content
    except Exception:
        logger.debug("Feature flag check failed, proceeding with graph context")

    # Try to add graph context
    try:
        from victor.storage.graph import create_graph_store

        # Get graph store path from settings
        graph_path = getattr(graph_enabled, "codebase_graph_path", None) or str(
            root / ".victor" / "graph.db"
        )

        graph_store = create_graph_store("sqlite", graph_path, root)
        await graph_store.initialize()

        # Check if graph has data
        stats = await graph_store.stats()
        if stats.get("node_count", 0) == 0:
            logger.info("Graph store is empty, skipping graph context")
            await graph_store.close()
            return content

        # Build graph context
        from victor.context.graph_context_builder import GraphEnhancedContextBuilder

        builder = GraphEnhancedContextBuilder(graph_store)

        # Use provided task or generic understanding task
        query_task = task or f"Understand the {root.name} codebase structure and key components"

        result = await builder.build_context(query_task, max_symbols)

        # Add graph context section
        if result.context and result.symbols_included:
            sections.append("\n## Code Graph Context\n")
            sections.append(
                f"*Graph analysis: {len(result.symbols_included)} relevant symbols, "
                f"{result.build_time_ms:.0f}ms build time*\n"
            )

            # Add symbols section
            if result.symbols_included:
                sections.append("### Key Symbols\n")
                for symbol in result.symbols_included[:20]:  # Limit display
                    sections.append(f"- `{symbol}`")
                sections.append("")

            # Add dependencies section
            if result.dependencies_found:
                sections.append("### Dependencies\n")
                count = 0
                for symbol_name, deps in result.dependencies_found.items():
                    if deps and count < 10:  # Limit display
                        sections.append(
                            f"- `{symbol_name}` → " + ", ".join(f"`{d}`" for d in deps[:3])
                        )
                        count += 1
                sections.append("")

            # Add data flow section if available
            if result.metadata.get("dependency_count", 0) > 0:
                sections.append(
                    f"*Total dependencies tracked: {result.metadata['dependency_count']}*\n"
                )

        await graph_store.close()

        logger.info(f"Generated graph-enhanced init.md with {len(result.symbols_included)} symbols")

    except ImportError as e:
        logger.debug(f"Graph dependencies not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to generate graph context: {e}")

    return "\n".join(sections)


async def init_victor_md_with_graph(
    root_path: Optional[str] = None,
    force: bool = False,
    task: Optional[str] = None,
    max_symbols: int = 50,
) -> Optional[Path]:
    """Initialize project context file with graph-based enhancements (async).

    Creates the file at the configured location (default: .victor/init.md)
    with graph-aware context included.

    Args:
        root_path: Root directory to create file in. Defaults to current directory.
        force: If True, overwrite existing file.
        task: Optional task description for context relevance.
        max_symbols: Maximum number of symbols to include in graph context.

    Returns:
        Path to created file, or None if file exists and force=False.
    """
    root = Path(root_path) if root_path else Path.cwd()

    # Use settings-driven path
    paths = get_project_paths(root)
    target_file = paths.project_context_file

    # Ensure .victor directory exists
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if target_file.exists() and not force:
        logger.warning(f"{VICTOR_CONTEXT_FILE} already exists at {target_file}")
        return None

    content = await generate_victor_md_with_graph(root_path, task, max_symbols)

    try:
        target_file.write_text(content, encoding="utf-8")
        logger.info(f"Created graph-enhanced {VICTOR_CONTEXT_FILE} at {target_file}")
        return target_file
    except Exception as e:
        logger.error(f"Failed to create {VICTOR_CONTEXT_FILE}: {e}")
        return None


__all__ = [
    "ProjectContext",
    "generate_victor_md",
    "init_victor_md",
    "generate_victor_md_with_graph",
    "init_victor_md_with_graph",
]
