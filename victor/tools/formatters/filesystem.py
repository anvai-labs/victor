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

"""File system operations formatter for Rich console output."""

from typing import Any, Dict

from .base import ToolFormatter, FormattedOutput


class FileSystemFormatter(ToolFormatter):
    """Formatter for file system operations (ls, find, cat, overview).

    Provides color-coded output for:
    - Directories (bold blue)
    - Files (regular)
    - Executable files (green)
    - Hidden files (dim)
    - Symbolic links (cyan)
    - File sizes and permissions
    """

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate file system operation data.

        Args:
            data: File system operation result

        Returns:
            True if data is valid, False otherwise
        """
        return isinstance(data, dict) and (
            "files" in data
            or "directories" in data
            or "path" in data
            or "content" in data
            or "entries" in data
            or "matches" in data
            or "results" in data
        )

    def format(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format file system operation results with Rich markup.

        Args:
            data: File system operation result
            **kwargs: Additional options (max_items, show_hidden, etc.)

        Returns:
            FormattedOutput with Rich markup
        """
        max_items = kwargs.get("max_items", 50)
        show_hidden = kwargs.get("show_hidden", False)

        lines = []

        # Handle directory listing (ls, overview)
        if "entries" in data or "files" in data or "directories" in data:
            entries = data.get("entries", data.get("files", []))
            dirs = data.get("directories", [])

            # Show directories first
            if dirs:
                lines.append(f"[bold blue]Directories ({len(dirs)}):[/]")
                for i, d in enumerate(dirs[:max_items]):
                    if not show_hidden and d.startswith("."):
                        continue
                    lines.append(f"  [bold blue]{d}/[/]")
                if len(dirs) > max_items:
                    lines.append(f"  [dim]... and {len(dirs) - max_items} more directories[/]")
                lines.append("")

            # Show files
            if entries:
                lines.append(f"[bold]Files ({len(entries)}):[/]")
                for i, f in enumerate(entries[:max_items]):
                    if not show_hidden and f.startswith("."):
                        continue

                    # Color-code based on file type
                    if f.endswith((".py", ".js", ".ts", ".java", ".go", ".rs")):
                        color = "green"  # Source code
                    elif f.endswith((".md", ".txt", ".json", ".yaml", ".yml")):
                        color = "cyan"  # Text/data
                    elif f.endswith((".sh", ".bash", ".zsh")):
                        color = "yellow"  # Scripts
                    else:
                        color = "white"  # Regular files

                    lines.append(f"  [{color}]{f}[/]")

                if len(entries) > max_items:
                    lines.append(f"  [dim]... and {len(entries) - max_items} more files[/]")

        # Handle file content (cat, read)
        elif "content" in data or "text" in data:
            content = data.get("content", data.get("text", ""))
            path = data.get("path", "")

            if path:
                lines.append(f"[bold cyan]{path}[/]")
                lines.append("")

            # Truncate long content
            max_lines = kwargs.get("max_lines", 100)
            content_lines = content.splitlines()

            if len(content_lines) > max_lines:
                lines.extend(content_lines[:max_lines])
                lines.append("")
                lines.append(f"[dim]... ({len(content_lines) - max_lines} more lines)[/]")
            else:
                lines.append(content)

        # Handle find results
        elif "matches" in data or "results" in data:
            matches = data.get("matches", data.get("results", []))
            path = data.get("path", ".")

            if path:
                lines.append(f"[bold]Search results in: {path}[/]")
                lines.append("")

            for i, match in enumerate(matches[:max_items]):
                match_path = match.get("path", match) if isinstance(match, dict) else match
                lines.append(f"  [green]✓[/] {match_path}")

            if len(matches) > max_items:
                lines.append(f"  [dim]... and {len(matches) - max_items} more matches[/]")

            lines.append("")
            lines.append(f"[dim]Total: {len(matches)} matches[/]")

        # Handle overview/project structure
        elif "structure" in data or "tree" in data:
            structure = data.get("structure", data.get("tree", ""))
            lines.append("[bold]Project Structure:[/]")
            lines.append("")
            lines.append(structure)

        # Fallback for other formats
        else:
            lines.append(str(data))

        content = "\n".join(lines)
        summary = self._extract_summary(data)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=summary,
            contains_markup=True,
        )

    def _extract_summary(self, data: Dict[str, Any]) -> str:
        """Extract summary from file system data.

        Args:
            data: File system operation result

        Returns:
            Summary string
        """
        if "entries" in data or "files" in data:
            count = len(data.get("entries", data.get("files", [])))
            return f"{count} items"

        if "content" in data:
            path = data.get("path", "file")
            return f"{path}"

        if "matches" in data or "results" in data:
            count = len(data.get("matches", data.get("results", [])))
            return f"{count} matches"

        return "File system operation"

    def get_fallback(self) -> "ToolFormatter":
        """Return fallback formatter.

        Returns:
            GenericFormatter instance
        """
        from .generic import GenericFormatter

        return GenericFormatter()
