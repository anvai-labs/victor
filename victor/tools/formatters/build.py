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

"""Build tool formatter for Rich console output."""

from typing import Any, Dict

from .base import ToolFormatter, FormattedOutput


class BuildFormatter(ToolFormatter):
    """Formatter for build tool operations (make, cmake, cargo, npm, pip).

    Provides color-coded output for:
    - Build status (✓ success, ✗ failed, ○ skipped)
    - Warnings (yellow)
    - Errors (red)
    - Compilation steps
    - Build duration
    - Artifacts produced
    """

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate build tool operation data.

        Args:
            data: Build tool operation result

        Returns:
            True if data is valid, False otherwise
        """
        return isinstance(data, dict) and (
            "operation" in data
            or "target" in data
            or "build" in data
            or "success" in data
            or "artifacts" in data
        )

    def format(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format build tool operation results with Rich markup.

        Args:
            data: Build tool operation result
            **kwargs: Additional options (max_errors, max_warnings, etc.)

        Returns:
            FormattedOutput with Rich markup
        """
        max_errors = kwargs.get("max_errors", 10)
        max_warnings = kwargs.get("max_warnings", 20)

        lines = []

        # Build tool type
        tool = data.get("tool", "build").lower()
        operation = data.get("operation", "build")
        target = data.get("target", "")

        # Header
        if target:
            lines.append(f"[bold cyan]{tool.upper()} {operation}[/] [dim]{target}[/]")
        else:
            lines.append(f"[bold cyan]{tool.upper()} {operation}[/]")
        lines.append("")

        # Status
        success = data.get("success", True)
        if success:
            lines.append("[green]✓ Build succeeded[/]")
        else:
            lines.append("[red]✗ Build failed[/]")

        # Duration
        if "duration_ms" in data or "duration" in data:
            duration = data.get("duration_ms", data.get("duration", 0))
            lines.append(f"[dim]Duration: {duration:.2f}s[/]")

        lines.append("")

        # Artifacts
        if "artifacts" in data:
            artifacts = data["artifacts"]
            lines.append(f"[bold]Artifacts ({len(artifacts)}):[/]")
            for artifact in artifacts[:10]:
                size = artifact.get("size", "")
                if size:
                    lines.append(f"  [green]✓[/] {artifact['name']} [dim]({size})[/]")
                else:
                    lines.append(f"  [green]✓[/] {artifact['name']}")
            if len(artifacts) > 10:
                lines.append(f"  [dim]... and {len(artifacts) - 10} more artifacts[/]")
            lines.append("")

        # Warnings
        warnings = data.get("warnings", [])
        if warnings:
            lines.append(f"[yellow]⚠ Warnings ({len(warnings)}):[/]")
            for i, warning in enumerate(warnings[:max_warnings]):
                # Extract file:line from warning if available
                warning_str = warning.get("message", str(warning))
                lines.append(f"  [yellow]{i+1}.[/] {warning_str}")
            if len(warnings) > max_warnings:
                lines.append(f"  [dim]... and {len(warnings) - max_warnings} more warnings[/]")
            lines.append("")

        # Errors
        errors = data.get("errors", [])
        if errors:
            lines.append(f"[red bold]Errors ({len(errors)}):[/]")
            for i, error in enumerate(errors[:max_errors]):
                # Extract file:line: error message
                if isinstance(error, dict):
                    file_path = error.get("file", error.get("path", ""))
                    line = error.get("line", "")
                    message = error.get("message", error.get("error", ""))
                    if file_path and line:
                        lines.append(f"  [red]{i+1}.[/] [bold]{file_path}:{line}[/]")
                        lines.append(f"      [red]{message}[/]")
                    else:
                        lines.append(f"  [red]{i+1}.[/] {message}")
                else:
                    lines.append(f"  [red]{i+1}.[/] {error}")
            if len(errors) > max_errors:
                lines.append(f"  [dim]... and {len(errors) - max_errors} more errors[/]")
            lines.append("")

        # Compilation steps
        if "steps" in data:
            steps = data["steps"]
            lines.append("[bold]Steps:[/]")
            for step in steps:
                step_name = step.get("name", step)
                step_success = step.get("success", True)
                step_duration = step.get("duration_ms", 0)

                if step_success:
                    lines.append(f"  [green]✓[/] {step_name} [dim]({step_duration:.2f}s)[/]")
                else:
                    lines.append(f"  [red]✗[/] {step_name} [dim]({step_duration:.2f}s)[/]")

        content = "\n".join(lines)
        summary = self._extract_summary(data)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=summary,
            contains_markup=True,
        )

    def _extract_summary(self, data: Dict[str, Any]) -> str:
        """Extract summary from build tool data.

        Args:
            data: Build tool operation result

        Returns:
            Summary string
        """
        tool = data.get("tool", "build").capitalize()
        operation = data.get("operation", "build")
        target = data.get("target", "")
        success = data.get("success", True)

        status = "✓" if success else "✗"

        if target:
            return f"{status} {tool} {operation} {target}"
        else:
            return f"{status} {tool} {operation}"

    def get_fallback(self) -> "ToolFormatter":
        """Return fallback formatter.

        Returns:
            GenericFormatter instance
        """
        from .generic import GenericFormatter

        return GenericFormatter()
