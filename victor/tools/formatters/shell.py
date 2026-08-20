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

"""Shell command execution formatter for Rich console output."""

from typing import Any, Dict

from .base import ToolFormatter, FormattedOutput


class ShellFormatter(ToolFormatter):
    """Formatter for shell command execution (shell, bash, exec).

    Provides color-coded output for:
    - Command executed (bold yellow)
    - Exit status (✓ success, ✗ failure)
    - Duration (dim)
    - stdout/stderr separation
    - Error messages (red)
    """

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate shell command execution data.

        Args:
            data: Shell command execution result

        Returns:
            True if data is valid, False otherwise
        """
        return isinstance(data, dict) and (
            "success" in data
            or "exit_code" in data
            or "stdout" in data
            or "stderr" in data
            or "command" in data
        )

    def format(self, data: Dict[str, Any], **kwargs) -> FormattedOutput:
        """Format shell command execution results with Rich markup.

        Args:
            data: Shell command execution result
            **kwargs: Additional options (max_output_lines, show_command, etc.)

        Returns:
            FormattedOutput with Rich markup
        """
        max_output_lines = kwargs.get("max_output_lines", 100)
        show_command = kwargs.get("show_command", True)

        lines = []

        # Show command if available
        if show_command and "command" in data:
            command = data["command"]
            lines.append(f"[bold yellow]$ {command}[/]")
            lines.append("")

        # Show exit status
        success = data.get("success", True)
        exit_code = data.get("exit_code", 0)

        if success and exit_code == 0:
            lines.append("[green]✓ Command succeeded[/]")
        else:
            lines.append(f"[red]✗ Command failed (exit code: {exit_code})[/]")

        # Show duration if available
        if "duration_ms" in data or "duration" in data:
            duration = data.get("duration_ms", data.get("duration", 0))
            lines.append(f"[dim]Duration: {duration}ms[/]")

        lines.append("")

        # Show stdout
        if "stdout" in data:
            stdout = data["stdout"]
            if stdout:
                lines.append("[bold]Output:[/]")
                stdout_lines = stdout.splitlines()

                if len(stdout_lines) > max_output_lines:
                    lines.extend(stdout_lines[:max_output_lines])
                    lines.append(f"[dim]... ({len(stdout_lines) - max_output_lines} more lines)[/]")
                else:
                    lines.append(stdout)
                lines.append("")

        # Show stderr
        if "stderr" in data:
            stderr = data["stderr"]
            if stderr:
                lines.append("[red bold]Error Output:[/]")
                lines.append(stderr)
                lines.append("")

        # Show error message if available
        if "error" in data:
            error = data["error"]
            lines.append(f"[red]Error: {error}[/]")
            lines.append("")

        content = "\n".join(lines)
        summary = self._extract_summary(data)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=summary,
            contains_markup=True,
        )

    def _extract_summary(self, data: Dict[str, Any]) -> str:
        """Extract summary from shell command data.

        Args:
            data: Shell command execution result

        Returns:
            Summary string
        """
        command = data.get("command", "")
        success = data.get("success", True)
        exit_code = data.get("exit_code", 0)

        if command:
            status = "✓" if success and exit_code == 0 else "✗"
            # Truncate long commands
            if len(command) > 40:
                command = command[:40] + "..."
            return f"{status} {command}"

        if success:
            return "Command succeeded"
        else:
            return f"Command failed (exit {exit_code})"

    def get_fallback(self) -> "ToolFormatter":
        """Return fallback formatter.

        Returns:
            GenericFormatter instance
        """
        from .generic import GenericFormatter

        return GenericFormatter()
