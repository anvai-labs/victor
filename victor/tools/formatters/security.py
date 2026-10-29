"""Security scan results formatter with Rich markup."""

from typing import Dict, Any

from .base import ToolFormatter, FormattedOutput


class SecurityFormatter(ToolFormatter):
    """Format security scan results with Rich markup.

    Produces color-coded output for:
    - Severity levels (red=critical, orange=high, yellow=medium, blue=low)
    - Vulnerability details with CVE IDs
    - Affected files and line numbers
    - Remediation suggestions
    """

    def validate_input(self, data: Dict) -> bool:
        """Validate security scan result has required fields."""
        return isinstance(data, dict) and (
            "vulnerabilities" in data or "findings" in data or "issues" in data or "summary" in data
        )

    def format(self, data: Dict[str, Any], max_findings: int = 20, **kwargs) -> FormattedOutput:
        """Format security scan results with Rich markup.

        Args:
            data: Security scan result dict with vulnerabilities/findings/issues
            max_findings: Maximum findings to display (default: 20)

        Returns:
            FormattedOutput with Rich markup
        """
        # Support multiple result structures
        findings = data.get("vulnerabilities") or data.get("findings") or data.get("issues") or []

        summary = data.get("summary", {})

        lines = []

        if not findings:
            lines.append("[green bold]✓ No security issues found[/]")
            return FormattedOutput(
                content="\n".join(lines),
                format_type="rich",
                summary="No issues",
                line_count=1,
                contains_markup=True,
            )

        # Summary header with severity breakdown
        total = len(findings)
        critical = summary.get("critical", 0)
        high = summary.get("high", 0)
        medium = summary.get("medium", 0)
        low = summary.get("low", 0)

        summary_parts = []
        if critical > 0:
            summary_parts.append(f"[red bold]{critical} critical[/]")
        if high > 0:
            summary_parts.append(f"[orange1]{high} high[/]")
        if medium > 0:
            summary_parts.append(f"[yellow]{medium} medium[/]")
        if low > 0:
            summary_parts.append(f"[blue]{low} low[/]")

        if summary_parts:
            lines.append("[bold]Security Scan Results:[/]")
            lines.append(" ".join(summary_parts) + f" [dim]• {total} total[/]")
            lines.append("")

        # Display findings
        lines.append("[bold]Findings:[/]")
        lines.append("")

        for i, finding in enumerate(list(findings)[:max_findings], 1):
            severity = finding.get("severity", "unknown").lower()
            title = finding.get("title", finding.get("description", "Unknown issue"))
            cve_id = finding.get("cve_id", "")
            file_path = finding.get("file", finding.get("path", ""))
            line_num = finding.get("line", "")

            # Color-code severity
            if severity == "critical":
                severity_color = "red bold"
                icon = "‼"
            elif severity == "high":
                severity_color = "orange1 bold"
                icon = "⚠"
            elif severity == "medium":
                severity_color = "yellow bold"
                icon = "⚡"
            elif severity == "low":
                severity_color = "blue bold"
                icon = "⚐"
            else:
                severity_color = "white bold"
                icon = "•"

            # Finding header
            lines.append(
                f"  [{severity_color}]{icon}[/] [{severity_color}]{severity.upper()}[/] [bold]{title}[/]"
            )

            # CVE ID if available
            if cve_id:
                lines.append(f"    [cyan]{cve_id}[/]")

            # File location
            if file_path:
                if line_num:
                    lines.append(f"    [dim]{file_path}:{line_num}[/]")
                else:
                    lines.append(f"    [dim]{file_path}[/]")

            # Remediation if available
            remediation = finding.get("remediation", finding.get("fix", ""))
            if remediation:
                lines.append(f"    [dim]Fix: {remediation}[/]")

            # Blank line between findings (except after last one)
            if i < len(findings) and i < max_findings:
                lines.append("")

        # Add indicator if truncated
        if len(findings) > max_findings:
            lines.append("")
            lines.append(f"[dim]... and {len(findings) - max_findings} more findings[/]")

        content = "\n".join(lines)

        return FormattedOutput(
            content=content,
            format_type="rich",
            summary=f"{total} findings",
            line_count=len(lines),
            contains_markup=True,
        )
