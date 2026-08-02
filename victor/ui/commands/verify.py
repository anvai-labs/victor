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

"""Codebase verification commands with semantic validation.

Provides commands for verifying codebase analysis results with
false positive detection, documentation cross-reference, and severity weighting.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from victor.tools.verification import (
    ClaimVerifier,
    ClaimIssue,
    DocumentationCrossReference,
    FalsePositiveDetector,
    SeverityLevel,
    SeverityWeighting,
    TemporalContextAnalyzer,
    TemporalNature,
)

app = typer.Typer(help="Codebase verification with semantic validation")
console = Console()


def _run_async(coro):
    """Run async function, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - create task
        import concurrent.futures
        import threading

        result = [None]
        error = [None]

        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                result[0] = new_loop.run_until_complete(coro)
            except Exception as e:
                error[0] = e
            finally:
                new_loop.close()

        thread = threading.Thread(target=run_in_new_loop, daemon=True)
        thread.start()
        thread.join(timeout=30)

        if error[0]:
            raise error[0]
        return result[0]
    except RuntimeError:
        # No running loop - use asyncio.run
        return asyncio.run(coro)


@app.command()
def issue(
    issue_type: str = typer.Argument(..., help="Type of issue to verify"),
    file_path: str = typer.Option(None, "--file", "-f", help="Path to file"),
    line_number: int = typer.Option(None, "--line", "-l", help="Line number"),
    description: str = typer.Option("", "--desc", "-d", help="Issue description"),
    show_evidence: bool = typer.Option(False, "--evidence", "-e", help="Show collected evidence"),
    show_fp_analysis: bool = typer.Option(False, "--fp", help="Show false positive analysis"),
    project_root: str = typer.Option(".", "--root", "-r", help="Project root directory"),
):
    """
    Verify a single codebase issue with semantic validation.

    Examples:
        victor verify issue cross_layer_dependency --file src/storage/lib.rs
        victor verify issue global_mutable_state --file tests/test.rs --fp
        victor verify issue security_vulnerability --desc "Buffer overflow" --evidence
    """

    async def _verify() -> None:
        root = Path(project_root).resolve()

        # Create issue
        issue = ClaimIssue(
            issue_type=issue_type,
            description=description or issue_type,
            file_path=file_path,
            line_number=line_number,
        )

        # Run verification
        verifier = ClaimVerifier(project_root=root)
        result = await verifier.verify_claim(issue)

        # Display results
        console.print(f"\n[bold]Verification Result:[/bold]")
        console.print(
            f"  Status: {'[green]✓ Verified[/green]' if result.is_grounded else '[red]✗ Not Verified[/red]'}"
        )
        console.print(f"  Confidence: {result.confidence:.1%}")
        console.print(f"  Reason: {result.reason}")

        if show_evidence and result.evidence:
            console.print(f"\n[bold]Evidence:[/bold]")
            for i, source in enumerate(result.evidence.get("sources", [])[:5], 1):
                console.print(f"  {i}. {source.get('source_file', 'N/A')}")
                if source.get("line_number"):
                    console.print(f"     Line {source['line_number']}")
                if source.get("snippet"):
                    snippet = (
                        source["snippet"][:60] + "..."
                        if len(source.get("snippet", "")) > 60
                        else source.get("snippet", "")
                    )
                    console.print(f"     {snippet}")

        # False positive analysis
        if show_fp_analysis:
            fp_detector = FalsePositiveDetector()
            is_fp, fp_reason, fp_conf = fp_detector.is_likely_false_positive(issue)

            console.print(f"\n[bold]False Positive Analysis:[/bold]")
            fp_status = (
                "[yellow]⚠ Likely False Positive[/yellow]"
                if is_fp
                else "[green]✓ Genuine Issue[/green]"
            )
            console.print(f"  Status: {fp_status}")
            console.print(f"  Confidence: {fp_conf:.1%}")
            if is_fp:
                console.print(f"  Reason: {fp_reason}")

        # Severity weighting
        weighting = SeverityWeighting()
        score, severity = weighting.score_and_classify(issue)
        console.print(f"\n[bold]Severity:[/bold]")
        severity_color = {
            SeverityLevel.CRITICAL: "red",
            SeverityLevel.HIGH: "orange3",
            SeverityLevel.MEDIUM: "yellow",
            SeverityLevel.LOW: "blue",
            SeverityLevel.INFO: "dim",
        }.get(severity, "white")
        console.print(f"  Level: [{severity_color}]{severity.value}[/{severity_color}]")
        console.print(f"  Score: {score:.2f}")

    return _run_async(_verify())


@app.command()
def batch(
    input_file: str = typer.Argument(..., help="JSON file with issues to verify"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file"),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", "-c", help="Minimum confidence threshold"
    ),
    project_root: str = typer.Option(".", "--root", "-r", help="Project root directory"),
):
    """
    Verify multiple issues from a JSON file.

    Input file format:
        [
            {
                "issue_type": "cross_layer_dependency",
                "description": "Storage depends on Index",
                "file_path": "src/storage/lib.rs"
            },
            ...
        ]

    Examples:
        victor verify batch issues.json
        victor verify batch issues.json --output verified.json --min-confidence 0.7
    """

    async def _verify_batch() -> None:
        root = Path(project_root).resolve()
        input_path = Path(input_file)

        if not input_path.exists():
            console.print(f"[red]Error:[/red] Input file not found: {input_file}")
            raise typer.Exit(1)

        # Load issues
        issues = json.loads(input_path.read_text())
        console.print(f"Loaded {len(issues)} issues from {input_file}")

        # Verify each issue
        results = []
        for i, issue_dict in enumerate(issues, 1):
            console.print(
                f"[{i}/{len(issues)}] Verifying: {issue_dict.get('issue_type', 'unknown')}..."
            )

            issue = ClaimIssue(**issue_dict)
            verifier = ClaimVerifier(project_root=root)
            result = await verifier.verify_claim(issue)

            # Add false positive analysis
            fp_detector = FalsePositiveDetector()
            is_fp, fp_reason, fp_conf = fp_detector.is_likely_false_positive(issue)

            # Add severity
            weighting = SeverityWeighting()
            score, severity = weighting.score_and_classify(issue)

            results.append(
                {
                    **issue_dict,
                    "verified": result.is_grounded,
                    "confidence": result.confidence,
                    "false_positive": is_fp,
                    "false_positive_confidence": fp_conf,
                    "severity": severity.value,
                    "severity_score": score,
                }
            )

        # Summary
        verified_count = sum(
            1 for r in results if r["verified"] and r["confidence"] >= min_confidence
        )
        fp_count = sum(
            1 for r in results if r["false_positive"] and r["false_positive_confidence"] > 0.7
        )

        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total issues: {len(results)}")
        console.print(f"  Verified: [green]{verified_count}[/green]")
        console.print(f"  False positives: [yellow]{fp_count}[/yellow]")

        # Severity distribution
        severity_dist: dict[str, int] = {}
        for r in results:
            s = r["severity"]
            severity_dist[s] = severity_dist.get(s, 0) + 1

        console.print(f"\n[bold]Severity Distribution:[/bold]")
        for severity, count in sorted(
            severity_dist.items(), key=lambda x: -SeverityLevel.weight(x[0])
        ):
            console.print(f"  {severity}: {count}")

        # Output file
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(json.dumps(results, indent=2))
            console.print(f"\nResults saved to: {output_file}")

    return _run_async(_verify_batch())


@app.command()
def report(
    project_root: str = typer.Option(".", "--root", "-r", help="Project root directory"),
    output_file: str = typer.Option(
        "verification_report.json", "--output", "-o", help="Output file"
    ),
):
    """
    Generate a comprehensive verification report for the project.

    Analyzes the project and generates a report including:
    - False positive analysis
    - Severity distribution
    - Documentation cross-references
    - Temporal analysis

    Examples:
        victor verify report
        victor verify report --output my_report.json
    """

    async def _generate_report() -> None:
        root = Path(project_root).resolve()
        console.print(f"Analyzing project: {root}")

        # Initialize components
        fp_detector = FalsePositiveDetector()
        crossref = DocumentationCrossReference(project_root=root)
        temporal = TemporalContextAnalyzer(project_root=root)

        # Generate report
        report = {
            "project_root": str(root),
            "timestamp": (str(Path.ctime(Path.cwd())) if hasattr(Path, "ctime") else "N/A"),
            "components": {
                "false_positive_detection": {
                    "enabled": True,
                    "patterns_count": len(fp_detector.FALSE_POSITIVE_PATTERNS),
                },
                "documentation_crossref": {
                    "enabled": True,
                    "tech_debt_entries": len(crossref.get_tech_debt_markers()),
                    "roadmap_priorities": crossref.get_roadmap_priorities(),
                },
                "temporal_analysis": {
                    "enabled": True,
                    "git_available": temporal._git_analyzer._is_git_repo,
                },
            },
        }

        # Save report
        output_path = Path(output_file)
        output_path.write_text(json.dumps(report, indent=2))
        console.print(f"\n[bold]Report saved to:[/bold] {output_file}")

        # Display summary
        console.print(f"\n[bold]Verification Components:[/bold]")
        console.print(
            f"  ✓ False positive detection ({report['components']['false_positive_detection']['patterns_count']} patterns)"
        )
        console.print(
            f"  ✓ Documentation cross-reference ({report['components']['documentation_crossref']['tech_debt_entries']} debt entries)"
        )
        console.print(
            f"  ✓ Temporal analysis ({'Git available' if report['components']['temporal_analysis']['git_available'] else 'Git not available'})"
        )

    return _run_async(_generate_report())


@app.command()
def fp_patterns(
    show_custom: bool = typer.Option(False, "--custom", "-c", help="Show custom patterns only"),
):
    """
    List all false positive detection patterns.

    Examples:
        victor verify fp-patterns
        victor verify fp-patterns --custom
    """
    from rich.text import Text

    detector = FalsePositiveDetector()

    console.print("\n[bold]False Positive Detection Patterns[/bold]")
    for category, patterns in detector.FALSE_POSITIVE_PATTERNS.items():
        if show_custom and not category.startswith("custom"):
            continue
        console.print(f"\n{category}: {len(patterns)} patterns")
        # Show first few patterns as examples (use Text.append to avoid markup parsing)
        for pattern in patterns[:3]:
            prefix = Text("  - ")
            if hasattr(pattern, "pattern"):
                pattern_text = Text(pattern.pattern, style="dim")
            else:
                pattern_text = Text(str(pattern), style="dim")
            console.print(prefix + pattern_text)
        if len(patterns) > 3:
            console.print(f"  ... and {len(patterns) - 3} more")


@app.command()
def doc_check(
    project_root: str = typer.Option(".", "--root", "-r", help="Project root directory"),
):
    """
    Check project documentation for technical debt tracking.

    Displays information about tracked technical debt and roadmap items.

    Examples:
        victor verify doc-check
        victor verify doc-check --root /path/to/project
    """
    root = Path(project_root).resolve()
    crossref = DocumentationCrossReference(project_root=root)

    # Tech debt markers
    markers = crossref.get_tech_debt_markers()
    console.print(f"\n[bold]Technical Debt Markers:[/bold] ({len(markers)} found)")
    for marker in markers[:10]:  # Show first 10
        console.print(f"  • {marker}")
    if len(markers) > 10:
        console.print(f"  ... and {len(markers) - 10} more")

    # Roadmap priorities
    priorities = crossref.get_roadmap_priorities()
    console.print(f"\n[bold]Roadmap Priorities:[/bold]")
    for priority, count in sorted(priorities.items()):
        console.print(f"  {priority}: {count} items")
