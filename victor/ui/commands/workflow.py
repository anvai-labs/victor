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

"""Workflow CLI commands for Victor.

Provides commands for validating, rendering, listing, and executing
YAML-defined workflows.

Commands:
    validate  - Validate YAML workflow files (with escape hatch and handler checks)
    render    - Render workflow DAG as diagram (ascii, mermaid, d2, dot, svg, png)
    list      - List available workflows in a directory
    run       - Execute a workflow

Validation checks:
    - YAML syntax and structure
    - Node references (next_nodes, branches, parallel_nodes)
    - Escape hatch references (CONDITIONS/TRANSFORMS in condition/transform nodes)
    - Handler references (registered compute handlers)
    - HITL node configuration
    - Capability provider integration hints

Example:
    victor workflow validate ./workflows/analysis.yaml
    victor workflow validate ./workflows/analysis.yaml --check-handlers
    victor workflow render ./workflows/analysis.yaml --format svg -o diagram.svg
    victor workflow list ./workflows/
    victor workflow run ./workflows/analysis.yaml --context '{"symbol": "AAPL"}'
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from victor.core.async_utils import run_sync
from victor.ui.delegate_follow_up import (
    DelegateFollowUpContractError,
    load_delegate_follow_up_contract_file,
)
from victor.ui.errors import print_error

workflow_app = typer.Typer(
    name="workflow",
    help="Manage and execute YAML workflows.",
)

console = Console()


def _create_compile_only_compiler() -> Any:
    """Create the canonical compile-only workflow compiler for CLI checks."""
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    return UnifiedWorkflowCompiler(enable_caching=False)


def _load_workflow_file(workflow_path: Path) -> dict:
    """Load and parse a workflow YAML file.

    Args:
        workflow_path: Path to the YAML file

    Returns:
        Dictionary of workflow name -> WorkflowDefinition

    Raises:
        typer.Exit: If file not found or invalid
    """
    from victor.workflows import load_workflow_from_file, YAMLWorkflowError
    from victor.workflows.definition import WorkflowDefinition

    if not workflow_path.exists():
        print_error(f"Workflow file not found: {workflow_path}")
        raise typer.Exit(1)

    if workflow_path.suffix not in {".yaml", ".yml"}:
        print_error(f"File must be .yaml or .yml: {workflow_path}")
        raise typer.Exit(1)

    try:
        loaded = load_workflow_from_file(str(workflow_path))

        workflows: dict[str, WorkflowDefinition] = {}
        if isinstance(loaded, dict):
            workflows = loaded
        else:
            workflows = {loaded.name: loaded}

        if not workflows:
            print_error("No workflows found in file")
            raise typer.Exit(1)

        return workflows

    except YAMLWorkflowError as e:
        print_error(f"Failed to parse workflow: {e}")
        raise typer.Exit(1)


def _load_delegate_follow_up_contract_file(contract_path: Path) -> Dict[str, Any]:
    """Load a delegate follow-up contract JSON file for workflow resume."""
    try:
        return load_delegate_follow_up_contract_file(contract_path)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Delegate follow-up contract not found: {contract_path}")
        raise typer.Exit(1)
    except DelegateFollowUpContractError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


def _display_workflow_info(workflow, wf_name: str) -> None:
    """Display workflow information in a table."""
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Description", workflow.description or "(none)")
    table.add_row("Nodes", str(len(workflow.nodes)))
    table.add_row("Start Node", workflow.start_node)

    # Count node types
    node_types: dict[str, int] = {}
    for node in workflow.nodes.values():
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1

    types_str = ", ".join(f"{k}: {v}" for k, v in sorted(node_types.items()))
    table.add_row("Node Types", types_str)

    console.print(table)


def _detect_vertical_from_path(workflow_path: Path) -> Optional[str]:
    """Detect vertical from workflow file path.

    Looks for vertical names in the path structure, e.g.:
    - victor/coding/workflows/feature.yaml -> coding
    - victor/research/workflows/paper.yaml -> research
    """
    known_verticals = {
        "coding",
        "devops",
        "rag",
        "dataanalysis",
        "research",
        "benchmark",
    }
    parts = workflow_path.parts

    for part in parts:
        if part in known_verticals:
            return part
    return None


def _load_escape_hatches(vertical: Optional[str]) -> tuple[Set[str], Set[str]]:
    """Load escape hatch registries (CONDITIONS and TRANSFORMS) for a vertical.

    Args:
        vertical: Vertical name (coding, research, etc.)

    Returns:
        Tuple of (condition_names, transform_names)
    """
    conditions: Set[str] = set()
    transforms: Set[str] = set()

    if not vertical:
        return conditions, transforms

    try:
        # Import the escape_hatches module for the vertical
        module_name = f"victor.{vertical}.escape_hatches"
        import importlib

        module = importlib.import_module(module_name)

        if hasattr(module, "CONDITIONS"):
            conditions = set(module.CONDITIONS.keys())
        if hasattr(module, "TRANSFORMS"):
            transforms = set(module.TRANSFORMS.keys())

    except ImportError:
        # No escape hatches module for this vertical
        pass

    return conditions, transforms


def _load_registered_handlers(vertical: Optional[str]) -> Set[str]:
    """Load registered compute handler names.

    First ensures vertical handlers are registered, then returns all known handlers.
    """
    handlers: Set[str] = set()

    # Try to register vertical handlers first
    if vertical:
        try:
            module_name = f"victor.{vertical}.handlers"
            import importlib
            import inspect

            module = importlib.import_module(module_name)

            if hasattr(module, "register_handlers"):
                from victor.workflows.compute_registry import register_compute_handler

                register_handlers = module.register_handlers
                try:
                    parameters = inspect.signature(register_handlers).parameters
                except (TypeError, ValueError):
                    parameters = {}

                if parameters:
                    register_handlers(register_compute_handler)
                else:
                    register_handlers()
            if hasattr(module, "HANDLERS"):
                handlers.update(module.HANDLERS.keys())
        except ImportError:
            pass

    # Get all registered handlers from executor
    try:
        from victor.workflows.compute_registry import list_compute_handlers

        handlers.update(list_compute_handlers())
    except ImportError:
        pass

    return handlers


def _validate_escape_hatches_and_handlers(
    workflow,
    vertical: Optional[str],
    check_handlers: bool = False,
) -> tuple[List[str], List[str]]:
    """Validate escape hatch and handler references in workflow.

    Args:
        workflow: WorkflowDefinition to validate
        vertical: Vertical name for loading escape hatches
        check_handlers: Whether to validate handler references

    Returns:
        Tuple of (errors, warnings)
    """
    from victor.workflows.definition import (
        ConditionNode,
        TransformNode,
        ComputeNode,
        ParallelNode,
    )
    from victor.workflows.hitl import HITLNode

    errors: List[str] = []
    warnings: List[str] = []

    # Load registries
    conditions, transforms = _load_escape_hatches(vertical)
    handlers = _load_registered_handlers(vertical) if check_handlers else set()

    # Track what's used
    used_conditions: Set[str] = set()
    used_transforms: Set[str] = set()
    used_handlers: Set[str] = set()

    # Names that indicate simple/inline conditions (not escape hatch references)
    simple_condition_names = {
        "<lambda>",
        "truthy_condition",
        "condition",
        "in_condition",
        "literal_transform",
        "ref_transform",
    }

    for node_id, node in workflow.nodes.items():
        # Check condition nodes for escape hatch references
        if isinstance(node, ConditionNode):
            # Try to determine if this is a simple expression or escape hatch reference
            # Simple expressions get compiled to functions with generic names
            condition_name = getattr(node.condition, "__name__", None)
            if condition_name and condition_name not in simple_condition_names:
                used_conditions.add(condition_name)
                if conditions and condition_name not in conditions:
                    warnings.append(
                        f"Condition node '{node_id}' uses escape hatch '{condition_name}' "
                        f"not found in {vertical}.escape_hatches.CONDITIONS"
                    )

        # Check transform nodes for escape hatch references
        if isinstance(node, TransformNode):
            transform_name = getattr(node.transform, "__name__", None)
            if transform_name and transform_name not in {
                "<lambda>",
                "literal_transform",
                "ref_transform",
            }:
                used_transforms.add(transform_name)
                if transforms and transform_name not in transforms:
                    warnings.append(
                        f"Transform node '{node_id}' uses escape hatch '{transform_name}' "
                        f"not found in {vertical}.escape_hatches.TRANSFORMS"
                    )

        # Check compute nodes for handler references
        if isinstance(node, ComputeNode):
            if node.handler:
                used_handlers.add(node.handler)
                if check_handlers and node.handler not in handlers:
                    errors.append(
                        f"Compute node '{node_id}' references unregistered handler '{node.handler}'"
                    )

        # Check parallel nodes
        if isinstance(node, ParallelNode):
            if not node.parallel_nodes:
                warnings.append(f"Parallel node '{node_id}' has no parallel_nodes defined")

        # Check HITL nodes
        if isinstance(node, HITLNode):
            if not node.prompt:
                warnings.append(f"HITL node '{node_id}' has no prompt defined")
            if node.hitl_type.value == "choice" and not node.choices:
                errors.append(f"HITL choice node '{node_id}' has no choices defined")

    return errors, warnings


def _display_validation_details(
    workflow,
    verbose: bool = False,
    check_handlers: bool = False,
    vertical: Optional[str] = None,
) -> tuple[bool, List[str], List[str]]:
    """Display detailed validation results for a workflow.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    from victor.workflows.definition import (
        AgentNode,
        ConditionNode,
        TransformNode,
        ComputeNode,
        ParallelNode,
    )

    errors: List[str] = []
    warnings: List[str] = []

    # Basic validation
    basic_errors = workflow.validate()
    errors.extend(basic_errors)

    # Extended validation for escape hatches and handlers
    ext_errors, ext_warnings = _validate_escape_hatches_and_handlers(
        workflow, vertical, check_handlers
    )
    errors.extend(ext_errors)
    warnings.extend(ext_warnings)

    if verbose:
        console.print("\n[dim]Nodes:[/]")
        for node_id, node in workflow.nodes.items():
            node_type = type(node).__name__
            next_nodes = getattr(node, "next_nodes", []) or []
            next_str = " -> " + ", ".join(next_nodes) if next_nodes else ""

            # Add additional info based on node type
            info_parts = []
            if isinstance(node, AgentNode):
                info_parts.append(f"role={node.role}")
            elif isinstance(node, ComputeNode) and node.handler:
                info_parts.append(f"handler={node.handler}")
            elif isinstance(node, ConditionNode):
                branches = list(node.branches.keys())[:3]
                info_parts.append(f"branches={branches}")
            elif isinstance(node, ParallelNode):
                info_parts.append(f"parallel={node.parallel_nodes}")

            info_str = f" [{', '.join(info_parts)}]" if info_parts else ""
            console.print(f"  [dim]{node_id}[/] ({node_type}){info_str}{next_str}")

    return len(errors) == 0, errors, warnings


@workflow_app.command("validate")
def validate_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to validate",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed node information",
    ),
    check_handlers: bool = typer.Option(
        False,
        "--check-handlers",
        help="Validate that referenced handlers are registered",
    ),
    check_escape_hatches: bool = typer.Option(
        True,
        "--check-escape-hatches/--no-check-escape-hatches",
        help="Validate escape hatch references against vertical registries",
    ),
    vertical: Optional[str] = typer.Option(
        None,
        "--vertical",
        help="Vertical name for escape hatch validation (auto-detected from path if not specified)",
    ),
) -> None:
    """Validate a YAML workflow file.

    Checks that the workflow file is valid YAML, contains valid node
    definitions, and can be compiled to a state graph.

    Extended validation includes:
    - Escape hatch references (CONDITIONS/TRANSFORMS)
    - Handler references (when --check-handlers is specified)
    - HITL node configuration
    - Parallel node structure

    Example:
        victor workflow validate ./workflows/analysis.yaml
        victor workflow validate ./workflows/analysis.yaml -v
        victor workflow validate ./workflows/analysis.yaml --check-handlers
        victor workflow validate ./workflows/analysis.yaml --vertical coding
    """
    path = Path(workflow_path)
    console.print(f"\n[bold blue]Validating:[/] {path.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    # Auto-detect vertical if not specified
    detected_vertical = vertical or _detect_vertical_from_path(path)
    if detected_vertical:
        console.print(f"[dim]Vertical: {detected_vertical}[/]")

    workflows = _load_workflow_file(path)

    compiler = _create_compile_only_compiler()
    all_valid = True
    total_warnings = 0

    for wf_name, workflow in workflows.items():
        console.print(f"\n[bold cyan]Workflow:[/] {wf_name}")
        _display_workflow_info(workflow, wf_name)

        # Run extended validation
        is_valid, errors, warnings = _display_validation_details(
            workflow,
            verbose=verbose,
            check_handlers=check_handlers,
            vertical=detected_vertical if check_escape_hatches else None,
        )

        # Display warnings
        for warning in warnings:
            console.print(f"[yellow]  Warning:[/] {warning}")
            total_warnings += 1

        # Display errors
        for error in errors:
            console.print(f"[red]  Error:[/] {error}")

        # Try to compile to state graph
        try:
            _compiled = compiler.compile_definition(workflow)
            if is_valid:
                console.print("[bold green]✓[/] Validation passed")
            else:
                console.print("[bold yellow]✓[/] Compiled with errors")
                all_valid = False
        except Exception as e:
            console.print(f"[bold red]✗[/] Compilation failed: {e}")
            all_valid = False

    console.print()

    # Show escape hatch and handler info if verbose
    if verbose and detected_vertical:
        conditions, transforms = _load_escape_hatches(detected_vertical)
        if conditions or transforms:
            console.print(f"\n[dim]Available escape hatches for '{detected_vertical}':[/]")
            if conditions:
                console.print(f"  CONDITIONS: {', '.join(sorted(conditions))}")
            if transforms:
                console.print(f"  TRANSFORMS: {', '.join(sorted(transforms))}")

        if check_handlers:
            handlers = _load_registered_handlers(detected_vertical)
            if handlers:
                console.print(f"  HANDLERS: {', '.join(sorted(handlers))}")
            console.print()

    if all_valid:
        msg = "[bold green]All workflows validated successfully.[/]"
        if total_warnings > 0:
            msg += f" [yellow]({total_warnings} warning(s))[/]"
        console.print(msg)
    else:
        console.print("[bold red]Some workflows failed validation.[/]")
        raise typer.Exit(1)


@workflow_app.command("render")
def render_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to render",
    ),
    format: str = typer.Option(
        "ascii",
        "--format",
        "-f",
        help="Output format: ascii, mermaid, d2, dot, plantuml, svg, png",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (required for svg/png, optional for others)",
    ),
    workflow_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Workflow name to render (if file contains multiple workflows)",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        "-s",
        help="Show workflow summary with node type counts before rendering",
    ),
) -> None:
    """Render a workflow DAG as a diagram.

    Supports multiple output formats:
    - ascii: Terminal-friendly ASCII art (shows all node types with distinct icons)
    - mermaid: Mermaid markdown (for docs, supports HITL/parallel node shapes)
    - d2: D2 diagram language (modern styling with capability hints)
    - dot: Graphviz DOT format
    - plantuml: PlantUML format (with fork/join for parallel nodes)
    - svg: SVG image (requires d2, graphviz, or matplotlib)
    - png: PNG image (requires d2 or graphviz)

    Node type visualization:
    - AgentNode: Stadium shape, blue (@)
    - ComputeNode: Parallelogram, green (#)
    - ConditionNode: Diamond, orange (?)
    - ParallelNode: Double-border subroutine, purple (=)
    - TransformNode: Rectangle, gray (>)
    - HITLNode: Circle, red (!)

    Example:
        victor workflow render ./workflows/analysis.yaml
        victor workflow render ./workflows/analysis.yaml -f mermaid
        victor workflow render ./workflows/analysis.yaml -f svg -o diagram.svg
        victor workflow render ./workflows/analysis.yaml --summary
    """
    from victor.workflows.visualization import (
        WorkflowVisualizer,
        OutputFormat as VizFormat,
        RenderBackend,
    )

    path = Path(workflow_path)
    workflows = _load_workflow_file(path)

    # Select workflow
    if workflow_name:
        if workflow_name not in workflows:
            console.print(f"[bold red]Error:[/] Workflow '{workflow_name}' not found")
            console.print(f"Available: {', '.join(workflows.keys())}")
            raise typer.Exit(1)
        workflow = workflows[workflow_name]
    else:
        workflow = next(iter(workflows.values()))
        if len(workflows) > 1:
            console.print(f"[dim]Multiple workflows found, rendering '{workflow.name}'[/]")
            console.print(f"[dim]Use --name to specify: {', '.join(workflows.keys())}[/]")

    # Show summary if requested
    if summary:
        from victor.workflows.definition import (
            AgentNode,
            ComputeNode,
            ConditionNode,
            ParallelNode,
            TransformNode,
        )

        try:
            from victor.workflows.hitl import HITLNode

            has_hitl = True
        except ImportError:
            HITLNode = None
            has_hitl = False

        console.print(f"\n[bold cyan]Workflow:[/] {workflow.name}")
        _display_workflow_info(workflow, workflow.name)

        # Show node breakdown with icons
        console.print("\n[dim]Node breakdown:[/]")
        for node_id, node in workflow.nodes.items():
            node_type = type(node).__name__
            if node_type == "AgentNode":
                icon = "@"
                extra = f"role={node.role}"
            elif node_type == "ComputeNode":
                icon = "#"
                extra = f"handler={node.handler}" if node.handler else f"tools={len(node.tools)}"
            elif node_type == "ConditionNode":
                icon = "?"
                extra = f"branches={list(node.branches.keys())}"
            elif node_type == "ParallelNode":
                icon = "="
                extra = f"parallel={node.parallel_nodes}"
            elif node_type == "TransformNode":
                icon = ">"
                extra = ""
            elif has_hitl and isinstance(node, HITLNode):
                icon = "!"
                extra = f"type={node.hitl_type.value}"
            else:
                icon = "*"
                extra = ""

            extra_str = f" [{extra}]" if extra else ""
            console.print(f"  [{icon}] {node_id}{extra_str}")
        console.print()

    # Map format string to enum
    format_map = {
        "ascii": VizFormat.ASCII,
        "mermaid": VizFormat.MERMAID,
        "plantuml": VizFormat.PLANTUML,
        "dot": VizFormat.DOT,
        "d2": VizFormat.D2,
        "svg": VizFormat.SVG,
        "png": VizFormat.PNG,
    }

    fmt = format_map.get(format.lower())
    if not fmt:
        console.print(f"[bold red]Error:[/] Unknown format: {format}")
        console.print(f"Supported: {', '.join(format_map.keys())}")
        raise typer.Exit(1)

    # SVG/PNG require output path or generate default
    if fmt in {VizFormat.SVG, VizFormat.PNG} and not output:
        output = str(path.with_suffix(f".{format.lower()}"))
        console.print(f"[dim]Output: {output}[/]")

    viz = WorkflowVisualizer(workflow)

    try:
        result = viz.render(fmt, output, RenderBackend.AUTO)

        if output:
            console.print(f"[bold green]✓[/] Saved to {output}")
        else:
            console.print()
            console.print(result)

    except ImportError as e:
        console.print(f"[bold red]Error:[/] Rendering backend not available: {e}")
        console.print("[dim]For SVG/PNG, install: pip install matplotlib networkx[/]")
        console.print("[dim]Or install d2: brew install d2[/]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Rendering failed: {e}")
        raise typer.Exit(1)


@workflow_app.command("list")
def list_workflows(
    directory: str = typer.Argument(
        ".",
        help="Directory containing workflow YAML files",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search subdirectories recursively",
    ),
) -> None:
    """List available workflows in a directory.

    Scans for .yaml/.yml files and displays workflow information.

    Example:
        victor workflow list ./workflows/
        victor workflow list ./workflows/ -r
    """
    from victor.workflows import load_workflow_from_file, YAMLWorkflowError

    path = Path(directory)
    if not path.exists():
        console.print(f"[bold red]Error:[/] Directory not found: {directory}")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[bold red]Error:[/] Not a directory: {directory}")
        raise typer.Exit(1)

    # Find YAML files
    pattern = "**/*.yaml" if recursive else "*.yaml"
    yaml_files = list(path.glob(pattern)) + list(path.glob(pattern.replace(".yaml", ".yml")))

    if not yaml_files:
        console.print(f"[dim]No workflow files found in {directory}[/]")
        return

    table = Table(title="Available Workflows")
    table.add_column("File", style="cyan")
    table.add_column("Workflow", style="green")
    table.add_column("Description")
    table.add_column("Nodes", justify="right")
    table.add_column("Types")

    total_workflows = 0

    for yaml_file in sorted(yaml_files):
        try:
            loaded = load_workflow_from_file(str(yaml_file))

            if isinstance(loaded, dict):
                workflows = loaded
            else:
                workflows = {loaded.name: loaded}

            for wf_name, workflow in workflows.items():
                node_types = {}
                for node in workflow.nodes.values():
                    node_type = type(node).__name__.replace("Node", "")
                    node_types[node_type] = node_types.get(node_type, 0) + 1

                types_str = ", ".join(f"{v}{k[0]}" for k, v in sorted(node_types.items()))

                table.add_row(
                    yaml_file.name,
                    wf_name,
                    (workflow.description or "")[:40],
                    str(len(workflow.nodes)),
                    types_str,
                )
                total_workflows += 1

        except YAMLWorkflowError:
            table.add_row(yaml_file.name, "[red]Error[/]", "Failed to parse", "-", "-")
        except Exception as e:
            table.add_row(yaml_file.name, "[red]Error[/]", str(e)[:40], "-", "-")

    console.print(table)
    console.print(f"\n[dim]Total: {total_workflows} workflow(s) in {len(yaml_files)} file(s)[/]")


async def _execute_workflow_async(
    workflow: Any,
    initial_context: Dict[str, Any],
    profile: Optional[str],
) -> None:
    from victor.config.settings import load_settings

    from victor.framework.session_config import SessionConfig
    from victor.framework.session_runner import create_victor_client
    from victor.ui.commands.utils import graceful_shutdown
    from victor.workflows import ExecutorConfig, StateGraphExecutor
    from victor.workflows.definition import AgentNode

    import logging

    logger = logging.getLogger(__name__)
    load_settings()

    orchestrators = {}
    has_agent_nodes = any(isinstance(node, AgentNode) for node in workflow.nodes.values())

    if has_agent_nodes:
        node_profiles = set()
        profile_to_nodes = {}

        for node in workflow.nodes.values():
            if isinstance(node, AgentNode) and hasattr(node, "profile") and node.profile:
                node_profiles.add(node.profile)
                profile_to_nodes.setdefault(node.profile, []).append(node.id)
            elif isinstance(node, AgentNode):
                profile_to_nodes.setdefault("default", []).append(node.id)

        all_profiles = node_profiles | {profile}

        console.print(
            f"[dim]Creating orchestrators for profiles: {', '.join(sorted(all_profiles))}[/]"
        )

        for prof, nodes in sorted(profile_to_nodes.items()):
            logger.debug(f"Profile '{prof}' assigned to nodes: {', '.join(nodes)}")

        for profile_name in sorted(all_profiles):
            try:
                logger.debug(f"Creating orchestrator for profile: {profile_name}")
                # ✅ PROPER: Use the framework client seam instead of FrameworkShim
                config = SessionConfig(profile=profile_name)
                client = create_victor_client(config)
                orchestrator = await client.initialize()
                orchestrators[profile_name] = orchestrator
                logger.debug(f"Successfully created orchestrator for profile: {profile_name}")
                console.print(f"  [dim]✓ Profile '{profile_name}' initialized[/]")
            except Exception as e:
                logger.error(
                    f"Failed to create orchestrator for profile '{profile_name}': {e}",
                    exc_info=True,
                )
                console.print(f"  [red]✗ Profile '{profile_name}' failed: {e}[/]")
                if profile_name == profile:
                    raise

    executor = StateGraphExecutor(
        config=ExecutorConfig(
            enable_checkpointing=True,
            max_iterations=workflow.max_iterations or 15,
            timeout=workflow.max_execution_timeout_seconds or 3600.0,
            default_profile=profile,
        ),
        orchestrators=orchestrators,
    )

    project_root_override = workflow.metadata.get("project_root")
    if project_root_override:
        from victor.config.settings import set_project_root

        project_path = Path(project_root_override).expanduser().resolve()
        set_project_root(project_path)
        console.print(f"[dim]📁 Project root set to: {project_path}[/]\n")
        logger.debug(f"Project root set from workflow metadata: {project_path}")

    console.print(f"\n[bold]Executing workflow '{workflow.name}'...[/]\n")
    logger.debug(f"\n{'='*80}")
    logger.debug(f"🚀 WORKFLOW EXECUTION START: {workflow.name}")
    logger.debug(f"   Initial Context: {list(initial_context.keys())}")
    logger.debug(f"   Total Nodes: {len(workflow.nodes)}")
    logger.debug(f"   Node IDs: {', '.join(workflow.nodes.keys())}")
    logger.debug(f"{'='*80}\n")

    try:
        result = await executor.execute(workflow, initial_context)
        logger.debug(f"\n{'='*80}")
        logger.debug(f"✅ WORKFLOW EXECUTION COMPLETE: {workflow.name}")
        logger.debug(f"   Success: {result.success}")
        logger.debug(f"   Duration: {result.duration_seconds:.2f}s")
        logger.debug(f"   Nodes Executed: {len(result.nodes_executed)}")
        # Convert Pydantic model to dict if needed for keys()
        state_dict = result.state.to_dict() if hasattr(result.state, "to_dict") else result.state
        logger.debug(f"   Final State Keys: {list(state_dict.keys())}")
        logger.debug(f"{'='*80}\n")

        console.print("[dim]" + "─" * 50 + "[/]")

        if result.success:
            console.print("[bold green]✓[/] Workflow completed successfully")
            console.print(f"  [dim]Duration: {result.duration_seconds:.2f}s[/]")
            console.print(f"  [dim]Nodes executed: {', '.join(result.nodes_executed)}[/]")

            if result.state:
                console.print("\n[bold]Final State:[/]")
                # Convert Pydantic model to dict if needed
                state_dict = (
                    result.state.to_dict() if hasattr(result.state, "to_dict") else result.state
                )
                display_state = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("_") and k not in {"messages", "history"}
                }
                console.print(json.dumps(display_state, indent=2, default=str)[:2000])
        else:
            console.print("[bold red]✗[/] Workflow failed")
            if result.error:
                console.print(f"  [red]{result.error}[/]")
            raise typer.Exit(1)

    finally:
        if orchestrators:
            for orch_name, orch in orchestrators.items():
                try:
                    await graceful_shutdown(orch)
                    console.print(f"[dim]✓ Shut down orchestrator '{orch_name}'[/]")
                except Exception as e:
                    console.print(f"[red]✗ Failed to shutdown '{orch_name}': {e}[/]")


@workflow_app.command("run")
def run_workflow(
    workflow_path: str = typer.Argument(
        ...,
        help="Path to YAML workflow file to execute",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help='Initial context as JSON string (e.g., \'{"symbol": "AAPL"}\')',
    ),
    context_file: Optional[str] = typer.Option(
        None,
        "--context-file",
        "-f",
        help="Path to JSON file with initial context",
    ),
    delegate_follow_up_contract: Optional[str] = typer.Option(
        None,
        "--delegate-follow-up-contract",
        help=(
            "Path to a JSON delegate follow-up contract to inject into workflow state "
            "for TeamStep resume/review/merge execution"
        ),
    ),
    delegate_next_step_id: Optional[str] = typer.Option(
        None,
        "--delegate-next-step-id",
        help="Explicit follow-up step_id to execute from the delegate follow-up contract",
    ),
    workflow_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Workflow name to run (if file contains multiple workflows)",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Profile to use for agent nodes",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and show execution plan without running",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
) -> None:
    """Execute a workflow.

    Runs the workflow with the provided initial context. Agent nodes
    will use the specified profile for LLM interactions.

    Example:
        victor workflow run ./workflows/analysis.yaml -c '{"symbol": "AAPL"}'
        victor workflow run ./workflows/analysis.yaml -f context.json
        victor workflow run ./workflows/delegate.yaml --delegate-follow-up-contract follow-up.json
        victor workflow run ./workflows/analysis.yaml --dry-run
        victor workflow run ./workflows/analysis.yaml --log-level DEBUG
    """
    # Set log level if specified
    if log_level:
        import logging

        level = getattr(logging, log_level.upper(), None)
        if not isinstance(level, int):
            console.print(f"[bold red]Error:[/] Invalid log level: {log_level}")
            raise typer.Exit(1)

        # Configure logging
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

        # Filter out noisy HTTP third-party logs at DEBUG level
        # Only show WARNING and above for HTTP libraries
        if level <= logging.DEBUG:
            for http_logger in ["httpcore", "httpx", "urllib3"]:
                logging.getLogger(http_logger).setLevel(logging.WARNING)

        console.print(f"[dim]Log level set to: {log_level}[/]")

    path = Path(workflow_path)
    workflows = _load_workflow_file(path)

    # Select workflow
    if workflow_name:
        if workflow_name not in workflows:
            console.print(f"[bold red]Error:[/] Workflow '{workflow_name}' not found")
            raise typer.Exit(1)
        workflow = workflows[workflow_name]
    else:
        workflow = next(iter(workflows.values()))

    # Parse context
    initial_context = {}
    if context:
        try:
            initial_context = json.loads(context)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Invalid JSON in --context: {e}")
            raise typer.Exit(1)
        if not isinstance(initial_context, dict):
            console.print("[bold red]Error:[/] --context must be a JSON object")
            raise typer.Exit(1)

    if context_file:
        context_path = Path(context_file)
        if not context_path.exists():
            console.print(f"[bold red]Error:[/] Context file not found: {context_file}")
            raise typer.Exit(1)
        try:
            with open(context_path) as f:
                file_context = json.load(f)
                if not isinstance(file_context, dict):
                    console.print("[bold red]Error:[/] Context file must contain a JSON object")
                    raise typer.Exit(1)
                initial_context.update(file_context)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] Invalid JSON in context file: {e}")
            raise typer.Exit(1)

    if delegate_follow_up_contract:
        follow_up_contract = _load_delegate_follow_up_contract_file(
            Path(delegate_follow_up_contract)
        )
        initial_context["delegate_follow_up_contract"] = follow_up_contract
        if delegate_next_step_id:
            initial_context["delegate_next_step_id"] = delegate_next_step_id
    elif delegate_next_step_id:
        console.print(
            "[bold red]Error:[/] --delegate-next-step-id requires " "--delegate-follow-up-contract"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Workflow:[/] {workflow.name}")
    console.print("[dim]" + "─" * 50 + "[/]")
    _display_workflow_info(workflow, workflow.name)

    if initial_context:
        console.print(f"\n[dim]Context: {json.dumps(initial_context, default=str)[:100]}...[/]")

    if dry_run:
        console.print("\n[bold yellow]Dry run mode[/] - showing execution plan:")

        # Show execution order
        compiler = _create_compile_only_compiler()
        try:
            compiler.compile_definition(workflow)  # Validate compilation
            console.print("[bold green]✓[/] Workflow can be compiled")
            console.print(f"[dim]Start node: {workflow.start_node}[/]")
        except Exception as e:
            console.print(f"[bold red]✗[/] Compilation failed: {e}")
            raise typer.Exit(1)
        return

    run_sync(_execute_workflow_async(workflow, initial_context, profile))


@workflow_app.command("backends")
def list_backends() -> None:
    """Show available rendering backends.

    Lists which rendering backends are installed and available
    for SVG/PNG output.

    Example:
        victor workflow backends
    """
    from victor.workflows.visualization import get_available_backends

    backends = get_available_backends()

    table = Table(title="Rendering Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Status")
    table.add_column("Install Command")

    install_hints = {
        "d2": "brew install d2",
        "graphviz": "brew install graphviz",
        "mermaid-cli": "npm install -g @mermaid-js/mermaid-cli",
        "plantuml": "brew install plantuml",
        "matplotlib": "pip install matplotlib networkx",
        "kroki": "(cloud API - always available)",
        "ascii": "(built-in - always available)",
    }

    for name, available in backends.items():
        status = "[green]✓ Available[/]" if available else "[red]✗ Not installed[/]"
        hint = install_hints.get(name, "")
        table.add_row(name, status, hint if not available else "[dim]installed[/]")

    console.print(table)
    console.print("\n[dim]For SVG/PNG rendering, at least one backend is required.[/]")


def _convert_nodes_to_yaml(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert node list to YAML-compatible format."""
    yaml_nodes = []
    for node in nodes:
        yaml_node: Dict[str, Any] = {
            "id": node.get("id", "unknown"),
            "type": node.get("type", "agent"),
        }

        node_type = node.get("type", "agent")
        if node_type == "agent":
            yaml_node["role"] = node.get("role", "executor")
            yaml_node["goal"] = node.get("goal", "")
            if "tool_budget" in node:
                yaml_node["tool_budget"] = node["tool_budget"]
            if "tools" in node:
                yaml_node["tools"] = node["tools"]

        elif node_type == "compute":
            if "handler" in node:
                yaml_node["handler"] = node["handler"]
            if "input_mapping" in node:
                yaml_node["inputs"] = node["input_mapping"]

        elif node_type == "condition":
            yaml_node["condition"] = node.get("condition", "")
            yaml_node["branches"] = node.get("branches", {})

        elif node_type == "parallel":
            yaml_node["parallel_nodes"] = node.get("parallel_nodes", [])

        elif node_type == "transform":
            yaml_node["transform"] = node.get("transform", "")

        if "output_key" in node:
            yaml_node["output"] = node["output_key"]

        if "next_nodes" in node:
            yaml_node["next"] = node["next_nodes"]
        elif "edges" in node:
            targets = [e.get("target") for e in node.get("edges", [])]
            if targets:
                yaml_node["next"] = targets

        yaml_nodes.append(yaml_node)

    return yaml_nodes


async def _generate_workflow_async(
    description: str,
    output: Optional[str],
    vertical: str,
    profile: Optional[str],
    gen_strategy: Any,
    strategy: str,
    interactive: bool,
    validate: bool,
    max_retries: int,
    dry_run: bool,
) -> None:
    from victor.config.settings import load_settings

    from victor.framework.session_config import SessionConfig
    from victor.framework.session_runner import create_victor_client
    from victor.ui.commands.utils import graceful_shutdown
    from victor.workflows.generation import (
        PipelineMode,
        RequirementPipeline,
        WorkflowGenerationPipeline,
        WorkflowValidator,
    )

    import yaml

    load_settings()

    profile_name = profile or "default"
    try:
        # ✅ PROPER: Use the framework client seam instead of FrameworkShim
        config = SessionConfig(profile=profile_name, vertical=vertical)
        client = create_victor_client(config)
        orchestrator = await client.initialize()
        console.print(f"[dim]Using profile: {profile_name}[/]")
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to create orchestrator: {e}")
        raise typer.Exit(1)

    try:
        console.print("\n[bold]Step 1:[/] Extracting requirements...")

        req_pipeline = RequirementPipeline(orchestrator, vertical=vertical)
        requirements = await req_pipeline.extract_and_validate(description)

        console.print(f"  [green]✓[/] Extracted {len(requirements.functional.tasks)} tasks")
        console.print(f"  [dim]Tools: {list(requirements.functional.tools.keys())}[/]")
        console.print(f"  [dim]Execution order: {requirements.structural.execution_order}[/]")

        if requirements.metadata.ambiguities:
            console.print(
                f"  [yellow]⚠[/] {len(requirements.metadata.ambiguities)} ambiguities detected"
            )
            for amb in requirements.metadata.ambiguities[:3]:
                console.print(f"    - {amb.description}")

            if interactive:
                console.print("\n[bold]Interactive clarification:[/]")
                console.print("  [dim](Interactive mode not yet fully implemented)[/]")

        if dry_run:
            console.print("\n[bold yellow]Dry run mode[/] - requirements extracted:")
            console.print(
                Panel(
                    json.dumps(requirements.to_dict(), indent=2, default=str)[:2000],
                    title="Requirements",
                    border_style="blue",
                )
            )
            return

        console.print(f"\n[bold]Step 2:[/] Generating workflow (strategy: {strategy})...")

        pipeline = WorkflowGenerationPipeline(
            orchestrator=orchestrator,
            vertical=vertical,
            generation_strategy=gen_strategy,
            max_retries=max_retries,
        )

        result = await pipeline.run(
            requirements,
            mode=PipelineMode.GENERATE_ONLY,
        )

        if not result.success:
            console.print(f"[bold red]✗[/] Generation failed: {result.error}")
            if result.validation_result and not result.validation_result.is_valid:
                for err in result.validation_result.all_errors[:5]:
                    console.print(f"  - {err.message}")
            raise typer.Exit(1)

        console.print("  [green]✓[/] Generated workflow schema")
        console.print(f"  [dim]Duration: {result.metadata.duration_seconds:.2f}s[/]")

        console.print("\n[bold]Step 3:[/] Converting to YAML...")

        workflow_schema = result.workflow_schema
        if workflow_schema is None:
            console.print("[bold red]✗[/] No workflow schema generated")
            raise typer.Exit(1)

        yaml_data = {
            "workflows": {
                workflow_schema.get("workflow_name", "generated_workflow"): {
                    "description": workflow_schema.get("description", description[:100]),
                    "metadata": {
                        "vertical": vertical,
                        "generated_by": "victor workflow generate",
                        "generation_strategy": strategy,
                    },
                    "nodes": _convert_nodes_to_yaml(workflow_schema.get("nodes", [])),
                }
            }
        }

        yaml_output = yaml.dump(
            yaml_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

        if validate:
            console.print("\n[bold]Step 4:[/] Validating generated workflow...")
            validator = WorkflowValidator()
            val_result = validator.validate(workflow_schema)

            if val_result.is_valid:
                console.print("  [green]✓[/] Validation passed")
            else:
                console.print(f"  [yellow]⚠[/] Validation warnings: {len(val_result.all_errors)}")
                for err in val_result.all_errors[:3]:
                    console.print(f"    - {err.message}")

        console.print("\n[dim]" + "─" * 50 + "[/]")

        if output:
            output_path = Path(output)
            output_path.write_text(yaml_output)
            console.print(f"[bold green]✓[/] Saved to {output}")
        else:
            console.print("[bold]Generated Workflow:[/]\n")
            console.print(yaml_output)

        console.print("\n[bold green]✓[/] Workflow generation complete")

    finally:
        try:
            await graceful_shutdown(orchestrator)
        except Exception:
            pass


@workflow_app.command("generate")
def generate_workflow(
    description: str = typer.Argument(
        ...,
        help="Natural language description of the workflow to generate",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for generated YAML (defaults to stdout)",
    ),
    vertical: str = typer.Option(
        "coding",
        "--vertical",
        "-V",
        help="Target vertical (coding, devops, research, rag, dataanalysis)",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Profile to use for LLM generation",
    ),
    strategy: str = typer.Option(
        "multi_stage",
        "--strategy",
        "-s",
        help="Generation strategy: multi_stage, single_stage, or template",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Enable interactive clarification for ambiguous requirements",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate generated workflow after generation",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Maximum LLM generation retry attempts",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Extract requirements only, don't generate workflow",
    ),
) -> None:
    """Generate a workflow YAML from natural language description.

    Uses LLM-based extraction to convert natural language descriptions
    into structured workflow definitions compatible with StateGraph.

    The generation pipeline:
    1. Extract requirements from natural language
    2. Optionally clarify ambiguous requirements (--interactive)
    3. Generate workflow schema using chosen strategy
    4. Validate generated workflow
    5. Output YAML file or display

    Strategies:
    - multi_stage: Most accurate, uses 3-stage LLM generation
    - single_stage: Faster, single LLM call
    - template: Fastest, matches to pre-defined templates

    Example:
        victor workflow generate "Analyze code, find bugs, fix them, run tests"
        victor workflow generate "Research topic, summarize findings, create report" -V research
        victor workflow generate "Build and deploy container" -V devops -o deploy.yaml
        victor workflow generate "Analyze data, visualize results" --interactive
    """
    from victor.workflows.generation import GenerationStrategy

    # Map strategy string to enum
    strategy_map = {
        "multi_stage": GenerationStrategy.LLM_MULTI_STAGE,
        "single_stage": GenerationStrategy.LLM_SINGLE_STAGE,
        "template": GenerationStrategy.TEMPLATE_BASED,
    }

    gen_strategy = strategy_map.get(strategy.lower())
    if not gen_strategy:
        console.print(f"[bold red]Error:[/] Unknown strategy: {strategy}")
        console.print(f"Valid options: {', '.join(strategy_map.keys())}")
        raise typer.Exit(1)

    # Validate vertical
    valid_verticals = {
        "coding",
        "devops",
        "research",
        "rag",
        "dataanalysis",
        "benchmark",
    }
    if vertical.lower() not in valid_verticals:
        console.print(f"[bold red]Error:[/] Unknown vertical: {vertical}")
        console.print(f"Valid options: {', '.join(valid_verticals)}")
        raise typer.Exit(1)

    console.print("\n[bold blue]Generating workflow from description...[/]")
    console.print("[dim]" + "─" * 50 + "[/]")
    console.print(
        f"[dim]Description:[/] {description[:100]}{'...' if len(description) > 100 else ''}"
    )
    console.print(f"[dim]Vertical:[/] {vertical}")
    console.print(f"[dim]Strategy:[/] {strategy}")

    run_sync(
        _generate_workflow_async(
            description=description,
            output=output,
            vertical=vertical,
            profile=profile,
            gen_strategy=gen_strategy,
            strategy=strategy,
            interactive=interactive,
            validate=validate,
            max_retries=max_retries,
            dry_run=dry_run,
        )
    )
