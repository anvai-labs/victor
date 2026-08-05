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

"""Product bundle management commands."""

from __future__ import annotations

from typing import Optional

import typer

from victor.core.async_utils import run_sync
from victor.verticals.product_bundle import (
    list_bundles,
    get_bundle,
    get_install_command,
    resolve_bundle_dependencies,
)
from victor.verticals.unified_registry import get_registry, VerticalStatus

app = typer.Typer(help="Product bundle management commands")


@app.command()
def list(
    bundle: Optional[str] = typer.Option(None, "--bundle", "-b", help="Filter by bundle"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
):
    """
    List installed verticals.

    Examples:
        victor bundle list
        victor bundle list --bundle engineering
        victor bundle list --status installed
    """

    async def _list() -> None:
        registry = await get_registry()

        if bundle:
            bundle_obj = get_bundle(bundle)
            if not bundle_obj:
                typer.echo(f"Bundle '{bundle}' not found")
                typer.echo("Run 'victor bundle bundles' to see available bundles")
                raise typer.Exit(1)

            typer.echo(f"\n{bundle_obj.display_name}")
            typer.echo(f"  {bundle_obj.description}")
            typer.echo("\nVerticals:")

            for vertical_name in bundle_obj.verticals:
                info = registry.get_vertical(vertical_name)
                if info:
                    status_icon = "✓" if info.is_available() else "✗"
                    version_str = info.installed_version or "missing"
                    typer.echo(f"  {status_icon} {vertical_name} ({version_str})")
                else:
                    typer.echo(f"  ✗ {vertical_name} (not registered)")
        else:
            # Apply status filter if provided
            status_filter = None
            if status:
                try:
                    status_filter = VerticalStatus(status)
                except ValueError:
                    typer.echo(f"Invalid status: {status}")
                    typer.echo(f"Valid statuses: {[s.value for s in VerticalStatus]}")
                    raise typer.Exit(1)

            verticals = registry.list_verticals(status=status_filter)

            if status_filter:
                typer.echo(f"\nInstalled Verticals (status={status_filter}):")
            else:
                typer.echo("\nInstalled Verticals:")

            if not verticals:
                typer.echo("  No verticals found")
                return

            for info in verticals:
                status_icon = {
                    VerticalStatus.INSTALLED: "✓",
                    VerticalStatus.MISSING: "✗",
                    VerticalStatus.VERSION_MISMATCH: "⚠",
                    VerticalStatus.INCOMPATIBLE: "!",
                }.get(info.status, "?")

                version_str = info.installed_version or "missing"
                typer.echo(f"  {status_icon} {info.name} ({version_str})")

                if info.capabilities:
                    typer.echo(f"    Capabilities: {', '.join(sorted(info.capabilities))}")

    run_sync(_list())


@app.command()
def bundles():
    """
    List available product bundles.

    Example:
        victor bundle bundles
    """
    typer.echo("\nAvailable Product Bundles:\n")

    for bundle in list_bundles():
        tier_str = bundle.tier.value.upper()
        typer.echo(f"  {bundle.name} - {bundle.display_name} [{tier_str}]")
        typer.echo(f"    {bundle.description}")

        verticals_str = ", ".join(bundle.verticals) if bundle.verticals else "None (core framework)"
        typer.echo(f"    Verticals: {verticals_str}")

        if bundle.optional_verticals:
            optional_str = ", ".join(bundle.optional_verticals)
            typer.echo(f"    Optional: {optional_str}")

        typer.echo()


@app.command()
def info(
    bundle: str = typer.Argument(..., help="Bundle name to show info for"),
):
    """
    Show detailed information about a bundle.

    Examples:
        victor bundle info engineering
        victor bundle info enterprise
    """
    bundle_obj = get_bundle(bundle)

    if not bundle_obj:
        typer.echo(f"Bundle '{bundle}' not found")
        typer.echo("Run 'victor bundle bundles' to see available bundles")
        raise typer.Exit(1)

    typer.echo(f"\n{bundle_obj.display_name}")
    typer.echo("=" * len(bundle_obj.display_name))
    typer.echo(f"\nDescription: {bundle_obj.description}")
    typer.echo(f"Tier: {bundle_obj.tier.value.upper()}")

    all_verticals = bundle_obj.get_all_verticals()
    typer.echo(f"\nRequired Verticals ({len(all_verticals)}):")
    for vertical in sorted(all_verticals):
        typer.echo(f"  - {vertical}")

    if bundle_obj.optional_verticals:
        typer.echo(f"\nOptional Verticals ({len(bundle_obj.optional_verticals)}):")
        for vertical in sorted(bundle_obj.optional_verticals):
            typer.echo(f"  - {vertical}")

    if bundle_obj.dependencies:
        typer.echo(f"\nBundle Dependencies: {', '.join(bundle_obj.dependencies)}")

    typer.echo("\nInstall Command:")
    typer.echo(f"  {bundle_obj.get_install_command()}")

    # Check if bundle is available
    async def _check_availability() -> None:
        registry = await get_registry()

        if registry.is_bundle_available(bundle):
            typer.echo("\nStatus: ✓ All verticals installed")
        else:
            missing = registry.get_missing_verticals_for_bundle(bundle)
            typer.echo(f"\nStatus: ✗ Missing {len(missing)} vertical(s)")
            typer.echo(f"  Missing: {', '.join(missing)}")

    run_sync(_check_availability())
    typer.echo()


@app.command()
def install(
    bundle: str = typer.Argument(..., help="Bundle name to install"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show command without executing"),
):
    """
    Install a product bundle.

    Examples:
        victor bundle install engineering
        victor bundle install enterprise --dry-run
    """
    bundle_obj = get_bundle(bundle)

    if not bundle_obj:
        typer.echo(f"Bundle '{bundle}' not found")
        typer.echo("Run 'victor bundle bundles' to see available bundles")
        raise typer.Exit(1)

    typer.echo(f"\nInstalling {bundle_obj.display_name}...")
    typer.echo(f"\nThis will install: {', '.join(bundle_obj.verticals) or 'core framework only'}")

    if bundle_obj.optional_verticals:
        typer.echo(f"\nOptional verticals: {', '.join(bundle_obj.optional_verticals)}")

    install_cmd = get_install_command(bundle)

    typer.echo("\nInstall command:")
    typer.echo(f"  {install_cmd}")

    if dry_run:
        typer.echo("\n[Dry run - command not executed]")
    else:
        typer.echo("\nTo install, run the command above manually.")
        typer.echo("Note: Some verticals may require additional configuration.")


@app.command()
def check(
    bundle: Optional[str] = typer.Option(None, "--bundle", "-b", help="Check specific bundle"),
):
    """
    Check vertical installation status.

    Examples:
        victor bundle check
        victor bundle check --bundle engineering
    """

    async def _check() -> None:
        registry = await get_registry()

        if bundle:
            bundle_obj = get_bundle(bundle)
            if not bundle_obj:
                typer.echo(f"Bundle '{bundle}' not found")
                raise typer.Exit(1)

            typer.echo(f"\nChecking {bundle_obj.display_name}...")

            if registry.is_bundle_available(bundle):
                typer.echo("  ✓ All required verticals installed")
            else:
                missing = registry.get_missing_verticals_for_bundle(bundle)
                typer.echo(f"  ✗ Missing {len(missing)} vertical(s):")
                for m in missing:
                    typer.echo(f"    - {m}")
        else:
            typer.echo("\nChecking all verticals...\n")

            verticals = registry.list_verticals()
            installed = sum(1 for v in verticals if v.is_available())
            total = len(verticals)

            typer.echo(f"Installed: {installed}/{total}")

            for info in verticals:
                status_icon = "✓" if info.is_available() else "✗"
                typer.echo(f"  {status_icon} {info.name}")

    run_sync(_check())


@app.command()
def capabilities():
    """
    List all available capabilities across verticals.

    Example:
        victor bundle capabilities
    """

    async def _list_capabilities() -> None:
        registry = await get_registry()

        capabilities = sorted(registry.get_capabilities())

        typer.echo(f"\nAvailable Capabilities ({len(capabilities)}):\n")

        for capability in capabilities:
            verticals = registry.get_verticals_with_capability(capability)
            vertical_names = [v.name for v in verticals if v.is_available()]
            typer.echo(f"  {capability}")
            if vertical_names:
                typer.echo(f"    Provided by: {', '.join(vertical_names)}")

    run_sync(_list_capabilities())
