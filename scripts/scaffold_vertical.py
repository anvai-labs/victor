#!/usr/bin/env python3
"""Scaffold a new Victor vertical package.

Generates a complete vertical package skeleton based on the research
vertical pattern. No external dependencies required (no cookiecutter).

Usage:
    python scripts/scaffold_vertical.py my-vertical
    python scripts/scaffold_vertical.py my-vertical --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent


def to_module_name(name: str) -> str:
    """Convert package name to Python module name."""
    return name.replace("-", "_").replace(" ", "_").lower()


def to_class_name(name: str) -> str:
    """Convert package name to class name."""
    parts = name.replace("-", " ").replace("_", " ").split()
    return "".join(p.capitalize() for p in parts)


def create_file(path: Path, content: str) -> None:
    """Create a file with content, ensuring parent dirs exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip("\n"))
    print(f"  Created: {path}")


def scaffold(name: str, output_dir: Path) -> Path:
    """Generate the vertical package structure.

    Args:
        name: Package name (e.g., 'my-vertical')
        output_dir: Parent directory for the package

    Returns:
        Path to the created package directory
    """
    pkg_name = f"victor-{name}" if not name.startswith("victor-") else name
    mod_name = to_module_name(pkg_name)
    class_name = to_class_name(name.removeprefix("victor-") if name.startswith("victor-") else name)
    vertical_name = name.removeprefix("victor-") if name.startswith("victor-") else name

    pkg_dir = output_dir / pkg_name
    src_dir = pkg_dir / mod_name

    if pkg_dir.exists():
        print(f"Error: Directory {pkg_dir} already exists", file=sys.stderr)
        sys.exit(1)

    print(f"Scaffolding vertical: {pkg_name}")
    print(f"  Module: {mod_name}")
    print(f"  Class: {class_name}Assistant")
    print(f"  Vertical name: {vertical_name}")
    print()

    # pyproject.toml
    create_file(
        pkg_dir / "pyproject.toml",
        f"""
        [build-system]
        requires = ["setuptools>=68.0", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{pkg_name}"
        version = "0.1.0"
        description = "Victor vertical for {vertical_name}"
        requires-python = ">=3.12"
        dependencies = [
            "victor-contracts>=0.7.0",
        ]

        [project.entry-points."victor.plugins"]
        {vertical_name} = "{mod_name}:plugin"
    """,
    )

    # Package __init__.py
    create_file(
        src_dir / "__init__.py",
        f'''
        """{class_name} Vertical Package for Victor."""

        from typing import Optional

        import typer
        from victor_contracts import PluginContext, VictorPlugin

        from {mod_name}.assistant import {class_name}Assistant


        class {class_name}Plugin(VictorPlugin):
            """Victor Plugin for {class_name} vertical."""

            @property
            def name(self) -> str:
                return "{vertical_name}"

            def register(self, context: PluginContext) -> None:
                """Register the {vertical_name} vertical."""
                context.register_vertical({class_name}Assistant)

            def get_cli_app(self) -> Optional[typer.Typer]:
                return None

            def on_activate(self) -> None:
                pass

            def on_deactivate(self) -> None:
                pass

            async def on_activate_async(self) -> None:
                pass

            async def on_deactivate_async(self) -> None:
                pass

            def health_check(self) -> dict[str, object]:
                return {{"healthy": True, "vertical": "{vertical_name}"}}


        plugin = {class_name}Plugin()

        __all__ = ["{class_name}Assistant", "{class_name}Plugin", "plugin"]
    ''',
    )

    # Assistant module
    create_file(
        src_dir / "assistant.py",
        f'''
        """{class_name} Assistant - contract-first vertical definition."""

        from __future__ import annotations

        from typing import ClassVar

        from victor_contracts import (
            CapabilityIds,
            CapabilityRequirement,
            StageDefinition,
            ToolNames,
            ToolRequirement,
            VerticalBase,
            register_vertical,
        )


        @register_vertical(
            name="{vertical_name}",
            version="0.1.0",
            min_framework_version=">=0.6.0",
            plugin_namespace="{mod_name}",
        )
        class {class_name}Assistant(VerticalBase):
            """Domain-specific assistant for {vertical_name}."""

            name: ClassVar[str] = "{vertical_name}"
            description: ClassVar[str] = "Assistant for {vertical_name} tasks"
            version: ClassVar[str] = "0.1.0"

            @classmethod
            def get_name(cls) -> str:
                return cls.name

            @classmethod
            def get_description(cls) -> str:
                return cls.description

            @classmethod
            def get_tool_requirements(cls) -> list[ToolRequirement]:
                return [
                    ToolRequirement(ToolNames.READ, purpose="inspect project files"),
                    ToolRequirement(ToolNames.WRITE, required=False, purpose="apply changes"),
                    ToolRequirement(ToolNames.SHELL, required=False, purpose="run project checks"),
                ]

            @classmethod
            def get_tools(cls) -> list[str]:
                return [requirement.tool_name for requirement in cls.get_tool_requirements()]

            @classmethod
            def get_capability_requirements(cls) -> list[CapabilityRequirement]:
                return [
                    CapabilityRequirement(
                        capability_id=CapabilityIds.FILE_OPS,
                        purpose="read and update local project files",
                    ),
                ]

            @classmethod
            def get_system_prompt(cls) -> str:
                return (
                    "You are a specialized assistant for {vertical_name} tasks. "
                    "Help the user accomplish their goals efficiently and safely."
                )

            @classmethod
            def get_stages(cls) -> dict[str, StageDefinition]:
                return {{
                    "default": StageDefinition(
                        name="default",
                        description="Default execution stage for {vertical_name} work.",
                        required_tools=[ToolNames.READ],
                        optional_tools=[ToolNames.WRITE, ToolNames.SHELL],
                    )
                }}
    ''',
    )

    # Safety module
    create_file(
        src_dir / "safety.py",
        f'''
        """{class_name} Safety Patterns."""

        from __future__ import annotations

        from typing import List


        # Dangerous patterns that should trigger confirmation
        DANGEROUS_PATTERNS: List[str] = []

        # Commands that are always blocked
        BLOCKED_COMMANDS: List[str] = []
    ''',
    )

    # Prompts module
    create_file(
        src_dir / "prompts.py",
        f'''
        """{class_name} Prompt Templates."""

        from __future__ import annotations


        SYSTEM_PROMPT = (
            "You are a specialized assistant for {vertical_name} tasks. "
            "Help the user accomplish their goals efficiently and safely."
        )
    ''',
    )

    # Tests
    create_file(pkg_dir / "tests" / "__init__.py", "")

    create_file(
        pkg_dir / "tests" / f"test_{to_module_name(vertical_name)}.py",
        f'''
        """Tests for {class_name} vertical."""

        from {mod_name}.assistant import {class_name}Assistant


        def test_vertical_name():
            assert {class_name}Assistant.name == "{vertical_name}"


        def test_get_tools():
            tools = {class_name}Assistant.get_tools()
            assert isinstance(tools, list)
            assert len(tools) > 0


        def test_get_system_prompt():
            prompt = {class_name}Assistant.get_system_prompt()
            assert isinstance(prompt, str)
            assert len(prompt) > 0


        def test_get_definition():
            definition = {class_name}Assistant.get_definition()
            assert definition.name == "{vertical_name}"
            assert definition.version == "0.1.0"
            assert definition.tools


        def test_get_manifest():
            manifest = {class_name}Assistant.get_manifest()
            assert manifest.name == "{vertical_name}"
            assert manifest.version == "0.1.0"
    ''',
    )

    print(f"\nVertical package created at: {pkg_dir}")
    print("\nNext steps:")
    print(f"  cd {pkg_dir}")
    print("  pip install -e .")
    print("  pytest tests/ -v")

    return pkg_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new Victor vertical package")
    parser.add_argument("name", help="Vertical name (e.g., 'my-vertical' or 'security')")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Parent directory for the package (default: current directory)",
    )

    args = parser.parse_args()
    scaffold(args.name, args.output_dir)


if __name__ == "__main__":
    main()
