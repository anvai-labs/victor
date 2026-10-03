#!/usr/bin/env python3
"""Build standalone Victor binary using PyInstaller.

This script creates a standalone executable that includes:
- Victor server (API + orchestrator)
- All tools (54+)
- Sentence-transformers for embeddings
- Provider integrations

Usage:
    python scripts/build_binary.py
    python scripts/build_binary.py --onefile  # Single file executable
    python scripts/build_binary.py --clean    # Clean build artifacts first

The resulting binary can be:
1. Distributed with VS Code extension for zero-dependency install
2. Used standalone for air-gapped deployments
3. Run as a service on remote machines
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_platform_suffix() -> str:
    """Get platform-specific suffix for binary name."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        return f"macos-{machine}"
    elif system == "windows":
        return f"windows-{machine}"
    else:
        return f"linux-{machine}"


def clean_build(dist_dir: Path, build_dir: Path) -> None:
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print(f"  Removed {dist_dir}")

    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"  Removed {build_dir}")

    # Clean PyInstaller cache
    pycache = Path("__pycache__")
    if pycache.exists():
        shutil.rmtree(pycache)

    # Clean .spec file if exists
    spec_file = Path("victor-server.spec")
    if spec_file.exists():
        spec_file.unlink()
        print(f"  Removed {spec_file}")


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import PyInstaller

        print(f"PyInstaller version: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("ERROR: PyInstaller not installed")
        print("Install with: pip install pyinstaller")
        return False


def get_hidden_imports() -> list[str]:
    """Get list of hidden imports for PyInstaller."""
    return [
        # Victor modules
        "victor",
        "victor.ui",
        "victor.ui.cli",
        "victor.ui.commands",
        "victor.agent",
        "victor.agent.orchestrator",
        "victor.agent.tool_executor",
        "victor.agent.tool_calling",
        "victor.agent.tool_calling.adapters",
        "victor.agent.modes",
        "victor.agent.model_switcher",
        "victor.agent.change_tracker",
        "victor.agent.conversation_embedding_store",
        "victor.agent.conversation_memory",
        "victor.agent.conversation_controller",
        "victor.providers",
        "victor.providers.anthropic_provider",
        "victor.providers.openai_provider",
        "victor.providers.ollama_provider",
        "victor.providers.google_provider",
        "victor.providers.xai_provider",
        "victor.tools",
        "victor.tools.filesystem",
        "victor.tools.git_tool",
        "victor.tools.code_search",
        "victor.tools.semantic_selector",
        "victor.mcp",
        "victor.mcp.server",
        "victor.config",
        "victor.config.settings",
        "victor.evaluation",
        "victor.evaluation.benchmarks",
        "victor.evaluation.protocol",
        # Dependencies
        "aiohttp",
        "aiohttp.web",
        "httpx",
        "pydantic",
        "pydantic_settings",
        "yaml",
        "tiktoken",
        "tiktoken_ext",
        "tiktoken_ext.openai_public",
        # Async
        "asyncio",
        "aiofiles",
        # Other
        "jsonschema",
        "pygments",
    ]
    # NOTE: sentence_transformers/transformers/torch/lancedb/pyarrow/docker
    # were pruned — none are core deps any more (lean-default move), so these
    # hidden imports only produced "not found" warnings that masked real
    # misses. numpy is intentionally NOT excluded: the bytecode scan pulls it
    # via the numpy-optional lazy sites, and bundling ~30MB is safer than a
    # crash in an unexercised optional path.


def get_data_files() -> list[tuple[str, str]]:
    """Get data files to include in the bundle."""
    data_files = []

    # Config files
    config_dir = Path("victor/config")
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            data_files.append((str(config_file), "victor/config"))

    # Examples
    examples_dir = Path("examples")
    if examples_dir.exists():
        data_files.append((str(examples_dir / "profiles.yaml.example"), "examples"))

    return data_files


def build_binary(
    onefile: bool = False, debug: bool = False, output_name: str = "victor-server"
) -> Path | None:
    """Build the Victor binary.

    Args:
        onefile: Create single-file executable
        debug: Include debug info
        output_name: Name of output binary

    Returns:
        Path to built binary or None on failure
    """
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Prepare PyInstaller command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        output_name,
        "--noconfirm",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    if not debug:
        cmd.append("--clean")

    # Add hidden imports
    for hidden_import in get_hidden_imports():
        cmd.extend(["--hidden-import", hidden_import])

    # Add data files
    for src, dest in get_data_files():
        cmd.extend(["--add-data", f"{src}:{dest}"])

    # Exclude unnecessary packages to reduce size
    excludes = [
        "matplotlib",
        "scipy",
        "PIL",
        "cv2",
        "numpy.distutils",
        "pytest",
        "sphinx",
        "IPython",
        "jupyter",
        "notebook",
    ]
    for exclude in excludes:
        cmd.extend(["--exclude-module", exclude])

    # Entry point - use the CLI module
    cmd.append("victor/ui/cli.py")

    print("Building Victor binary...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)

        # Find the output
        dist_dir = project_root / "dist"
        if onefile:
            binary_path = dist_dir / output_name
            if platform.system() == "Windows":
                binary_path = binary_path.with_suffix(".exe")
        else:
            binary_path = dist_dir / output_name / output_name
            if platform.system() == "Windows":
                binary_path = binary_path.with_suffix(".exe")

        if binary_path.exists():
            print(f"\nSuccess! Binary built at: {binary_path}")
            print(f"Size: {binary_path.stat().st_size / (1024*1024):.1f} MB")
            return binary_path
        else:
            print(f"\nWARNING: Build completed but binary not found at {binary_path}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Build failed with exit code {e.returncode}")
        return None


def create_launcher_script(binary_path: Path) -> None:
    """Create launcher scripts for the binary."""
    dist_dir = binary_path.parent

    # Unix launcher
    launcher_unix = dist_dir / "victor-serve"
    launcher_unix.write_text(f"""#!/bin/bash
# Victor Server Launcher
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/{binary_path.name}" serve "$@"
""")
    launcher_unix.chmod(0o755)

    # Windows launcher
    launcher_win = dist_dir / "victor-serve.bat"
    launcher_win.write_text(f"""@echo off
REM Victor Server Launcher
set DIR=%~dp0
"%DIR%{binary_path.name}" serve %*
""")

    print("Created launcher scripts:")
    print(f"  - {launcher_unix}")
    print(f"  - {launcher_win}")


def main():
    parser = argparse.ArgumentParser(description="Build Victor standalone binary")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Create single-file executable (slower startup, easier distribution)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean build artifacts before building"
    )
    parser.add_argument("--debug", action="store_true", help="Include debug information")
    parser.add_argument(
        "--output",
        default="victor-server",
        help="Output binary name (default: victor-server)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"

    print("=" * 60)
    print("Victor Binary Builder")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Output: {args.output}")
    print(f"Mode: {'onefile' if args.onefile else 'directory'}")
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Clean if requested
    if args.clean:
        clean_build(dist_dir, build_dir)
        print()

    # Build
    binary_path = build_binary(onefile=args.onefile, debug=args.debug, output_name=args.output)

    if binary_path:
        create_launcher_script(binary_path)
        print()
        print("=" * 60)
        print("Build Complete!")
        print("=" * 60)
        print()
        print("To use:")
        print(f"  ./dist/{args.output}/victor-serve --port 8765")
        print()
        print("For VS Code extension:")
        print(f"  Copy dist/{args.output}/ to extension resources/")
        print("  Set victor.victorPath in VS Code settings")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
