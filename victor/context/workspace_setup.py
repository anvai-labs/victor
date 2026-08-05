# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Workspace setup utilities for project dependency management.

Framework-level service that ensures a project workspace is properly
configured for tool execution. This includes:
- Auto-installing Python projects that need `pip install -e .`
- Detecting build system (setuptools, poetry, flit, hatch)
- Installing without heavy dependencies (--no-deps by default)

Used by:
- Benchmark harness (SWE-bench task evaluation)
- Orchestrator.set_workspace() (regular agentic workflows)
- Any tool that needs the project importable
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def ensure_project_importable(
    project_name: str,
    project_root: Path,
    install_deps: bool = False,
    timeout: int = 120,
) -> bool:
    """Ensure a Python project is importable in the current environment.

    If the project can already be imported, this is a no-op.
    Otherwise, attempts `pip install -e .` (editable install).

    Args:
        project_name: Python package name (e.g., "django", "astropy")
        project_root: Path to project root (containing setup.py/pyproject.toml)
        install_deps: If True, install dependencies too (default: False for speed)
        timeout: Maximum seconds for installation

    Returns:
        True if project is importable after this call, False otherwise
    """
    # Check if already properly INSTALLED — not merely importable. A bare
    # __import__ succeeds for a project whose source dir is on sys.path even
    # when its compiled C-extensions aren't built (e.g. astropy), which then
    # yields "0 tests collected". So:
    #   - pip-installed distribution (extensions built) → done.
    #   - importable but NOT pip-installed AND has a build system (setup.py/
    #     pyproject/setup.cfg) → it's a source namespace that needs building;
    #     fall through to `pip install -e .`.
    #   - importable with no build system (stdlib, pure source) → nothing to
    #     build, done.
    try:
        import importlib.metadata

        importlib.metadata.distribution(project_name)  # raises if not pip-installed
        __import__(project_name)
        return True
    except (ImportError, importlib.metadata.PackageNotFoundError, ValueError):
        pass

    # Importable but not pip-installed: only force a build if the project HAS a
    # build system (it may have unbuilt C-extensions). Stdlib / build-less
    # packages are genuinely ready as-is.
    try:
        __import__(project_name)
        has_build_system = (
            (project_root / "setup.py").exists()
            or (project_root / "pyproject.toml").exists()
            or (project_root / "setup.cfg").exists()
        )
        if not has_build_system:
            return True
        # Has a build system + not pip-installed → needs building; fall through.
    except ImportError:
        pass

    # Check for build system
    setup_py = project_root / "setup.py"
    pyproject = project_root / "pyproject.toml"
    setup_cfg = project_root / "setup.cfg"

    if not (setup_py.exists() or pyproject.exists() or setup_cfg.exists()):
        logger.debug("No build system found in %s, skipping install", project_root)
        return False

    # Install the project
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        str(project_root),
        "--quiet",
    ]
    if not install_deps:
        cmd.append("--no-deps")

    logger.info("Installing %s for workspace setup...", project_name)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        if proc.returncode == 0:
            logger.info("Installed %s successfully", project_name)
            return True
        logger.warning(
            "Failed to install %s (exit %d): %s",
            project_name,
            proc.returncode,
            stderr.decode()[:300],
        )
        await _uninstall_partial(project_name)
        return False

    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("Installation of %s timed out after %ds", project_name, timeout)
        await _uninstall_partial(project_name)
        return False
    except Exception as e:
        logger.warning("Installation of %s failed: %s", project_name, e)
        await _uninstall_partial(project_name)
        return False


async def _uninstall_partial(project_name: str) -> None:
    """Best-effort uninstall of a partial/failed editable install.

    A failed ``pip install -e .`` (e.g. C-extension build failure) can leave a
    broken namespace shadow in site-packages (``import`` succeeds but
    ``__file__`` is None / ``__version__`` missing), which then breaks
    subsequent tasks' imports. Removing it keeps one failed task from poisoning
    the rest of a benchmark run.
    """
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", project_name, "--quiet"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=60)
        logger.info("Cleaned up partial install of %s", project_name)
    except Exception as e:
        logger.debug("Best-effort uninstall of %s skipped: %s", project_name, e)


def detect_project_name(project_root: Path) -> Optional[str]:
    """Detect the Python package name from a project root.

    Checks pyproject.toml, setup.cfg, and directory structure.

    Args:
        project_root: Path to project root

    Returns:
        Package name or None if not detectable
    """
    # Try pyproject.toml
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None

        if tomllib:
            try:
                data = tomllib.loads(pyproject.read_text())
                name = data.get("project", {}).get("name")
                if name:
                    return name.replace("-", "_")
            except Exception:
                pass

    # Try setup.cfg
    setup_cfg = project_root / "setup.cfg"
    if setup_cfg.exists():
        try:
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read(str(setup_cfg))
            name = cfg.get("metadata", "name", fallback=None)
            if name:
                return name.replace("-", "_")
        except Exception:
            pass

    # Fallback: directory name
    return project_root.name.replace("-", "_")
