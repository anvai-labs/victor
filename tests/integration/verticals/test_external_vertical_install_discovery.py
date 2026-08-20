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

"""Slow integration checks for the contract-only external vertical example."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VICTOR_SDK_DIR = REPO_ROOT / "victor-contracts"
EXTERNAL_EXAMPLE_DIR = REPO_ROOT / "examples" / "external_vertical"


def _venv_bin(venv_dir: Path, executable: str) -> Path:
    """Return a venv executable path across platforms."""

    bin_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv_dir / bin_dir / f"{executable}{suffix}"


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
    """Build a deterministic environment for isolated install checks."""

    home_dir = tmp_path / "home"
    pip_cache_dir = tmp_path / "pip-cache"
    home_dir.mkdir(parents=True, exist_ok=True)
    pip_cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "PIP_CACHE_DIR": str(pip_cache_dir),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return env


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and surface stdout/stderr on failure."""

    completed = subprocess.run(
        args,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"{' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _create_validation_venv(tmp_path: Path, env: dict[str, str]) -> Path:
    """Create an isolated venv for offline local-package installs.

    `--system-site-packages` is required on this Python 3.12 environment so the
    temporary venv can access the local `setuptools` build backend without
    downloading anything from PyPI.
    """

    venv_dir = tmp_path / "venv"
    _run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        cwd=REPO_ROOT,
        env=env,
    )
    return venv_dir


def _pip_install_editable(venv_dir: Path, env: dict[str, str], *paths: Path) -> None:
    """Install local packages into the validation venv without external deps."""

    args = [
        str(_venv_bin(venv_dir, "pip")),
        "install",
        "--no-build-isolation",
        "--no-deps",
    ]
    for path in paths:
        args.extend(["-e", str(path)])
    _run(args, cwd=REPO_ROOT, env=env)


def _pip_install(venv_dir: Path, env: dict[str, str], *artifacts: Path) -> None:
    """Install local wheel artifacts into the validation venv without external deps."""

    args = [
        str(_venv_bin(venv_dir, "pip")),
        "install",
        "--no-build-isolation",
        "--no-deps",
    ]
    args.extend(str(artifact) for artifact in artifacts)
    _run(args, cwd=REPO_ROOT, env=env)


def _build_wheel(venv_dir: Path, env: dict[str, str], package_dir: Path, output_dir: Path) -> Path:
    """Build a wheel for a local package without reaching out to PyPI."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(_venv_bin(venv_dir, "pip")),
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            str(package_dir),
            "-w",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    wheels = sorted(output_dir.glob("*.whl"))
    assert wheels, f"No wheel built for {package_dir}"
    return wheels[-1]


def _run_python(
    venv_dir: Path, env: dict[str, str], script: str
) -> subprocess.CompletedProcess[str]:
    """Execute Python in the validation venv."""

    return _run(
        [str(_venv_bin(venv_dir, "python")), "-c", script],
        cwd=REPO_ROOT,
        env=env,
    )


def _json_from_stdout(stdout: str) -> dict[str, object]:
    """Parse the final JSON line emitted by the validation subprocess."""

    return json.loads(stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_external_vertical_sdk_only_install_exposes_entry_point_and_definition(
    tmp_path: Path,
) -> None:
    """The example should install cleanly and expose its SDK definition contract."""

    env = _subprocess_env(tmp_path)
    venv_dir = _create_validation_venv(tmp_path, env)
    wheel_path = _build_wheel(
        venv_dir,
        env,
        EXTERNAL_EXAMPLE_DIR,
        tmp_path / "dist" / "external-sdk-only",
    )
    _pip_install_editable(venv_dir, env, VICTOR_SDK_DIR)
    _pip_install(venv_dir, env, wheel_path)

    completed = _run_python(
        venv_dir,
        env,
        textwrap.dedent("""
            import json
            from importlib.metadata import entry_points
            from victor_contracts.discovery import collect_verticals_from_candidate

            eps = {ep.name: ep for ep in entry_points(group="victor.plugins")}
            assert "security" in eps, sorted(eps)
            plugin = eps["security"].load()
            discovered = collect_verticals_from_candidate(plugin)
            assert list(discovered) == ["security"]
            assistant_cls = discovered["security"]
            definition = assistant_cls.get_definition()
            print(
                json.dumps(
                    {
                        "name": definition.name,
                        "tool_requirements": [
                            requirement.tool_name for requirement in definition.tool_requirements
                        ],
                        "capability_requirements": [
                            requirement.capability_id
                            for requirement in definition.capability_requirements
                        ],
                        "teams": [team.team_id for team in definition.team_metadata.teams],
                        "default_team": definition.team_metadata.default_team,
                    }
                )
            )
            """).strip(),
    )

    payload = _json_from_stdout(completed.stdout)
    assert payload == {
        "name": "security",
        "tool_requirements": [
            "read",
            "ls",
            "code_search",
            "overview",
            "shell",
            "web_search",
            "write",
        ],
        "capability_requirements": ["file_ops", "git", "web_access"],
        "teams": ["security_review_team"],
        "default_team": "security_review_team",
    }


@pytest.mark.slow
def test_external_vertical_runtime_install_is_discoverable_by_vertical_loader(
    tmp_path: Path,
) -> None:
    """The example should remain discoverable by Victor after local runtime install."""

    env = _subprocess_env(tmp_path)
    venv_dir = _create_validation_venv(tmp_path, env)
    wheel_path = _build_wheel(
        venv_dir,
        env,
        EXTERNAL_EXAMPLE_DIR,
        tmp_path / "dist" / "external-runtime",
    )
    _pip_install_editable(venv_dir, env, VICTOR_SDK_DIR, REPO_ROOT)
    _pip_install(venv_dir, env, wheel_path)

    entry_point_cache_dir = tmp_path / "victor-cache"
    completed = _run_python(
        venv_dir,
        env,
        textwrap.dedent(f"""
            import json
            from pathlib import Path

            from victor.core.verticals.base import VerticalRegistry
            from victor.core.verticals.vertical_loader import VerticalLoader
            from victor.framework.module_loader import EntryPointCache

            EntryPointCache.reset_instance()
            EntryPointCache.get_instance(cache_dir=Path(r"{entry_point_cache_dir}"))
            VerticalRegistry.reset_discovery()

            loader = VerticalLoader()
            discovered = loader.discover_verticals(force_refresh=True)
            vertical = loader.load("security")
            definition = vertical.get_definition()
            team_specs = vertical.get_team_specs()

            print(
                json.dumps(
                    {{
                        "discovered": sorted(discovered),
                        "active_vertical": loader.active_vertical_name,
                        "definition_name": definition.name,
                        "tool_requirements": [
                            requirement.tool_name for requirement in definition.tool_requirements
                        ],
                        "team_specs": sorted(team_specs),
                        "default_team": definition.team_metadata.default_team,
                    }}
                )
            )
            """).strip(),
    )

    payload = _json_from_stdout(completed.stdout)
    assert "security" in payload["discovered"]
    assert payload["active_vertical"] == "security"
    assert payload["definition_name"] == "security"
    assert payload["tool_requirements"] == [
        "read",
        "ls",
        "code_search",
        "overview",
        "shell",
        "web_search",
        "write",
    ]
    assert payload["team_specs"] == ["security_review_team"]
    assert payload["default_team"] == "security_review_team"


@pytest.mark.slow
def test_external_vertical_runtime_load_registers_tool_and_mode_config(
    tmp_path: Path,
) -> None:
    """Framework-path loading should surface the custom tool and mode config."""

    env = _subprocess_env(tmp_path)
    venv_dir = _create_validation_venv(tmp_path, env)
    wheel_path = _build_wheel(
        venv_dir,
        env,
        EXTERNAL_EXAMPLE_DIR,
        tmp_path / "dist" / "external-runtime-extensions",
    )
    _pip_install_editable(venv_dir, env, VICTOR_SDK_DIR, REPO_ROOT)
    _pip_install(venv_dir, env, wheel_path)

    entry_point_cache_dir = tmp_path / "victor-cache"
    completed = _run_python(
        venv_dir,
        env,
        textwrap.dedent(f"""
            import asyncio
            import json
            from importlib.metadata import entry_points
            from pathlib import Path

            from victor.agent.protocols import ToolRegistryProtocol
            from victor.core.container import ServiceContainer
            from victor.core.plugins.context import HostPluginContext
            from victor.core.verticals.base import VerticalRegistry
            from victor.core.verticals.vertical_loader import VerticalLoader
            from victor.framework.module_loader import EntryPointCache
            from victor.tools.registry import ToolRegistry

            EntryPointCache.reset_instance()
            EntryPointCache.get_instance(cache_dir=Path(r"{entry_point_cache_dir}"))
            VerticalRegistry.reset_discovery()

            # Plugin registration through the host PluginContext routes the
            # duck-typed custom tool into the framework ToolRegistry.
            tool_registry = ToolRegistry()
            container = ServiceContainer()
            container.register_instance(ToolRegistryProtocol, tool_registry)
            plugin = {{ep.name: ep for ep in entry_points(group="victor.plugins")}}[
                "security"
            ].load()
            plugin.register(HostPluginContext(container))

            tool = tool_registry.get("secret_pattern_scan")
            assert tool is not None, "custom tool was not registered"
            scan = asyncio.run(
                tool.execute(text='api_key = "abcdef0123456789"')
            )

            # Framework-path vertical load + mode config discovery via the
            # victor.mode_configs entry point.
            loader = VerticalLoader()
            discovered = loader.discover_verticals(force_refresh=True)
            vertical = loader.load("security")
            provider = vertical.get_mode_config_provider()
            assert provider is not None, "mode config provider not discovered"

            print(
                json.dumps(
                    {{
                        "discovered": sorted(discovered),
                        "tool_name": tool.name,
                        "scan_success": scan["success"],
                        "scan_patterns": sorted(
                            {{finding["pattern"] for finding in scan["findings"]}}
                        ),
                        "mode_names": sorted(provider.get_mode_configs()),
                        "default_mode": provider.get_default_mode(),
                        "vulnerability_scan_budget": provider.get_tool_budget_for_task(
                            "vulnerability_scan"
                        ),
                    }}
                )
            )
            """).strip(),
    )

    payload = _json_from_stdout(completed.stdout)
    assert "security" in payload["discovered"]
    assert payload["tool_name"] == "secret_pattern_scan"
    assert payload["scan_success"] is True
    assert payload["scan_patterns"] == ["hardcoded_credential"]
    assert payload["mode_names"] == ["deep", "quick"]
    assert payload["default_mode"] == "quick"
    assert payload["vulnerability_scan_budget"] == 18
