# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Unified ``docker`` command tool — bash-style container/image surface.

Parses ``docker ps|logs|start|stop|restart|exec|inspect|images|stats|rm|pull|
networks|volumes`` and delegates to ``victor_devops.docker(operation=…)`` when
that package is importable (entry-point keyed under ``victor.tool_callables``),
falling back to a shell-driven ``docker`` invocation otherwise. Advertised only
in the DevOps vertical, so it costs no base-schema tokens otherwise.

Example commands:
    docker ps
    docker logs myapp
    docker stop myapp
    docker exec myapp "env"
    docker images
    docker pull nginx
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified._vertical_resolver import resolve_vertical_callable
from victor.tools.unified.parser import split_command

_READ_ONLY_OPS = {"ps", "logs", "inspect", "images", "stats", "networks", "volumes"}


class UnifiedDockerParser(argparse.ArgumentParser):
    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_docker_parser() -> UnifiedDockerParser:
    parser = UnifiedDockerParser(
        prog="docker", description="Unified Docker operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    for op in ("ps", "images", "stats", "networks", "volumes"):
        subparsers.add_parser(op, help=f"Docker {op}")

    for op in ("logs", "start", "stop", "restart", "rm", "inspect"):
        p = subparsers.add_parser(op, help=f"Docker {op}")
        p.add_argument("resource_id", help="Container id/name")

    exec_p = subparsers.add_parser("exec", help="Run a command in a container")
    exec_p.add_argument("resource_id", help="Container id/name")
    exec_p.add_argument("command", help="Command to run")

    pull = subparsers.add_parser("pull", help="Pull an image")
    pull.add_argument("resource_id", help="Image name")

    return parser


def _map_kwargs(parsed: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """Map parsed bash subcommand args to ``victor_devops.docker(operation=…)`` kwargs."""
    sub = parsed.subcommand
    if sub in {"ps", "images", "stats", "networks", "volumes"}:
        return sub, {}
    if sub == "exec":
        return "exec", {
            "resource_id": parsed.resource_id,
            "options": {"command": parsed.command},
        }
    if sub == "pull":
        return "pull", {"resource_id": parsed.resource_id, "resource_type": "image"}
    if sub in {"logs", "start", "stop", "restart", "rm", "inspect"}:
        return sub, {"resource_id": parsed.resource_id}
    raise ValueError(f"Unknown docker subcommand '{sub}'")


def _fallback_argv(parsed: argparse.Namespace) -> Tuple[str, List[str], bool]:
    """Build (subcommand, argv, readonly) for the shell fallback path."""
    sub = parsed.subcommand
    readonly = sub in _READ_ONLY_OPS
    if sub in {"ps", "images", "stats", "networks", "volumes"}:
        return sub, [], readonly
    if sub == "exec":
        return "exec", [parsed.resource_id, parsed.command], False
    if sub == "pull":
        return "pull", [parsed.resource_id], False
    return sub, [getattr(parsed, "resource_id", "")], readonly


def _format_result(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("success") is False:
            return (
                f"### ❌ ERROR\n{result.get('error') or result.get('output') or 'docker op failed'}"
            )
        return str(result.get("output") or result)
    return str(result)


async def _shell_docker(subcommand: str, argv: List[str], *, readonly: bool) -> str:
    from victor.tools.bash import shell

    cmd_parts = ["docker", subcommand, *[str(a) for a in argv]]
    result = await shell(cmd=" ".join(shlex.quote(p) for p in cmd_parts), readonly=readonly)
    if isinstance(result, dict):
        stdout = (result.get("stdout") or result.get("output") or "").strip()
        stderr = (result.get("stderr") or result.get("error") or "").strip()
        if result.get("success") is False:
            return f"### ❌ ERROR\n{stderr or stdout or 'docker command failed'}"
        return stdout or stderr or "Done."
    return str(result)


@tool(
    name="docker",
    category="docker",
    access_mode=AccessMode.EXECUTE,
    danger_level=DangerLevel.HIGH,
    execution_category=ExecutionCategory.EXECUTE,
    priority=Priority.MEDIUM,
    keywords=["docker", "container", "image", "compose"],
    task_types=["action"],
)
async def docker_tool(cmd: str) -> str:
    """Unified Docker tool with bash-like syntax. Delegates to the victor-devops
    docker implementation when available; otherwise runs plain ``docker`` via
    the shell tool. Examples: ``docker ps``, ``docker logs myapp``.
    """
    parser = create_docker_parser()
    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "docker":
            args_list = args_list[1:]
        parsed = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error parsing command: {e}"

    if not parsed.subcommand:
        return (
            "### ❌ ERROR\nNo docker subcommand given. Use: docker ps|logs|start|stop|exec|images|…"
        )

    docker_fn, _src = resolve_vertical_callable(
        "docker",
        fallback_module="victor_devops.tools.docker_tool",
        fallback_attr="docker",
    )
    if docker_fn is not None:
        try:
            operation, kwargs = _map_kwargs(parsed)
            return _format_result(await docker_fn(operation=operation, **kwargs))
        except Exception as e:
            return f"### ❌ ERROR\ndocker {parsed.subcommand} failed: {e}"

    try:
        subcommand, argv, readonly = _fallback_argv(parsed)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    try:
        return await _shell_docker(subcommand, argv, readonly=readonly)
    except Exception as e:
        return f"### ❌ ERROR\ndocker {parsed.subcommand} failed: {e}"


__all__ = ["docker_tool", "create_docker_parser"]
