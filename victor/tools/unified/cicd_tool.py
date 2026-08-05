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

"""Unified ``cicd`` command tool — bash-style pipeline-config surface.

Parses ``cicd list|generate|validate`` and delegates to
``victor_devops.cicd(operation=…)`` when that package is importable. There is
no plain-shell equivalent, so when the vertical is absent the tool returns a
graceful message. Advertised only in the DevOps vertical.

Example commands:
    cicd list
    cicd generate --workflow python-test
    cicd validate --file .github/workflows/ci.yml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Tuple

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified._vertical_resolver import resolve_vertical_callable
from victor.tools.unified.parser import split_command


class UnifiedCicdParser(argparse.ArgumentParser):
    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_cicd_parser() -> UnifiedCicdParser:
    parser = UnifiedCicdParser(
        prog="cicd", description="Unified CI/CD operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    subparsers.add_parser("list", help="List available workflow templates")

    gen = subparsers.add_parser("generate", help="Generate a CI/CD config")
    gen.add_argument("--workflow", default=None, help="Template name (e.g. python-test)")
    gen.add_argument("--type", default=None, help="Shortcut (test|build|deploy|release|publish)")
    gen.add_argument("--platform", default="github", help="github|gitlab|circleci")
    gen.add_argument("--file", default=None, help="Output file path")

    val = subparsers.add_parser("validate", help="Validate a CI/CD config")
    val.add_argument("--file", default=None, help="Config file to validate")
    val.add_argument("--command", default=None, help="Validation command to run")

    return parser


def _map_kwargs(parsed: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    sub = parsed.subcommand
    if sub == "list":
        return "list", {}
    if sub == "generate":
        return "generate", {
            "platform": parsed.platform,
            "workflow": parsed.workflow,
            "type": parsed.type,
            "file": parsed.file,
        }
    if sub == "validate":
        return "validate", {"file": parsed.file, "validate_command": parsed.command}
    raise ValueError(f"Unknown cicd subcommand '{sub}'")


def _format_result(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("success") is False:
            return f"### ❌ ERROR\n{result.get('error') or 'cicd op failed'}"
        return str(result.get("output") or result)
    return str(result)


@tool(
    name="cicd",
    category="cicd",
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.READ_ONLY,
    priority=Priority.MEDIUM,
    keywords=["cicd", "ci/cd", "pipeline", "github actions", "workflow", "gitlab"],
    task_types=["action", "generation"],
)
async def cicd_tool(cmd: str) -> str:
    """CI/CD domain (bash-style): list, generate, validate.
    Delegates to victor-devops (no plain-shell equivalent). e.g. cicd list · cicd generate --workflow python-test.
    """
    parser = create_cicd_parser()
    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "cicd":
            args_list = args_list[1:]
        parsed = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error parsing command: {e}"

    if not parsed.subcommand:
        return "### ❌ ERROR\nNo cicd subcommand given. Use: cicd list|generate|validate"

    cicd_fn, _src = resolve_vertical_callable(
        "cicd", fallback_module="victor_devops.tools.cicd_tool", fallback_attr="cicd"
    )
    if cicd_fn is None:
        return (
            "### ❌ ERROR\nCI/CD operations require the victor-devops package, which is not "
            "installed. Install victor-devops to generate/validate CI/CD pipelines."
        )
    try:
        operation, kwargs = _map_kwargs(parsed)
        return _format_result(await cicd_fn(operation=operation, **kwargs))
    except Exception as e:
        return f"### ❌ ERROR\ncicd {parsed.subcommand} failed: {e}"


__all__ = ["cicd_tool", "create_cicd_parser"]
