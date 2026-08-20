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

"""Repo-local verifiable task corpus for offline judge calibration (EVR-2, ADR-011).

Six task families, each with a programmatic verifier that decides completion from the actual
workspace state (or, for QA, the transcript) — no LLM, no network, no subprocess. Every template
is parameterized by a variant index so ``default_corpus(variants=N)`` yields ``6 × N``
deterministic tasks: enough statistical power for κ/α without any data files.

These verifiers are the gold-label source for
:class:`~victor.evaluation.judge_calibration_harness.JudgeCalibrationHarness`. They double as the
seed corpus for the ADR-012 acceptance-oracle work (EVR-5): each task is an executable
specification of "done".

Agreement caveat: with the default ``variants=1`` (6 tasks) α is smoke-level only. Use
``variants >= 8`` (48 tasks) before reading the gate decision as evidence.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

from victor.evaluation.judge_calibration_harness import Transcript, VerifiableTask

_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")


def _load_module(path: Path, name: str):
    """Import a module from a file path without touching sys.path. Returns None on failure."""
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------
# Task templates (each parameterized by a variant index for determinism + statistical power)
# ----------------------------------------------------------------------------------------------------


def _file_create_task(i: int) -> VerifiableTask:
    filename = f"settings_{i}.toml"
    line = f"port = {8000 + i}"

    def setup(ws: Path) -> None:
        (ws / "README.txt").write_text("Service scaffold. Configuration is TOML-based.\n")

    def verify(ws: Path, _t: Transcript) -> float:
        target = ws / filename
        return 1.0 if target.is_file() and line in target.read_text() else 0.0

    def solve(ws: Path) -> None:
        (ws / filename).write_text(f"{line}\n")

    def solve_flawed(ws: Path) -> None:
        # File created (looks done) but with the wrong port value.
        (ws / filename).write_text(f"port = {9000 + i}\n")

    return VerifiableTask(
        task_id=f"file-create-{i:02d}",
        family="file-create",
        prompt=f"Create a file named {filename} containing the line `{line}`.",
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _code_fix_task(i: int) -> VerifiableTask:
    module_name = f"mathutil_{i}"
    buggy = (
        "def sum_upto(n):\n"
        '    """Sum of integers from 1 to n inclusive."""\n'
        "    return sum(range(1, n))  # BUG: excludes n\n"
    )
    fixed = (
        "def sum_upto(n):\n"
        '    """Sum of integers from 1 to n inclusive."""\n'
        "    return sum(range(1, n + 1))\n"
    )

    def setup(ws: Path) -> None:
        (ws / f"{module_name}.py").write_text(buggy)

    def verify(ws: Path, _t: Transcript) -> float:
        module = _load_module(ws / f"{module_name}.py", f"_calib_{module_name}_{i}")
        if module is None:
            return 0.0
        try:
            ok = module.sum_upto(4 + i) == sum(range(1, 5 + i)) and module.sum_upto(1) == 1
        except Exception:
            return 0.0
        return 1.0 if ok else 0.0

    def solve(ws: Path) -> None:
        (ws / f"{module_name}.py").write_text(fixed)

    # Changed the code (looks fixed) but still wrong: overshoots by including n+1.
    flawed = (
        "def sum_upto(n):\n"
        '    """Sum of integers from 1 to n inclusive."""\n'
        "    return sum(range(1, n + 2))\n"
    )

    def solve_flawed(ws: Path) -> None:
        (ws / f"{module_name}.py").write_text(flawed)

    return VerifiableTask(
        task_id=f"code-fix-{i:02d}",
        family="code-fix",
        prompt=(
            f"{module_name}.py: sum_upto(n) must return the sum of 1..n inclusive, "
            f"but it currently excludes n. Fix the bug."
        ),
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _rename_task(i: int) -> VerifiableTask:
    old, new = f"old_helper_{i}", f"compute_result_{i}"
    core, app = f"core_{i}.py", f"app_{i}.py"

    def setup(ws: Path) -> None:
        (ws / core).write_text(f"def {old}(x):\n    return x * 2\n")
        (ws / app).write_text(
            f"from {core[:-3]} import {old}\n\n\ndef run(value):\n    return {old}(value)\n"
        )

    def verify(ws: Path, _t: Transcript) -> float:
        try:
            core_src = (ws / core).read_text()
            app_src = (ws / app).read_text()
            ast.parse(core_src)
            ast.parse(app_src)
        except Exception:
            return 0.0
        combined = core_src + app_src
        renamed = old not in combined and f"def {new}" in core_src and new in app_src
        return 1.0 if renamed else 0.0

    def solve(ws: Path) -> None:
        for name in (core, app):
            path = ws / name
            path.write_text(path.read_text().replace(old, new))

    def solve_flawed(ws: Path) -> None:
        # Renamed in the definition file only — the call site in app still uses the old
        # name, so the rename is incomplete (a classic missed-reference refactor).
        path = ws / core
        path.write_text(path.read_text().replace(old, new))

    return VerifiableTask(
        task_id=f"refactor-rename-{i:02d}",
        family="refactor",
        prompt=f"Rename the function `{old}` to `{new}` everywhere in {core} and {app}.",
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _docs_link_task(i: int) -> VerifiableTask:
    readme = f"README_{i}.md"
    missing, existing = f"docs/setup_{i}.md", f"docs/getting_started_{i}.md"

    def setup(ws: Path) -> None:
        (ws / "docs").mkdir()
        (ws / existing).write_text(f"# Getting started (variant {i})\n")
        (ws / readme).write_text(f"# Project {i}\n\nStart with the [setup guide]({missing}).\n")

    def verify(ws: Path, _t: Transcript) -> float:
        text = (ws / readme).read_text()
        targets = _MD_LINK_RE.findall(text)
        if not targets:
            return 0.0
        return 1.0 if all((ws / target).is_file() for target in targets) else 0.0

    def solve(ws: Path) -> None:
        path = ws / readme
        path.write_text(path.read_text().replace(missing, existing))

    def solve_flawed(ws: Path) -> None:
        # Changed the link (looks fixed) but to another doc that also does not exist.
        path = ws / readme
        path.write_text(path.read_text().replace(missing, f"docs/overview_{i}.md"))

    return VerifiableTask(
        task_id=f"docs-link-{i:02d}",
        family="docs",
        prompt=(
            f"{readme} links to {missing}, which does not exist. "
            f"Fix the link to point at the doc that does."
        ),
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _dead_code_task(i: int) -> VerifiableTask:
    module = f"util_{i}.py"
    dead = f"unused_legacy_{i}"
    kept_src = f"def active_transform(x):\n    return x + {i}\n"

    def setup(ws: Path) -> None:
        (ws / module).write_text(f"{kept_src}\n\ndef {dead}(y):\n    return y - {i}\n")

    def verify(ws: Path, _t: Transcript) -> float:
        try:
            tree = ast.parse((ws / module).read_text())
        except Exception:
            return 0.0
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        return 1.0 if dead not in names and "active_transform" in names else 0.0

    def solve(ws: Path) -> None:
        (ws / module).write_text(kept_src)

    def solve_flawed(ws: Path) -> None:
        # Removed a function (looks done) but the WRONG one — deleted active_transform and
        # left the dead function behind.
        (ws / module).write_text(f"def {dead}(y):\n    return y - {i}\n")

    return VerifiableTask(
        task_id=f"dead-code-{i:02d}",
        family="refactor",
        prompt=f"Remove the unused function `{dead}` from {module}; keep active_transform intact.",
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _qa_task(i: int) -> VerifiableTask:
    config = f"config_{i}.yaml"
    port = 9000 + i

    def setup(ws: Path) -> None:
        (ws / config).write_text(f"service:\n  name: svc-{i}\n  service_port: {port}\n")

    def verify(_ws: Path, transcript: Transcript) -> float:
        return 1.0 if str(port) in transcript.final_message else 0.0

    return VerifiableTask(
        task_id=f"qa-config-{i:02d}",
        family="qa",
        prompt=f"What port does the service listen on, according to {config}? State the number.",
        setup=setup,
        verify=verify,
        solve=lambda ws: None,
        reference_answer=f"The service listens on port {port} (from {config}).",
        # A confident, well-formed answer with the WRONG number (transposed digits).
        reference_answer_flawed=f"The service listens on port {port + 10} (from {config}).",
    )


_TEMPLATES = (
    _file_create_task,
    _code_fix_task,
    _rename_task,
    _docs_link_task,
    _dead_code_task,
    _qa_task,
)


def default_corpus(variants: int = 1) -> list[VerifiableTask]:
    """Build the offline calibration corpus: 6 families × ``variants`` deterministic tasks.

    Tasks are interleaved family-first per variant so scripted executors with periodic
    failure patterns spread outcomes across every family.
    """
    if variants < 1:
        raise ValueError("variants must be >= 1")
    return [template(i) for i in range(variants) for template in _TEMPLATES]
