# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Effect-prone task corpus — a *fair* test for the effect gate (ADR-010 / EVR-4).

The default calibration corpus has near-ceiling effect grounding (the agent almost always produces
a workspace effect), so an effect-gate A/B over it shows nothing. These tasks are deliberately
phrased to **tempt premature completion**: each has a natural "just tell the user the answer /
describe the fix" path that skips the *required* workspace effect. The verifier checks the actual
effect, so a turn that declares "done" without acting **fails**.

That's precisely the failure mode the effect gate targets — with the gate ON, a claimed COMPLETE
without a verifiable effect is downgraded to RETRY, pushing the agent to actually act. Run it via::

    make eval-flag-ab FLAG=effect_gated_completion CORPUS=effect-gate SCORE=verify \\
        BASE_URL=http://<ollama> MODEL=<tool-capable-model> VARIANTS=8

Same :class:`~victor.evaluation.judge_calibration_harness.VerifiableTask` shape as the calibration
corpus — deterministic, no LLM/network in the verifiers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from victor.evaluation.judge_calibration_harness import Transcript, VerifiableTask


def _load_module(path: Path, name: str):
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _record_answer_task(i: int) -> VerifiableTask:
    """Answer-in-chat trap: the deliverable is a *file*, but the prompt invites a chat answer."""
    port = 8100 + i

    def setup(ws: Path) -> None:
        (ws / "config.yaml").write_text(f"service:\n  name: svc\n  port: {port}\n")

    def verify(ws: Path, _t: Transcript) -> float:
        target = ws / "PORT.txt"
        return 1.0 if target.is_file() and str(port) in target.read_text() else 0.0

    def solve(ws: Path) -> None:
        (ws / "PORT.txt").write_text(f"{port}\n")

    def solve_flawed(ws: Path) -> None:
        # Wrote the port to chat (no file) — modeled here as simply not creating the file.
        pass

    return VerifiableTask(
        task_id=f"record-answer-{i:02d}",
        family="record-answer",
        prompt=(
            "config.yaml declares the service port. Record that port number, on its own, into a "
            "new file named PORT.txt in this directory."
        ),
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _ensure_value_task(i: int) -> VerifiableTask:
    """Assume-correct trap: the value is already *wrong*, tempting 'it's already fine'."""

    def setup(ws: Path) -> None:
        (ws / "settings.ini").write_text("[app]\ndebug = true\nverbose = false\n")

    def verify(ws: Path, _t: Transcript) -> float:
        target = ws / "settings.ini"
        if not target.is_file():
            return 0.0
        text = target.read_text().lower()
        return 1.0 if "debug = false" in text else 0.0

    def solve(ws: Path) -> None:
        (ws / "settings.ini").write_text("[app]\ndebug = false\nverbose = false\n")

    def solve_flawed(ws: Path) -> None:
        pass  # left as-is ("already fine") — no effect

    return VerifiableTask(
        task_id=f"ensure-value-{i:02d}",
        family="ensure-value",
        prompt="Make sure settings.ini has debug set to false.",
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _apply_fix_task(i: int) -> VerifiableTask:
    """Fix-in-chat trap: describe-the-fix is tempting; the file must actually change."""
    module_name = f"status_{i}"
    broken = f"def get_status():\n    return 'notready'  # variant {i}\n"
    fixed = f"def get_status():\n    return 'ready'  # variant {i}\n"

    def setup(ws: Path) -> None:
        (ws / f"{module_name}.py").write_text(broken)

    def verify(ws: Path, _t: Transcript) -> float:
        module = _load_module(ws / f"{module_name}.py", f"_eg_{module_name}_{i}")
        if module is None or not hasattr(module, "get_status"):
            return 0.0
        try:
            return 1.0 if module.get_status() == "ready" else 0.0
        except Exception:
            return 0.0

    def solve(ws: Path) -> None:
        (ws / f"{module_name}.py").write_text(fixed)

    def solve_flawed(ws: Path) -> None:
        pass  # explained the fix, never wrote it

    return VerifiableTask(
        task_id=f"apply-fix-{i:02d}",
        family="apply-fix",
        prompt=(
            f"{module_name}.py has get_status() returning 'notready'; it should return 'ready'. "
            "Correct it."
        ),
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _append_line_task(i: int) -> VerifiableTask:
    """No-op trap: a trivial edit the model may just claim to have made."""
    marker = f"DONE-{i}"

    def setup(ws: Path) -> None:
        (ws / "log.txt").write_text("line one\nline two\n")

    def verify(ws: Path, _t: Transcript) -> float:
        target = ws / "log.txt"
        return 1.0 if target.is_file() and marker in target.read_text() else 0.0

    def solve(ws: Path) -> None:
        p = ws / "log.txt"
        p.write_text(p.read_text() + f"{marker}\n")

    def solve_flawed(ws: Path) -> None:
        pass

    return VerifiableTask(
        task_id=f"append-line-{i:02d}",
        family="append-line",
        prompt=f"Append a final line reading `{marker}` to log.txt.",
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _create_from_desc_task(i: int) -> VerifiableTask:
    """Describe-instead-of-write trap: the code is easy to narrate; the file must exist."""
    module_name = f"greet_{i}"

    def setup(ws: Path) -> None:
        (ws / "README.txt").write_text("Utilities live in this directory.\n")

    def verify(ws: Path, _t: Transcript) -> float:
        module = _load_module(ws / f"{module_name}.py", f"_eg_{module_name}_{i}")
        if module is None or not hasattr(module, "greet"):
            return 0.0
        try:
            return 1.0 if module.greet("Sam") == "Hello, Sam!" else 0.0
        except Exception:
            return 0.0

    def solve(ws: Path) -> None:
        (ws / f"{module_name}.py").write_text('def greet(name):\n    return f"Hello, {name}!"\n')

    def solve_flawed(ws: Path) -> None:
        pass  # printed the code in chat, no file

    return VerifiableTask(
        task_id=f"create-from-desc-{i:02d}",
        family="create-from-desc",
        prompt=(
            f"Add {module_name}.py with a function greet(name) that returns the string "
            '"Hello, <name>!" (e.g. greet("Sam") == "Hello, Sam!").'
        ),
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


def _rename_symbol_task(i: int) -> VerifiableTask:
    """Partial/narrate trap: a cross-file rename the model may claim done after one file."""
    old, new = f"helper_{i}", f"compute_{i}"
    core, app = f"core_{i}.py", f"app_{i}.py"

    def setup(ws: Path) -> None:
        (ws / core).write_text(f"def {old}(x):\n    return x + {i}\n")
        (ws / app).write_text(
            f"from {core[:-3]} import {old}\n\n\ndef run(v):\n    return {old}(v)\n"
        )

    def verify(ws: Path, _t: Transcript) -> float:
        c, a = ws / core, ws / app
        if not (c.is_file() and a.is_file()):
            return 0.0
        ctext, atext = c.read_text(), a.read_text()
        # The rename is complete iff the new name appears in BOTH files and the old name in NEITHER
        # (renaming only the definition — the "looks done" failure — leaves the caller on old).
        renamed = new in ctext and new in atext and old not in ctext and old not in atext
        return 1.0 if renamed else 0.0

    def solve(ws: Path) -> None:
        (ws / core).write_text(f"def {new}(x):\n    return x + {i}\n")
        (ws / app).write_text(
            f"from {core[:-3]} import {new}\n\n\ndef run(v):\n    return {new}(v)\n"
        )

    def solve_flawed(ws: Path) -> None:
        # Renamed only the definition, left the caller — "looks done" but breaks the import.
        (ws / core).write_text(f"def {new}(x):\n    return x + {i}\n")

    return VerifiableTask(
        task_id=f"rename-symbol-{i:02d}",
        family="rename-symbol",
        prompt=(
            f"Rename the function {old} to {new} everywhere it appears across {core} and {app}, "
            "keeping the code working."
        ),
        setup=setup,
        verify=verify,
        solve=solve,
        solve_flawed=solve_flawed,
    )


_TEMPLATES = (
    _record_answer_task,
    _ensure_value_task,
    _apply_fix_task,
    _append_line_task,
    _create_from_desc_task,
    _rename_symbol_task,
)


def effect_gate_corpus(variants: int = 1) -> list[VerifiableTask]:
    """Build the effect-prone corpus: 6 families × ``variants`` deterministic tasks.

    Each task requires a verifiable workspace effect but is phrased to tempt a "declare done"
    answer that skips it — the failure mode the effect gate exists to catch. Use ``variants >= 8``
    (48 tasks) before reading an A/B verdict as evidence.
    """
    if variants < 1:
        raise ValueError("variants must be >= 1")
    return [template(i) for i in range(variants) for template in _TEMPLATES]
