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

"""Unit tests for the offline calibration corpus's workspace-effect verifiers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from victor.evaluation.calibration_corpus import default_corpus

# QA tasks are transcript-verified (no workspace effect); the rest verify the workspace.
_CORPUS = [t for t in default_corpus(variants=2) if t.family != "qa"]
_IDS = [t.task_id for t in _CORPUS]
_CODE_FIX = [t for t in _CORPUS if t.family == "code-fix"]


def _score(task: Any, action: Optional[Callable[[Path], None]]) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        task.setup(ws)
        if action is not None:
            action(ws)
        return float(task.verify(ws, None))


@pytest.mark.parametrize("task", _CORPUS, ids=_IDS)
def test_reference_solution_passes(task: Any) -> None:
    assert _score(task, task.solve) == 1.0


@pytest.mark.parametrize("task", _CORPUS, ids=_IDS)
def test_flawed_solution_does_not_pass(task: Any) -> None:
    assert _score(task, task.solve_flawed) < 1.0


@pytest.mark.parametrize("task", _CODE_FIX, ids=[t.task_id for t in _CODE_FIX])
def test_code_fix_unchanged_buggy_module_fails(task: Any) -> None:
    assert _score(task, None) == 0.0


@pytest.mark.parametrize("task", _CODE_FIX, ids=[t.task_id for t in _CODE_FIX])
def test_code_fix_unimportable_module_scores_zero(task: Any) -> None:
    def _break(ws: Path) -> None:
        for path in ws.glob("*.py"):
            path.write_text("def sum_upto(n:\n")

    assert _score(task, _break) == 0.0


def test_corpus_families_and_validation() -> None:
    assert len({t.family for t in default_corpus(1)}) == 5
    assert len(default_corpus(2)) == 2 * len(default_corpus(1))
    with pytest.raises(ValueError):
        default_corpus(0)
