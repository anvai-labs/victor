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

"""Unit tests for the effect-prone task corpus (fair effect-gate test)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from victor.evaluation.effect_gate_corpus import effect_gate_corpus

_CORPUS = effect_gate_corpus(variants=1)
_IDS = [t.task_id for t in _CORPUS]


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
def test_declaring_done_without_acting_fails(task: Any) -> None:
    # The defining property: with NO workspace effect, verify must fail. This is exactly the
    # "declare success, did nothing" failure the effect gate exists to catch.
    assert _score(task, None) == 0.0


@pytest.mark.parametrize("task", _CORPUS, ids=_IDS)
def test_flawed_solution_does_not_pass(task: Any) -> None:
    assert _score(task, task.solve_flawed) < 1.0


def test_corpus_size_families_and_determinism() -> None:
    assert len(effect_gate_corpus(2)) == 12  # 6 families × 2 variants
    assert len({t.family for t in _CORPUS}) == 6
    assert [t.task_id for t in effect_gate_corpus(2)] == [t.task_id for t in effect_gate_corpus(2)]
    with pytest.raises(ValueError):
        effect_gate_corpus(0)
