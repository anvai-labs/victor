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

"""Regression guard for the ``victor.evaluation`` <-> ``victor.framework`` import cycle."""

import subprocess
import sys


def test_evaluation_imports_cleanly_before_framework() -> None:
    """``victor.evaluation`` must import cleanly even when nothing has imported
    ``victor.framework`` yet.

    Regression guard: ``victor/evaluation/runtime_feedback.py`` used to import
    ``RuntimeEvaluationFeedback`` from ``victor.framework`` at *module scope*.
    Importing any ``victor.framework.*`` runs framework's heavy ``__init__``,
    which pulls in the agent runtime and loops back into the half-initialised
    evaluation module — a circular ``ImportError`` that broke
    ``import victor.evaluation.*`` (and the offline calibration runner) whenever
    evaluation was imported before the framework. The shared type now lives in
    ``victor_contracts``; this test runs the import in a *fresh* interpreter
    (where framework is not already loaded by an earlier test) to ensure the
    cycle does not return.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import victor.evaluation.calibration_corpus"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "Importing victor.evaluation before victor.framework failed — the "
        "evaluation<->framework import cycle may have regressed. Keep the shared "
        "runtime-evaluation types in victor_contracts (imported downward) rather "
        "than importing victor.framework at module scope from victor.evaluation:\n"
        + (result.stderr[-2000:] or result.stdout[-2000:])
    )
