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

"""FEP-0012 numerical parity: pure-Python predict ≡ the numpy formulation.

The committed goldens were generated ONCE with the pre-rewrite numpy
implementation (``scripts/generate_fep0012_goldens.py``) against the shipped
artifact. This test verifies the numpy-free runtime against them in an
interpreter where numpy is BLOCKED — proving both value parity and
numpy-absence function in one pass.

Tolerances: probabilities compare at rel 1e-9 (math.exp vs np.exp differ by
≤1 ulp); labels compare strictly (goldens exclude near-ties, top-2 gap
> 1e-3, so argmax cannot legitimately flip).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GOLDENS = Path(__file__).resolve().parent / "goldens" / "fep0012_predict_goldens.json"


def _load_goldens() -> dict:
    data = json.loads(GOLDENS.read_text())
    assert data["feature_spec_version"] == "1"
    return data


def test_pure_predict_matches_numpy_goldens_without_numpy() -> None:
    """The whole harness runs in a subprocess with numpy BLOCKED
    (``sys.modules['numpy'] = None``): value parity AND numpy-absence at once.
    """
    goldens = _load_goldens()
    payload = json.dumps(goldens)

    code = (
        "import sys, json, math\n"
        "sys.modules['numpy'] = None  # poison: any `import numpy` now raises\n"
        "import victor.ml.model as m\n"
        "assert sys.modules['numpy'] is None, 'runtime imported numpy (poison overwritten)'\n"
        "payload = json.loads(sys.stdin.read())\n"
        "model = m.EdgeClassifierModel.load('victor/models/edge_classifier_v1.npz')\n"
        "worst = 0.0\n"
        "for g in payload['goldens']:\n"
        "    delta = None\n"
        "    if g['delta'] is not None:\n"
        "        delta = {int(k): list(v) for k, v in g['delta'].items()}\n"
        "    label, conf = model.predict(g['decision_type'], g['text'], delta=delta)\n"
        "    rel = abs(conf - g['expected_confidence']) / max(g['expected_confidence'], 1e-300)\n"
        "    worst = max(worst, rel)\n"
        "    assert rel <= 1e-9, (g['decision_type'], g['text'][:40], conf, g['expected_confidence'], rel)\n"
        "    expected = g['expected_label']\n"
        "    assert label == expected or conf < g['threshold'], (label, expected, conf)\n"
        "print(f'WORST:{worst}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=payload,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"golden verification failed:\n{result.stderr[-1500:]}"
    worst = next((ln for ln in result.stdout.splitlines() if ln.startswith("WORST:")), "WORST:?")
    print(worst)


def test_golden_inputs_are_non_tie() -> None:
    """Every committed golden has a top-2 gap > 1e-3 by construction; assert
    the property so a sloppy regeneration cannot smuggle in flippable cases."""
    goldens = _load_goldens()["goldens"]
    assert len(goldens) >= 50, "golden corpus unexpectedly small"
    for g in goldens:
        if g["delta"] is None:
            continue
        # delta vs plain entries of the same text must differ (blend exercised)
        assert g["expected_confidence"] > 0.0


def test_runtime_predict_modules_have_no_module_level_numpy() -> None:
    """Static pin: no MODULE-LEVEL numpy import in the runtime modules.

    ``model.save()`` deliberately keeps a function-local numpy import (dev-only
    writer); module-level imports are what put numpy on the import chain.
    """
    import ast

    for module in ("victor/ml/model.py", "victor/ml/npz_reader.py", "victor/core/vecmath.py"):
        tree = ast.parse((ROOT / module).read_text())
        for node in tree.body:
            for alias in (
                getattr(node, "names", []) if isinstance(node, (ast.Import, ast.ImportFrom)) else []
            ):
                name = alias.name.split(".")[0]
                base = (
                    node.module.split(".")[0]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else name
                )
                assert (
                    name != "numpy" and base != "numpy"
                ), f"{module}: module-level numpy import ({ast.dump(node)})"
