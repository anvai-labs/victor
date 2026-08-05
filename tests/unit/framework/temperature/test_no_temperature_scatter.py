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

"""Boundary guard (ADR-013, PR-E): no NEW raw temperature arithmetic outside the sanctioned modules.

The unified temperature policy consolidated 6 scattered paths into one resolver. This AST guard keeps
it that way: any ``<…temperature…> +/- <number>`` expression must live in
``victor/framework/temperature/`` or the recovery escalation modules (the latter pending absorption
into the resolver). A new file doing ad-hoc temperature math fails this test — preventing a 7th scatter.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Files allowed to do raw temperature arithmetic. The framework/temperature package owns it; the
# recovery modules are the known escalation sites still pending absorption into the resolver.
_ALLOWLIST = {
    "framework/temperature/",  # the canonical owner (prefix match)
    "agent/recovery/temperature.py",  # ProgressiveTemperatureAdjuster (reactive escalation core)
}

_VICTOR = Path(__file__).resolve().parents[4] / "victor"


def _is_temperature_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "temperature" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return "temperature" in node.attr.lower()
    return False


def _is_number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _allowed(rel: str) -> bool:
    return any(rel == a or rel.startswith(a) for a in _ALLOWLIST)


def test_no_new_temperature_arithmetic_outside_policy():
    offenders = []
    for path in (_VICTOR / "agent").rglob("*.py"):
        rel = str(path.relative_to(_VICTOR))
        if _allowed(rel):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                a, b = node.left, node.right
                if (_is_temperature_name(a) and _is_number(b)) or (
                    _is_temperature_name(b) and _is_number(a)
                ):
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "Raw temperature arithmetic found outside the unified policy (ADR-013). Route it through "
        "victor.framework.temperature instead of ad-hoc math:\n  " + "\n  ".join(offenders)
    )
