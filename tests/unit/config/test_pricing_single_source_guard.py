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

"""G3 (sandhi typed-integration gap ledger): pricing has ONE source of truth.

Victor pricing lives in ``config/provider_metrics.yaml`` resolved via
``config/metrics_capabilities.py``. Before this change three other modules
carried hand-maintained price tables in two different unit systems —
``workflows/cost_router.py`` even multiplied per-MTok reference numbers as if
they were per-1k, overestimating costs 1000x.

The guard here keeps hardcoded nonzero pricing literals from returning
anywhere outside the canonical config layer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
VICTOR_DIR = REPO_ROOT / "victor"

PRICING_KEYWORDS = {
    "input_cost_per_1k",
    "output_cost_per_1k",
    "cost_per_1k_input",
    "cost_per_1k_output",
    "input_cost_per_mtok",
    "output_cost_per_mtok",
    "cache_read_cost_per_mtok",
    "cache_write_cost_per_mtok",
}

# The canonical pricing layer — the only place numeric pricing may live.
ALLOWED = {VICTOR_DIR / "config" / "metrics_capabilities.py"}


def _nonzero_literal(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value != 0
    )


class TestPricingSingleSourceGuard:
    def test_no_hardcoded_pricing_literals_outside_canonical_config(self):
        offenders = []
        for path in VICTOR_DIR.rglob("*.py"):
            if path in ALLOWED:
                continue
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword) and node.arg in PRICING_KEYWORDS:
                    if _nonzero_literal(node.value):
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.value.lineno}")
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if node.target.id in PRICING_KEYWORDS and node.value is not None:
                        if _nonzero_literal(node.value):
                            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.target.lineno}")

        assert not offenders, (
            "Hardcoded nonzero pricing literals found outside the canonical "
            f"config layer: {offenders}. Add pricing to "
            "victor/config/provider_metrics.yaml and resolve it via "
            "get_metrics_capabilities() instead."
        )


class TestCostRouterUsesCanonicalPricing:
    def test_estimate_cost_uses_per_mtok_math(self):
        from victor.workflows.cost_router import CostAwareRouter

        router = CostAwareRouter()
        cost = router.estimate_cost(
            model="claude-sonnet-4-20250514", input_tokens=1_000_000, output_tokens=0
        )
        # Canonical: 3.0 per MTok — NOT 3000 (the old per-1k misread)
        assert cost == 3.0

    def test_estimate_cost_falls_back_to_legacy_fields_when_canonical_disabled(self):
        from victor.workflows.cost_router import CostAwareRouter, CostTier, ModelConfig

        router = CostAwareRouter()
        router.register_model(
            ModelConfig(
                name="custom-local",
                provider="no-such-provider",
                cost_tier=CostTier.LOW,
                input_cost_per_1k=0.5,
                output_cost_per_1k=1.5,
            )
        )
        cost = router.estimate_cost(model="custom-local", input_tokens=2000, output_tokens=1000)
        assert cost == (2000 / 1000) * 0.5 + (1000 / 1000) * 1.5


class TestModelSwitcherHydratesCanonicalPricing:
    def test_zero_cost_models_hydrate_from_canonical_config(self):
        from victor.agent.model_switcher import ModelSwitcher, ModelInfo

        switcher = ModelSwitcher()
        switcher.register_model(
            ModelInfo(
                provider="anthropic",
                model_id="claude-sonnet-4-20250514",
                display_name="Sonnet 4",
            )
        )
        model = switcher._available_models["anthropic:claude-sonnet-4-20250514"]
        assert model.cost_per_1k_input == 3.0 / 1000
        assert model.cost_per_1k_output == 15.0 / 1000

    def test_explicit_costs_are_preserved(self):
        from victor.agent.model_switcher import ModelSwitcher, ModelInfo

        switcher = ModelSwitcher()
        switcher.register_model(
            ModelInfo(
                provider="anthropic",
                model_id="claude-sonnet-4-20250514",
                display_name="Sonnet 4",
                cost_per_1k_input=0.111,
                cost_per_1k_output=0.222,
            )
        )
        model = switcher._available_models["anthropic:claude-sonnet-4-20250514"]
        assert model.cost_per_1k_input == 0.111

    def test_canonical_lookup_failure_is_nonfatal(self):
        from victor.agent import model_switcher as ms

        info = ms.ModelInfo(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            display_name="Sonnet 4",
        )
        with patch(
            "victor.config.metrics_capabilities.get_metrics_capabilities",
            side_effect=RuntimeError("boom"),
        ):
            switcher = ms.ModelSwitcher()
            switcher.register_model(info)
        assert info.cost_per_1k_input == 0.0
