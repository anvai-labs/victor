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

"""Defaults + model bounds for the unified temperature policy (ADR-013).

Canonical home for the sampling-temperature constants so they live in exactly one place:
- ``GLOBAL_DEFAULT`` — the terminal fallback when no source resolves a base.
- ``DEFAULT_RATCHET_STEP`` / ``DEFAULT_RATCHET_CAP`` — the spin-escape ratchet parameters, grounded in
  arXiv 2606.01451 (coherence-flat region T≈0.3–0.8; degeneration cliff past T≈1.0 toward 1.5) so the
  step is modest and the cap stays well below 1.0.
- ``MODEL_TEMPERATURE_RANGES`` — per-model effective bounds (relocated from
  ``recovery/temperature.py``; that module now imports these back, eliminating the duplicate).
"""

from __future__ import annotations

from typing import Dict, Tuple

# Terminal base when no source resolves one. 0.6 favours determinism for agentic tool-use while
# leaving headroom for the spin ratchet to escape repetition loops (see ADR-013 / FEP-0007).
GLOBAL_DEFAULT: float = 0.6

# Spin-escape ratchet: modest step, hard cap below the degeneration cliff (arXiv 2606.01451).
DEFAULT_RATCHET_STEP: float = 0.05
DEFAULT_RATCHET_CAP: float = 0.9

# Per-model effective temperature ranges (min_effective, max_effective). Substring-matched against
# the model name. Single source of truth — recovery/temperature.py imports this.
MODEL_TEMPERATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "qwen": (0.3, 0.9),
    "llama": (0.2, 0.8),
    "mistral": (0.3, 0.85),
    "claude": (0.0, 1.0),
    "gpt": (0.0, 1.0),
    "deepseek": (0.2, 0.9),
    "glm": (0.0, 1.0),
    # Fixed-temperature models: bounds collapse to a point so the terminal
    # ModelBoundsModifier forces exactly that value (overriding profile, global
    # default, spin ratchet, and recovery alike). kimi-k3 rejects any temperature
    # != 1.0 ("only 1 is allowed for this model"), so no profile need carry it.
    "kimi-k3": (1.0, 1.0),
}

# Full range when no model pattern matches.
DEFAULT_MODEL_BOUNDS: Tuple[float, float] = (0.0, 1.0)


def escalate_temperature(base: float, increment: float, *, cap: float) -> float:
    """Raise ``base`` by ``increment``, clamped at ``cap`` — the single home for temperature-escalation
    arithmetic (ADR-013). The spin ratchet and the recovery ramps both call this instead of inlining
    ``base + <number>``, so the consolidation holds and the boundary guard has nothing to allowlist.
    """
    return min(base + increment, cap)


def model_bounds(model_name: str) -> Tuple[float, float]:
    """Return (min, max) effective temperature for ``model_name`` (substring match, full-range default).

    Mirrors ``ProgressiveTemperatureAdjuster._get_model_bounds`` so both share one definition.
    """
    name = (model_name or "").lower()
    for pattern, bounds in MODEL_TEMPERATURE_RANGES.items():
        if pattern in name:
            return bounds
    return DEFAULT_MODEL_BOUNDS
