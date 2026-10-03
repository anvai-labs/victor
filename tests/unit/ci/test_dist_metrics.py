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

"""Tests for scripts/ci/dist_metrics.py pure helpers and gates.

The metadata probe itself runs inside a target interpreter as a `-c` snippet;
here we test the driver-side pieces that are importable and deterministic:
requirement-name parsing, extras-marker filtering, gate evaluation, and the
subprocess timing path against a trivial interpreter target.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "dist_metrics.py"


def _load():
    spec = importlib.util.spec_from_file_location("dist_metrics", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load()


# ── probe-internal helpers (re-verified here via exec of the probe source) ────


def _probe_namespace(mod):
    """Execute the metadata probe source up to (not including) the closure call
    so its helpers can be unit-tested without a victor install."""
    namespace: dict = {}
    probe_lines = mod._META_PROBE.splitlines()
    # Everything before the `deps, size = _closure()` statement.
    prefix = "\n".join(probe_lines[: probe_lines.index("deps, size = _closure()")])
    exec(compile(prefix, "<meta-probe>", "exec"), namespace)
    return namespace


def test_probe_dist_name_strips_specs_and_extras(mod):
    ns = _probe_namespace(mod)
    assert ns["_dist_name"]("pydantic>=2.0") == "pydantic"
    assert ns["_dist_name"]("victor-contracts>=0.9.2,<1.0") == "victor-contracts"
    assert ns["_dist_name"]("some-pkg[extra1,extra2]>=1.0; python_version>='3.12'") == "some-pkg"
    assert ns["_dist_name"]("sandhi-gateway==0.10.1") == "sandhi-gateway"


def test_probe_marker_filter_drops_dependency_extras_only(mod):
    ns = _probe_namespace(mod)
    keep = ns["_keep"]
    # Extra-conditional requirements never belong to the bare install.
    assert keep("torch>=2.0; extra == 'embeddings'") is False
    assert keep("chainlit>=2,<3; extra == 'chat-ui'") is False
    # Unconditional requirements stay.
    assert keep("pydantic>=2.0") is True
    assert keep("numpy>=1.24,<2.5; python_version>='3.12'") is True


# ── gate evaluation ───────────────────────────────────────────────────────────


def _report(help_ms=100.0, version_ms=50.0, size_mb=10.0, numpy_present=False):
    return {
        "cli_help_ms": help_ms,
        "cli_version_ms": version_ms,
        "install_size_mb": size_mb,
        "numpy_present": numpy_present,
    }


def test_gates_empty_when_all_pass(mod):
    assert mod.evaluate_gates(_report(), {}) == []
    assert (
        mod.evaluate_gates(
            _report(help_ms=2200, version_ms=1100, size_mb=150),
            {"max_help_ms": 2200, "max_version_ms": 1100, "max_install_mb": 150},
        )
        == []
    )


@pytest.mark.parametrize(
    ("report_overrides", "gates", "expected_fragment"),
    [
        ({"help_ms": 2000.0}, {"max_help_ms": 500}, "--help"),
        ({"version_ms": 100.0}, {"max_version_ms": 10}, "--version"),
        ({"size_mb": 500.0}, {"max_install_mb": 5}, "install"),
        ({"numpy_present": True}, {"expect_no_numpy": True}, "numpy"),
    ],
)
def test_gates_breach_messages_name_the_metric(mod, report_overrides, gates, expected_fragment):
    breaches = mod.evaluate_gates(_report(**report_overrides), gates)
    assert len(breaches) == 1
    assert expected_fragment in breaches[0]


def test_gate_at_exact_boundary_passes(mod):
    assert mod.evaluate_gates(_report(help_ms=2200.0), {"max_help_ms": 2200}) == []


# ── subprocess timing path ────────────────────────────────────────────────────


def test_measure_env_reports_shape_in_victor_capable_interpreter(mod):
    """measure_env runs its probes against the real (editable) install in this
    repo's venv and returns the full report shape."""
    probe = __import__("subprocess").run(
        [sys.executable, "-c", "import victor"], capture_output=True
    )
    if probe.returncode != 0:
        pytest.skip("victor-ai not importable in this interpreter")

    report = mod.measure_env(sys.executable, 1)

    for key in (
        "dep_count",
        "install_size_mb",
        "numpy_present",
        "import_victor_ms",
        "cli_help_ms",
        "cli_version_ms",
        "iterations",
    ):
        assert key in report
    assert report["iterations"] == 1
    assert report["cli_help_ms"] >= 0 and report["cli_version_ms"] >= 0
    assert report["dep_count"] >= 1 and report["install_size_mb"] > 0
