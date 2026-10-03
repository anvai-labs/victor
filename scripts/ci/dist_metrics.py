#!/usr/bin/env python3
"""Measure victor-ai distribution metrics: install size, dep count, CLI timings.

Complements scripts/benchmark_startup_kpi.py (which owns import/Agent-level KPIs).
This script answers the distribution question: "what does a user's machine get,
and how fast does the entry point answer?" for a bare (extras-free) install.

Modes:
  env         Measure the interpreter given by --python (default: this one).
              victor-ai must already be importable in it.
  venv-wheel  Build a throwaway venv, pip-install the given wheel (+ its PyPI
              deps), then re-exec this script in env mode inside that venv.
              This is the release-gate mode: it measures exactly what users get
              from `pip install victor_ai-<version>-py3-none-any.whl`.

Gates (--max-* and --expect-no-numpy) exit 2 on breach so CI can block.

Usage:
    python scripts/ci/dist_metrics.py --mode env
    python scripts/ci/dist_metrics.py --mode venv-wheel --wheel dist/*.whl --json out.json
    python scripts/ci/dist_metrics.py --mode env --max-help-ms 2200 --expect-no-numpy
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path
from typing import Any, Dict, List

# The probe that runs inside the TARGET interpreter. Returns one JSON object
# with the metadata-derived metrics (dependency closure, install size, numpy
# presence). Kept as a single -c string so it works against any interpreter
# without installing this script into it.
_META_PROBE = r"""
import json, sys
try:
    from importlib import metadata
except ImportError:  # pragma: no cover - py3.12+ only, defensive
    sys.exit("importlib.metadata unavailable")

ROOT = "victor-ai"


def _dist_name(req: str) -> str:
    name = req.split(";")[0].split("[")[0].strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        idx = name.find(sep)
        if idx != -1:
            name = name[:idx]
    return name.strip()


def _keep(req: str) -> bool:
    marker = req.split(";", 1)
    if len(marker) == 2 and "extra ==" in marker[1]:
        return False  # extras of dependencies are not part of the bare install
    return True


def _closure() -> tuple[list[str], int]:
    seen: dict[str, float] = {}
    queue = [ROOT]
    while queue:
        name = queue.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            seen[key] = 0.0
            continue
        size = 0.0
        for f in dist.files or []:
            try:
                size += f.locate().stat().st_size
            except OSError:
                pass
        seen[key] = size
        for req in (dist.requires or []):
            if _keep(req):
                queue.append(_dist_name(req))
    deps = sorted(k for k in seen if k != ROOT and seen[k] > 0)
    return deps, sum(seen.values())


deps, size = _closure()
try:
    import importlib.util as _util
    numpy_present = _util.find_spec("numpy") is not None
except Exception:
    numpy_present = True
print(json.dumps({"deps": deps, "install_size_bytes": int(size), "numpy_present": numpy_present}))
"""

# CLI timing probe: run the typer app with the given argv inside the target.
_CLI_PROBE = (
    "import sys, time; t0 = time.perf_counter(); "
    "sys.argv = ['victor'] + sys.argv[1:]; "
    "from victor.ui.cli import app; "
    "code = 0\n"
    "try:\n    app()\nexcept SystemExit as e:\n    code = int(e.code or 0)\n"
    "print(time.perf_counter() - t0)"
)


def _median_ms(values: List[float]) -> float:
    return round(statistics.median(values) * 1000.0, 1) if values else 0.0


def measure_env(python: str, iterations: int) -> Dict[str, Any]:
    """Measure an interpreter where victor-ai is already installed."""
    meta = subprocess.run(
        [python, "-c", _META_PROBE], capture_output=True, text=True, timeout=180.0, check=False
    )
    if meta.returncode != 0:
        raise SystemExit(
            f"metadata probe failed in {python} (rc={meta.returncode}):\n{meta.stderr[-800:]}"
        )
    data = json.loads(meta.stdout)

    import_ms: List[float] = []
    help_ms: List[float] = []
    version_ms: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        subprocess.run([python, "-c", "import victor"], capture_output=True, timeout=180.0)
        import_ms.append(time.perf_counter() - t0)

        for bucket, arg in ((help_ms, "--help"), (version_ms, "--version")):
            probe = subprocess.run(
                [python, "-c", _CLI_PROBE, arg], capture_output=True, text=True, timeout=180.0
            )
            if probe.returncode != 0:
                raise SystemExit(f"CLI probe `{arg}` failed:\n{probe.stderr[-800:]}")
            bucket.append(float(probe.stdout.strip().splitlines()[-1]))

    return {
        "python": python,
        "dep_count": len(data["deps"]),
        "deps": data["deps"],
        "install_size_mb": round(data["install_size_bytes"] / (1024 * 1024), 1),
        "numpy_present": data["numpy_present"],
        "import_victor_ms": _median_ms(import_ms),
        "cli_help_ms": _median_ms(help_ms),
        "cli_version_ms": _median_ms(version_ms),
        "iterations": iterations,
    }


def measure_venv_wheel(wheel: Path, iterations: int, keep_venv: bool) -> Dict[str, Any]:
    """Build a throwaway venv, install the wheel, measure inside it.

    The venv is seeded from the interpreter RUNNING this script (venv.create
    semantics), so invoke the driver with the python generation you want the
    measurement venv to inherit.
    """
    tmp = tempfile.mkdtemp(prefix="dist-metrics-")
    venv_dir = Path(tmp) / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    venv_python = str(venv_dir / "bin" / "python")
    subprocess.run(
        [venv_python, "-m", "pip", "install", "--quiet", str(wheel)],
        check=True,
        capture_output=True,
        timeout=600.0,
    )
    try:
        return measure_env(venv_python, iterations)
    finally:
        if not keep_venv:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


def evaluate_gates(report: Dict[str, Any], gates: Dict[str, Any]) -> List[str]:
    """Return human-readable breach messages; empty list means all gates pass."""
    breaches: List[str] = []
    if gates.get("max_help_ms") is not None and report["cli_help_ms"] > gates["max_help_ms"]:
        breaches.append(f"--help {report['cli_help_ms']}ms > {gates['max_help_ms']}ms")
    if (
        gates.get("max_version_ms") is not None
        and report["cli_version_ms"] > gates["max_version_ms"]
    ):
        breaches.append(f"--version {report['cli_version_ms']}ms > {gates['max_version_ms']}ms")
    if (
        gates.get("max_install_mb") is not None
        and report["install_size_mb"] > gates["max_install_mb"]
    ):
        breaches.append(f"install {report['install_size_mb']}MB > {gates['max_install_mb']}MB")
    if gates.get("expect_no_numpy") and report["numpy_present"]:
        breaches.append("numpy present in a bare install (expected absent)")
    return breaches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("env", "venv-wheel"), default="env")
    parser.add_argument("--wheel", type=Path, help="wheel to install (venv-wheel mode)")
    parser.add_argument("--python", default=sys.executable, help="target interpreter (env mode)")
    parser.add_argument("--iterations", type=int, default=5, help="timing samples per probe")
    parser.add_argument("--json", type=Path, help="write the report here")
    parser.add_argument("--keep-venv", action="store_true", help="keep the throwaway venv")
    parser.add_argument("--max-help-ms", type=float, default=None)
    parser.add_argument("--max-version-ms", type=float, default=None)
    parser.add_argument("--max-install-mb", type=float, default=None)
    parser.add_argument("--expect-no-numpy", action="store_true")
    args = parser.parse_args()

    if args.mode == "venv-wheel":
        if args.wheel is None or not args.wheel.exists():
            raise SystemExit("--mode venv-wheel requires --wheel pointing at a built wheel")
        report = measure_venv_wheel(args.wheel, args.iterations, args.keep_venv)
    else:
        report = measure_env(args.python, args.iterations)

    gates = {
        "max_help_ms": args.max_help_ms,
        "max_version_ms": args.max_version_ms,
        "max_install_mb": args.max_install_mb,
        "expect_no_numpy": args.expect_no_numpy,
    }
    breaches = evaluate_gates(report, gates)

    if args.json:
        args.json.write_text(json.dumps({**report, "gates": gates, "breaches": breaches}, indent=2))

    print(
        f"deps={report['dep_count']}  install={report['install_size_mb']}MB  "
        f"numpy={'present' if report['numpy_present'] else 'ABSENT'}  "
        f"import={report['import_victor_ms']}ms  "
        f"--help={report['cli_help_ms']}ms  --version={report['cli_version_ms']}ms"
    )

    if breaches:
        for breach in breaches:
            print(f"FAIL: {breach}")
        sys.exit(2)
    print("All distribution gates passed.")


if __name__ == "__main__":
    main()
