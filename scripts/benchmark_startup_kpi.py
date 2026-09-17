#!/usr/bin/env python3
"""Benchmark startup KPIs for Victor import, Agent.create(), and discovery.

Measures:
- `import victor` cold + warm timings
- `Agent.create()` cold + warm timings
- optional vertical discovery cold + warm timings with cache counters

Usage:
    python scripts/benchmark_startup_kpi.py
    python scripts/benchmark_startup_kpi.py --iterations 10 --json
    python scripts/benchmark_startup_kpi.py --provider openai --model gpt-4o-mini
    python scripts/benchmark_startup_kpi.py --skip-agent-create --discovery-probe --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return float(ordered[idx])


@dataclass
class _TimingSummary:
    cold_ms: float
    warm_samples_ms: List[float]

    @property
    def warm_mean_ms(self) -> float:
        return statistics.fmean(self.warm_samples_ms) if self.warm_samples_ms else 0.0

    @property
    def warm_p95_ms(self) -> float:
        return _percentile(self.warm_samples_ms, 0.95)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cold_ms": self.cold_ms,
            "warm_samples_ms": self.warm_samples_ms,
            "warm_mean_ms": self.warm_mean_ms,
            "warm_p95_ms": self.warm_p95_ms,
        }


def _run_snippet(
    *,
    python_executable: str,
    code: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Run an isolated Python snippet and parse JSON payload from stdout."""
    result = subprocess.run(
        [python_executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"benchmark subprocess failed: {details}")

    line = (result.stdout or "").strip().splitlines()
    if not line:
        raise RuntimeError("benchmark subprocess returned no output")
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid benchmark payload: {line[-1]}") from exc


def _measure_import_victor(
    *,
    python_executable: str,
    iterations: int,
    timeout_seconds: float,
) -> _TimingSummary:
    code = textwrap.dedent(f"""
        import importlib
        import json
        import time

        warm_iters = {iterations}
        t0 = time.perf_counter()
        import victor  # noqa: F401
        cold_ms = (time.perf_counter() - t0) * 1000.0

        warm = []
        for _ in range(warm_iters):
            t = time.perf_counter()
            importlib.import_module("victor")
            warm.append((time.perf_counter() - t) * 1000.0)

        print(json.dumps({{"cold_ms": cold_ms, "warm_ms": warm}}))
        """)
    payload = _run_snippet(
        python_executable=python_executable,
        code=code,
        timeout_seconds=timeout_seconds,
    )
    return _TimingSummary(
        cold_ms=float(payload["cold_ms"]),
        warm_samples_ms=[float(v) for v in payload["warm_ms"]],
    )


def _measure_agent_create(
    *,
    python_executable: str,
    iterations: int,
    provider: str,
    model: str,
    timeout_seconds: float,
) -> _TimingSummary:
    provider_literal = json.dumps(provider)
    model_literal = json.dumps(model)
    code = textwrap.dedent(f"""
        import asyncio
        import json
        import time

        from victor.framework import Agent

        warm_iters = {iterations}
        provider = {provider_literal}
        model = {model_literal}

        async def run():
            t0 = time.perf_counter()
            agent = await Agent.create(
                provider=provider,
                model=model,
                enable_observability=False,
            )
            cold_ms = (time.perf_counter() - t0) * 1000.0
            await agent.close()

            warm = []
            for _ in range(warm_iters):
                t = time.perf_counter()
                agent = await Agent.create(
                    provider=provider,
                    model=model,
                    enable_observability=False,
                )
                warm.append((time.perf_counter() - t) * 1000.0)
                await agent.close()

            print(json.dumps({{"cold_ms": cold_ms, "warm_ms": warm}}))

        asyncio.run(run())
        """)
    payload = _run_snippet(
        python_executable=python_executable,
        code=code,
        timeout_seconds=timeout_seconds,
    )
    return _TimingSummary(
        cold_ms=float(payload["cold_ms"]),
        warm_samples_ms=[float(v) for v in payload["warm_ms"]],
    )


def _measure_activation_probe(
    *,
    python_executable: str,
    iterations: int,
    vertical: str,
    provider: str,
    model: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Measure activation probe timing and metrics for a specific vertical."""
    provider_literal = json.dumps(provider)
    vertical_literal = json.dumps(vertical)
    model_literal = json.dumps(model)
    code = textwrap.dedent(f"""
        import asyncio
        import json
        import time

        from victor.framework import Agent

        # Import the vertical module directly to ensure registration
        import victor.core.verticals
        victor.core.verticals._register_builtin_verticals()

        # Get vertical class from registry
        from victor.core.verticals.base import VerticalRegistry
        from victor.framework.shim import get_vertical

        vertical = {vertical_literal}
        provider = {provider_literal}
        model = {model_literal}
        warm_iters = {iterations}
        vertical_cls = get_vertical(vertical)
        if vertical_cls is None:
            raise ValueError(f"Unknown vertical: {{vertical}}")

        def _percentile(values, q):
            if not values:
                return 0.0
            if len(values) == 1:
                return float(values[0])
            ordered = sorted(values)
            idx = round((len(ordered) - 1) * q)
            return float(ordered[idx])

        def _runtime_component_lazy(orchestrator, runtime_name, component_names):
            runtime = getattr(orchestrator, runtime_name, None)
            if runtime is None:
                return False
            for component_name in component_names:
                component = getattr(runtime, component_name, None)
                if component is None:
                    return False
                initialized = getattr(component, "initialized", None)
                if initialized is None or bool(initialized):
                    return False
            return True

        def _extract_runtime_flags(orchestrator):
            settings = getattr(orchestrator, "settings", None)
            return {{
                "generic_result_cache_enabled": bool(
                    getattr(settings, "generic_result_cache_enabled", False)
                ),
                "http_connection_pool_enabled": bool(
                    getattr(settings, "http_connection_pool_enabled", False)
                ),
                "framework_preload_enabled": bool(
                    getattr(settings, "framework_preload_enabled", False)
                ),
                "coordination_runtime_lazy": _runtime_component_lazy(
                    orchestrator,
                    "_coordination_runtime",
                    (
                        "recovery_coordinator",
                        "chunk_generator",
                        "tool_planner",
                        "task_coordinator",
                    ),
                ),
                "interaction_runtime_lazy": _runtime_component_lazy(
                    orchestrator,
                    "_interaction_runtime",
                    (
                        "chat_coordinator",
                        "tool_coordinator",
                        "session_coordinator",
                    ),
                ),
            }}

        def _extract_registry_metrics(orchestrator):
            metrics = {{}}
            try:
                from victor.framework.framework_integration_registry_service import (
                    resolve_framework_integration_registry_service,
                )

                registry_service = resolve_framework_integration_registry_service(orchestrator)
                snapshot_metrics = getattr(registry_service, "snapshot_metrics", None)
                if callable(snapshot_metrics):
                    metrics = snapshot_metrics() or {{}}
            except Exception:
                metrics = {{}}
            return metrics

        def _registry_totals(metrics):
            attempted_total = 0
            applied_total = 0
            for entry in metrics.values():
                if not isinstance(entry, dict):
                    continue
                attempted_total += int(entry.get("attempted", 0) or 0)
                applied_total += int(entry.get("applied", 0) or 0)
            return attempted_total, applied_total

        async def _sample_activation():
            t0 = time.perf_counter()
            agent = await Agent.create(
                provider=provider,
                model=model,
                vertical=vertical_cls,
                enable_observability=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            orchestrator = agent._orchestrator
            runtime_flags = _extract_runtime_flags(orchestrator)
            framework_registry_metrics = _extract_registry_metrics(orchestrator)
            framework_registry_attempted_total, framework_registry_applied_total = (
                _registry_totals(framework_registry_metrics)
            )

            await agent.close()

            return {{
                "elapsed_ms": elapsed_ms,
                "runtime_flags": runtime_flags,
                "framework_registry_metrics": framework_registry_metrics,
                "framework_registry_attempted_total": framework_registry_attempted_total,
                "framework_registry_applied_total": framework_registry_applied_total,
            }}

        async def run():
            cold_probe = await _sample_activation()
            warm_samples_ms = []
            for _ in range(warm_iters):
                sample = await _sample_activation()
                warm_samples_ms.append(sample["elapsed_ms"])

            warm_mean_ms = (
                (sum(warm_samples_ms) / len(warm_samples_ms)) if warm_samples_ms else 0.0
            )
            warm_p95_ms = _percentile(warm_samples_ms, 0.95)

            print(json.dumps({{
                "vertical": vertical,
                "cold_ms": cold_probe["elapsed_ms"],
                "warm_samples_ms": warm_samples_ms,
                "warm_mean_ms": warm_mean_ms,
                "warm_p95_ms": warm_p95_ms,
                "runtime_flags": cold_probe["runtime_flags"],
                "framework_registry_metrics": cold_probe["framework_registry_metrics"],
                "framework_registry_attempted_total": cold_probe["framework_registry_attempted_total"],
                "framework_registry_applied_total": cold_probe["framework_registry_applied_total"],
            }}))

        asyncio.run(run())
        """)
    payload = _run_snippet(
        python_executable=python_executable,
        code=code,
        timeout_seconds=timeout_seconds,
    )
    return payload


def _measure_discovery_probe(
    *,
    python_executable: str,
    iterations: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Measure cold/warm vertical discovery behavior and cache usage."""
    code = textwrap.dedent(f"""
        import json
        import time

        from victor.core.verticals.vertical_loader import VerticalLoader
        from victor.framework.module_loader import get_entry_point_cache

        warm_iters = {iterations}
        loader = VerticalLoader()
        entry_point_cache = get_entry_point_cache()

        initial_loader_stats = loader.get_discovery_stats()

        t0 = time.perf_counter()
        discovered = loader.discover_verticals(force_refresh=True, emit_event=False)
        cold_ms = (time.perf_counter() - t0) * 1000.0

        warm = []
        for _ in range(warm_iters):
            t = time.perf_counter()
            loader.discover_verticals(emit_event=False)
            warm.append((time.perf_counter() - t) * 1000.0)

        final_loader_stats = loader.get_discovery_stats()
        vertical_stats = final_loader_stats.get("vertical", {{}})
        initial_vertical_stats = initial_loader_stats.get("vertical", {{}})
        entry_point_cache_stats = entry_point_cache.get_cache_stats()
        cached_vertical_group = entry_point_cache_stats.get("groups", {{}}).get(
            "victor.plugins",
            {{}},
        )

        def _delta(name):
            return int(vertical_stats.get(name, 0) or 0) - int(
                initial_vertical_stats.get(name, 0) or 0
            )

        def _percentile(values, q):
            if not values:
                return 0.0
            if len(values) == 1:
                return float(values[0])
            ordered = sorted(values)
            idx = round((len(ordered) - 1) * q)
            return float(ordered[idx])

        warm_mean_ms = (sum(warm) / len(warm)) if warm else 0.0

        print(json.dumps({{
            "cold_ms": cold_ms,
            "warm_ms": warm,
            "warm_mean_ms": warm_mean_ms,
            "warm_p95_ms": _percentile(warm, 0.95),
            "discovered_count": len(discovered),
            "discovered_vertical_names": sorted(discovered.keys()),
            "call_total": _delta("calls"),
            "cache_hit_total": _delta("cache_hits"),
            "scan_total": _delta("scans"),
            "last_discovery_ms": float(vertical_stats.get("last_discovery_ms", 0.0) or 0.0),
            "entry_point_groups_cached": int(entry_point_cache_stats.get("groups_cached", 0) or 0),
            "entry_point_vertical_entries": int(cached_vertical_group.get("entries", 0) or 0),
        }}))
        """)
    payload = _run_snippet(
        python_executable=python_executable,
        code=code,
        timeout_seconds=timeout_seconds,
    )
    return {
        "cold_ms": float(payload["cold_ms"]),
        "warm_samples_ms": [float(v) for v in payload["warm_ms"]],
        "warm_mean_ms": float(payload["warm_mean_ms"]),
        "warm_p95_ms": float(payload["warm_p95_ms"]),
        "discovered_count": int(payload["discovered_count"]),
        "discovered_vertical_names": [str(v) for v in payload["discovered_vertical_names"]],
        "call_total": int(payload["call_total"]),
        "cache_hit_total": int(payload["cache_hit_total"]),
        "scan_total": int(payload["scan_total"]),
        "last_discovery_ms": float(payload["last_discovery_ms"]),
        "entry_point_groups_cached": int(payload["entry_point_groups_cached"]),
        "entry_point_vertical_entries": int(payload["entry_point_vertical_entries"]),
    }


def _build_report(
    *,
    python_executable: str,
    iterations: int,
    provider: str,
    model: str,
    import_timing: _TimingSummary,
    create_timing: _TimingSummary | None,
    discovery_probe: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python": python_executable,
        "iterations": iterations,
        "import_victor": import_timing.to_dict(),
    }
    if create_timing is not None:
        report["agent_create"] = {
            "provider": provider,
            "model": model,
            **create_timing.to_dict(),
        }
    if discovery_probe is not None:
        report["discovery_probe"] = discovery_probe
    return report


def _lookup_report_value(report: Dict[str, Any], key: str) -> Any:
    """Resolve a dotted report key to a nested value."""
    value: Any = report
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue
        return None
    return value


def _collect_thresholds(args: argparse.Namespace) -> Dict[str, float]:
    """Collect threshold values from CLI args."""
    thresholds: Dict[str, float] = {}

    if hasattr(args, "max_import_cold_ms") and args.max_import_cold_ms is not None:
        thresholds["import_victor.cold_ms"] = args.max_import_cold_ms
    if hasattr(args, "max_import_warm_mean_ms") and args.max_import_warm_mean_ms is not None:
        thresholds["import_victor.warm_mean_ms"] = args.max_import_warm_mean_ms
    if hasattr(args, "max_agent_create_cold_ms") and args.max_agent_create_cold_ms is not None:
        thresholds["agent_create.cold_ms"] = args.max_agent_create_cold_ms
    if (
        hasattr(args, "max_agent_create_warm_mean_ms")
        and args.max_agent_create_warm_mean_ms is not None
    ):
        thresholds["agent_create.warm_mean_ms"] = args.max_agent_create_warm_mean_ms
    if hasattr(args, "max_activation_cold_ms") and args.max_activation_cold_ms is not None:
        thresholds["activation_probe.cold_ms"] = args.max_activation_cold_ms
    if (
        hasattr(args, "max_activation_warm_mean_ms")
        and args.max_activation_warm_mean_ms is not None
    ):
        thresholds["activation_probe.warm_mean_ms"] = args.max_activation_warm_mean_ms
    if hasattr(args, "max_discovery_cold_ms") and args.max_discovery_cold_ms is not None:
        thresholds["discovery_probe.cold_ms"] = args.max_discovery_cold_ms
    if hasattr(args, "max_discovery_warm_mean_ms") and args.max_discovery_warm_mean_ms is not None:
        thresholds["discovery_probe.warm_mean_ms"] = args.max_discovery_warm_mean_ms

    return thresholds


def _evaluate_threshold_failures(report: Dict[str, Any], thresholds: Dict[str, float]) -> List[str]:
    """Evaluate report against thresholds and return list of failure messages."""
    failures: List[str] = []

    for key, threshold in thresholds.items():
        value = _lookup_report_value(report, key)
        if value is None:
            continue

        if value > threshold:
            failures.append(f"{key}: {value:.2f} exceeds threshold {threshold:.2f}")

    return failures


def _collect_minimums(args: argparse.Namespace) -> Dict[str, float]:
    """Collect minimum value requirements from CLI args."""
    minimums: Dict[str, float] = {}

    if (
        hasattr(args, "min_framework_registry_attempted_total")
        and args.min_framework_registry_attempted_total is not None
    ):
        minimums["activation_probe.framework_registry_attempted_total"] = (
            args.min_framework_registry_attempted_total
        )
    if (
        hasattr(args, "min_framework_registry_applied_total")
        and args.min_framework_registry_applied_total is not None
    ):
        minimums["activation_probe.framework_registry_applied_total"] = (
            args.min_framework_registry_applied_total
        )
    if (
        hasattr(args, "min_discovery_vertical_count")
        and args.min_discovery_vertical_count is not None
    ):
        minimums["discovery_probe.discovered_count"] = args.min_discovery_vertical_count
    if hasattr(args, "min_discovery_cache_hits") and args.min_discovery_cache_hits is not None:
        minimums["discovery_probe.cache_hit_total"] = args.min_discovery_cache_hits

    return minimums


def _evaluate_minimum_failures(report: Dict[str, Any], minimums: Dict[str, float]) -> List[str]:
    """Evaluate report against minimums and return list of failure messages."""
    failures: List[str] = []

    for key, minimum in minimums.items():
        value = _lookup_report_value(report, key)
        if value is None:
            continue

        if value < minimum:
            failures.append(f"{key}: {value:.2f} below minimum {minimum:.2f}")

    return failures


def _collect_flag_expectations(args: argparse.Namespace) -> Dict[str, bool]:
    """Collect runtime flag expectations from CLI args."""
    expectations: Dict[str, bool] = {}

    if (
        hasattr(args, "require_generic_result_cache_enabled")
        and args.require_generic_result_cache_enabled
    ):
        expectations["activation_probe.runtime_flags.generic_result_cache_enabled"] = True
    if (
        hasattr(args, "require_http_connection_pool_enabled")
        and args.require_http_connection_pool_enabled
    ):
        expectations["activation_probe.runtime_flags.http_connection_pool_enabled"] = True
    if (
        hasattr(args, "require_framework_preload_enabled")
        and args.require_framework_preload_enabled
    ):
        expectations["activation_probe.runtime_flags.framework_preload_enabled"] = True
    if (
        hasattr(args, "require_coordination_runtime_lazy")
        and args.require_coordination_runtime_lazy
    ):
        expectations["activation_probe.runtime_flags.coordination_runtime_lazy"] = True
    if hasattr(args, "require_interaction_runtime_lazy") and args.require_interaction_runtime_lazy:
        expectations["activation_probe.runtime_flags.interaction_runtime_lazy"] = True

    return expectations


def _evaluate_flag_expectation_failures(
    report: Dict[str, Any], expectations: Dict[str, bool]
) -> List[str]:
    """Evaluate report against flag expectations and return list of failure messages."""
    failures: List[str] = []

    for key, expected_value in expectations.items():
        value = _lookup_report_value(report, key)
        if value is None:
            failures.append(f"{key}: not found in report")
        elif value != expected_value:
            failures.append(f"{key}: {value} != {expected_value}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark startup KPIs for import victor, Agent.create(), and discovery."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of warm iterations for each measurement (default: 5).",
    )
    parser.add_argument(
        "--provider",
        default="inferflux",
        help="Provider used for Agent.create() baseline (default: inferflux).",
    )
    parser.add_argument(
        "--model",
        default="qwen3-coder:30b",
        help="Model used for Agent.create() baseline (default: qwen3-coder:30b).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for subprocess probes (default: current interpreter).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Per-probe timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report only.",
    )
    parser.add_argument(
        "--skip-agent-create",
        action="store_true",
        help="Skip the Agent.create() probe when only import/discovery metrics are needed.",
    )
    parser.add_argument(
        "--discovery-probe",
        action="store_true",
        help="Measure vertical discovery cold/warm timings and cache behavior.",
    )

    # Threshold arguments for CI/CD validation
    parser.add_argument(
        "--max-import-cold-ms",
        type=float,
        default=None,
        help="Maximum acceptable import cold time in milliseconds.",
    )
    parser.add_argument(
        "--max-import-warm-mean-ms",
        type=float,
        default=None,
        help="Maximum acceptable import warm mean time in milliseconds.",
    )
    parser.add_argument(
        "--max-agent-create-cold-ms",
        type=float,
        default=None,
        help="Maximum acceptable agent_create cold time in milliseconds.",
    )
    parser.add_argument(
        "--max-agent-create-warm-mean-ms",
        type=float,
        default=None,
        help="Maximum acceptable agent_create warm mean time in milliseconds.",
    )
    parser.add_argument(
        "--max-activation-cold-ms",
        type=float,
        default=None,
        help="Maximum acceptable activation-probe cold time in milliseconds.",
    )
    parser.add_argument(
        "--max-activation-warm-mean-ms",
        type=float,
        default=None,
        help="Maximum acceptable activation-probe warm mean time in milliseconds.",
    )
    parser.add_argument(
        "--max-discovery-cold-ms",
        type=float,
        default=None,
        help="Maximum acceptable discovery-probe cold time in milliseconds.",
    )
    parser.add_argument(
        "--max-discovery-warm-mean-ms",
        type=float,
        default=None,
        help="Maximum acceptable discovery-probe warm mean time in milliseconds.",
    )

    # Minimum value arguments
    parser.add_argument(
        "--min-framework-registry-attempted-total",
        type=float,
        default=None,
        help="Minimum acceptable framework registry attempted total.",
    )
    parser.add_argument(
        "--min-framework-registry-applied-total",
        type=float,
        default=None,
        help="Minimum acceptable framework registry applied total.",
    )
    parser.add_argument(
        "--min-discovery-vertical-count",
        type=float,
        default=None,
        help="Minimum acceptable number of discovered external verticals.",
    )
    parser.add_argument(
        "--min-discovery-cache-hits",
        type=float,
        default=None,
        help="Minimum acceptable discovery cache hits across warm samples.",
    )

    # Flag expectation arguments
    parser.add_argument(
        "--require-generic-result-cache-enabled",
        action="store_true",
        help="Require generic_result_cache_enabled to be True.",
    )
    parser.add_argument(
        "--require-http-connection-pool-enabled",
        action="store_true",
        help="Require http_connection_pool_enabled to be True.",
    )
    parser.add_argument(
        "--require-framework-preload-enabled",
        action="store_true",
        help="Require framework_preload_enabled to be True.",
    )
    parser.add_argument(
        "--require-coordination-runtime-lazy",
        action="store_true",
        help="Require coordination runtime components to remain lazily uninitialized.",
    )
    parser.add_argument(
        "--require-interaction-runtime-lazy",
        action="store_true",
        help="Require interaction runtime components to remain lazily uninitialized.",
    )

    # Activation probe arguments
    parser.add_argument(
        "--activation-vertical",
        type=str,
        default=None,
        help="Vertical to use for activation probe (e.g., coding, research).",
    )

    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")

    import_timing = _measure_import_victor(
        python_executable=args.python,
        iterations=args.iterations,
        timeout_seconds=args.timeout_seconds,
    )
    create_timing = None
    if not args.skip_agent_create:
        create_timing = _measure_agent_create(
            python_executable=args.python,
            iterations=args.iterations,
            provider=args.provider,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )

    discovery_probe = None
    if args.discovery_probe:
        discovery_probe = _measure_discovery_probe(
            python_executable=args.python,
            iterations=args.iterations,
            timeout_seconds=args.timeout_seconds,
        )

    report = _build_report(
        python_executable=args.python,
        iterations=args.iterations,
        provider=args.provider,
        model=args.model,
        import_timing=import_timing,
        create_timing=create_timing,
        discovery_probe=discovery_probe,
    )

    # Add activation probe if vertical specified
    if args.activation_vertical:
        activation_probe = _measure_activation_probe(
            python_executable=args.python,
            iterations=args.iterations,
            vertical=args.activation_vertical,
            provider=args.provider,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        )
        report["activation_probe"] = activation_probe

    # Collect and evaluate thresholds
    thresholds = _collect_thresholds(args)
    threshold_failures = _evaluate_threshold_failures(report, thresholds) if thresholds else []

    # Collect and evaluate minimums
    minimums = _collect_minimums(args)
    minimum_failures = _evaluate_minimum_failures(report, minimums) if minimums else []

    # Collect and evaluate flag expectations
    flag_expectations = _collect_flag_expectations(args)
    flag_failures = (
        _evaluate_flag_expectation_failures(report, flag_expectations) if flag_expectations else []
    )

    # Combine all failures
    all_failures = threshold_failures + minimum_failures + flag_failures

    # Add failures to report
    if all_failures:
        report["threshold_failures"] = all_failures

    if args.json:
        report["thresholds"] = thresholds
        report["minimums"] = minimums
        report["flag_expectations"] = flag_expectations
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if all_failures else 0

    print("Startup KPI Benchmark")
    print(f"python: {report['python']}")
    print(f"iterations: {report['iterations']}")
    print("")
    print("import victor")
    print(f"  cold_ms: {report['import_victor']['cold_ms']:.2f}")
    print(f"  warm_mean_ms: {report['import_victor']['warm_mean_ms']:.2f}")
    print(f"  warm_p95_ms: {report['import_victor']['warm_p95_ms']:.2f}")
    if "agent_create" in report:
        print("")
        print("Agent.create()")
        print(f"  provider/model: {args.provider}/{args.model}")
        print(f"  cold_ms: {report['agent_create']['cold_ms']:.2f}")
        print(f"  warm_mean_ms: {report['agent_create']['warm_mean_ms']:.2f}")
        print(f"  warm_p95_ms: {report['agent_create']['warm_p95_ms']:.2f}")
    if "discovery_probe" in report:
        print("")
        print("Discovery probe")
        print(f"  cold_ms: {report['discovery_probe']['cold_ms']:.2f}")
        print(f"  warm_mean_ms: {report['discovery_probe']['warm_mean_ms']:.2f}")
        print(f"  warm_p95_ms: {report['discovery_probe']['warm_p95_ms']:.2f}")
        print(f"  discovered_count: {report['discovery_probe']['discovered_count']}")
        print(f"  cache_hit_total: {report['discovery_probe']['cache_hit_total']}")
    if "activation_probe" in report:
        print("")
        print(f"Activation probe ({report['activation_probe']['vertical']})")
        print(f"  cold_ms: {report['activation_probe']['cold_ms']:.2f}")
        print(f"  warm_mean_ms: {report['activation_probe']['warm_mean_ms']:.2f}")
        print(f"  warm_p95_ms: {report['activation_probe']['warm_p95_ms']:.2f}")

    if all_failures:
        print("")
        print("FAILURES:")
        for failure in all_failures:
            print(f"  - {failure}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
