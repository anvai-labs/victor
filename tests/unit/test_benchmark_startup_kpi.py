# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path


def _load_benchmark_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_startup_kpi.py"
    module_name = f"benchmark_startup_kpi_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load benchmark_startup_kpi.py module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_thresholds_maps_cli_args():
    module = _load_benchmark_module()
    args = argparse.Namespace(
        max_import_cold_ms=120.0,
        max_import_warm_mean_ms=15.5,
        max_agent_create_cold_ms=None,
        max_agent_create_warm_mean_ms=240.0,
        max_activation_cold_ms=980.0,
        max_activation_warm_mean_ms=None,
        max_discovery_cold_ms=45.0,
        max_discovery_warm_mean_ms=4.5,
    )

    thresholds = module._collect_thresholds(args)

    assert thresholds == {
        "import_victor.cold_ms": 120.0,
        "import_victor.warm_mean_ms": 15.5,
        "agent_create.warm_mean_ms": 240.0,
        "activation_probe.cold_ms": 980.0,
        "discovery_probe.cold_ms": 45.0,
        "discovery_probe.warm_mean_ms": 4.5,
    }


def test_evaluate_threshold_failures_reports_exceeded_metrics():
    module = _load_benchmark_module()
    report = {
        "import_victor": {"cold_ms": 140.0, "warm_mean_ms": 3.0},
        "agent_create": {"cold_ms": 600.0, "warm_mean_ms": 220.0},
    }
    thresholds = {
        "import_victor.cold_ms": 120.0,
        "import_victor.warm_mean_ms": 5.0,
        "agent_create.cold_ms": 700.0,
        "agent_create.warm_mean_ms": 200.0,
    }

    failures = module._evaluate_threshold_failures(report, thresholds)

    assert len(failures) == 2
    assert any("import_victor.cold_ms" in failure for failure in failures)
    assert any("agent_create.warm_mean_ms" in failure for failure in failures)


def test_collect_minimums_maps_cli_args():
    module = _load_benchmark_module()
    args = argparse.Namespace(
        min_framework_registry_attempted_total=4.0,
        min_framework_registry_applied_total=None,
        min_discovery_vertical_count=1.0,
        min_discovery_cache_hits=3.0,
    )

    minimums = module._collect_minimums(args)

    assert minimums == {
        "activation_probe.framework_registry_attempted_total": 4.0,
        "discovery_probe.discovered_count": 1.0,
        "discovery_probe.cache_hit_total": 3.0,
    }


def test_collect_flag_expectations_maps_cli_args():
    module = _load_benchmark_module()
    args = argparse.Namespace(
        require_generic_result_cache_enabled=True,
        require_http_connection_pool_enabled=False,
        require_framework_preload_enabled=True,
        require_coordination_runtime_lazy=True,
        require_interaction_runtime_lazy=False,
    )

    expectations = module._collect_flag_expectations(args)

    assert expectations == {
        "activation_probe.runtime_flags.generic_result_cache_enabled": True,
        "activation_probe.runtime_flags.framework_preload_enabled": True,
        "activation_probe.runtime_flags.coordination_runtime_lazy": True,
    }


def test_evaluate_minimum_failures_reports_below_minimum():
    module = _load_benchmark_module()
    report = {
        "activation_probe": {
            "framework_registry_attempted_total": 1.0,
            "framework_registry_applied_total": 2.0,
        }
    }
    minimums = {
        "activation_probe.framework_registry_attempted_total": 2.0,
        "activation_probe.framework_registry_applied_total": 2.0,
    }

    failures = module._evaluate_minimum_failures(report, minimums)

    assert len(failures) == 1
    assert "framework_registry_attempted_total" in failures[0]


def test_evaluate_flag_expectation_failures_reports_mismatch():
    module = _load_benchmark_module()
    report = {
        "activation_probe": {
            "runtime_flags": {
                "generic_result_cache_enabled": False,
                "framework_preload_enabled": True,
                "coordination_runtime_lazy": False,
            }
        }
    }
    expectations = {
        "activation_probe.runtime_flags.generic_result_cache_enabled": True,
        "activation_probe.runtime_flags.framework_preload_enabled": True,
        "activation_probe.runtime_flags.coordination_runtime_lazy": True,
    }

    failures = module._evaluate_flag_expectation_failures(report, expectations)

    assert len(failures) == 2
    assert "generic_result_cache_enabled" in failures[0]
    assert any("coordination_runtime_lazy" in failure for failure in failures)


def test_measure_discovery_probe_returns_summary(monkeypatch):
    module = _load_benchmark_module()
    monkeypatch.setattr(
        module,
        "_run_snippet",
        lambda **_: {
            "cold_ms": 18.0,
            "warm_ms": [1.0, 2.0, 3.0],
            "warm_mean_ms": 2.0,
            "warm_p95_ms": 3.0,
            "discovered_count": 2,
            "discovered_vertical_names": ["coding", "research"],
            "call_total": 4,
            "cache_hit_total": 3,
            "scan_total": 1,
            "last_discovery_ms": 18.0,
            "entry_point_groups_cached": 1,
            "entry_point_vertical_entries": 2,
        },
    )

    probe = module._measure_discovery_probe(
        python_executable=sys.executable,
        iterations=3,
        timeout_seconds=30.0,
    )

    assert probe["cold_ms"] == 18.0
    assert probe["warm_samples_ms"] == [1.0, 2.0, 3.0]
    assert probe["warm_mean_ms"] == 2.0
    assert probe["cache_hit_total"] == 3
    assert probe["discovered_vertical_names"] == ["coding", "research"]


def test_measure_discovery_probe_targets_canonical_plugin_cache_group(monkeypatch):
    module = _load_benchmark_module()
    captured: dict[str, object] = {}

    def _fake_run_snippet(**kwargs):
        captured.update(kwargs)
        return {
            "cold_ms": 18.0,
            "warm_ms": [1.0],
            "warm_mean_ms": 1.0,
            "warm_p95_ms": 1.0,
            "discovered_count": 0,
            "discovered_vertical_names": [],
            "call_total": 1,
            "cache_hit_total": 0,
            "scan_total": 1,
            "last_discovery_ms": 18.0,
            "entry_point_groups_cached": 1,
            "entry_point_vertical_entries": 0,
        }

    monkeypatch.setattr(module, "_run_snippet", _fake_run_snippet)

    module._measure_discovery_probe(
        python_executable=sys.executable,
        iterations=1,
        timeout_seconds=30.0,
    )

    code = str(captured["code"])
    assert '"victor.plugins"' in code
    assert '"victor.verticals"' not in code


def test_main_can_run_discovery_probe_without_agent_create(monkeypatch, capsys):
    module = _load_benchmark_module()
    monkeypatch.setattr(
        module,
        "_measure_import_victor",
        lambda **_: module._TimingSummary(cold_ms=10.0, warm_samples_ms=[1.0, 1.0]),
    )

    def _unexpected_agent_create(**_):
        raise AssertionError("Agent.create probe should have been skipped")

    monkeypatch.setattr(module, "_measure_agent_create", _unexpected_agent_create)
    monkeypatch.setattr(
        module,
        "_measure_discovery_probe",
        lambda **_: {
            "cold_ms": 30.0,
            "warm_samples_ms": [3.0, 3.0],
            "warm_mean_ms": 3.0,
            "warm_p95_ms": 3.0,
            "discovered_count": 0,
            "discovered_vertical_names": [],
            "call_total": 3,
            "cache_hit_total": 2,
            "scan_total": 1,
            "last_discovery_ms": 30.0,
            "entry_point_groups_cached": 1,
            "entry_point_vertical_entries": 0,
        },
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "benchmark_startup_kpi.py",
            "--json",
            "--skip-agent-create",
            "--discovery-probe",
        ],
    )

    exit_code = module.main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert "agent_create" not in payload
    assert payload["discovery_probe"]["cache_hit_total"] == 2
    assert payload["discovery_probe"]["discovered_count"] == 0


def test_main_returns_exit_code_2_when_thresholds_fail(monkeypatch, capsys):
    module = _load_benchmark_module()
    monkeypatch.setattr(
        module,
        "_measure_import_victor",
        lambda **_: module._TimingSummary(cold_ms=150.0, warm_samples_ms=[1.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(
        module,
        "_measure_agent_create",
        lambda **_: module._TimingSummary(cold_ms=90.0, warm_samples_ms=[10.0, 12.0, 14.0]),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "benchmark_startup_kpi.py",
            "--json",
            "--max-import-cold-ms",
            "120",
        ],
    )

    exit_code = module.main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 2
    assert payload["thresholds"]["import_victor.cold_ms"] == 120.0
    assert payload["threshold_failures"]
    assert "import_victor.cold_ms" in payload["threshold_failures"][0]


def test_main_returns_exit_code_2_when_activation_expectations_fail(monkeypatch, capsys):
    module = _load_benchmark_module()
    monkeypatch.setattr(
        module,
        "_measure_import_victor",
        lambda **_: module._TimingSummary(cold_ms=10.0, warm_samples_ms=[1.0, 1.0]),
    )
    monkeypatch.setattr(
        module,
        "_measure_agent_create",
        lambda **_: module._TimingSummary(cold_ms=20.0, warm_samples_ms=[2.0, 2.0]),
    )
    monkeypatch.setattr(
        module,
        "_measure_activation_probe",
        lambda **_: {
            "vertical": "coding",
            "cold_ms": 30.0,
            "warm_samples_ms": [3.0, 3.0],
            "warm_mean_ms": 3.0,
            "warm_p95_ms": 3.0,
            "runtime_flags": {
                "generic_result_cache_enabled": False,
                "http_connection_pool_enabled": True,
                "framework_preload_enabled": True,
                "coordination_runtime_lazy": False,
                "interaction_runtime_lazy": True,
            },
            "framework_registry_metrics": {"workflows": {"attempted": 1, "applied": 1}},
            "framework_registry_attempted_total": 1,
            "framework_registry_applied_total": 1,
        },
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "benchmark_startup_kpi.py",
            "--json",
            "--activation-vertical",
            "coding",
            "--max-activation-cold-ms",
            "20",
            "--require-generic-result-cache-enabled",
            "--require-coordination-runtime-lazy",
            "--min-framework-registry-attempted-total",
            "2",
        ],
    )

    exit_code = module.main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 2
    assert "activation_probe" in payload
    assert payload["threshold_failures"]
    assert any("generic_result_cache_enabled" in item for item in payload["threshold_failures"])
    assert any("coordination_runtime_lazy" in item for item in payload["threshold_failures"])
    assert any("activation_probe.cold_ms" in item for item in payload["threshold_failures"])
    assert any(
        "framework_registry_attempted_total" in item for item in payload["threshold_failures"]
    )


def test_main_returns_exit_code_2_when_discovery_expectations_fail(monkeypatch, capsys):
    module = _load_benchmark_module()
    monkeypatch.setattr(
        module,
        "_measure_import_victor",
        lambda **_: module._TimingSummary(cold_ms=10.0, warm_samples_ms=[1.0, 1.0]),
    )
    monkeypatch.setattr(
        module,
        "_measure_discovery_probe",
        lambda **_: {
            "cold_ms": 35.0,
            "warm_samples_ms": [4.0, 4.0],
            "warm_mean_ms": 4.0,
            "warm_p95_ms": 4.0,
            "discovered_count": 0,
            "discovered_vertical_names": [],
            "call_total": 3,
            "cache_hit_total": 1,
            "scan_total": 1,
            "last_discovery_ms": 35.0,
            "entry_point_groups_cached": 1,
            "entry_point_vertical_entries": 0,
        },
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "benchmark_startup_kpi.py",
            "--json",
            "--skip-agent-create",
            "--discovery-probe",
            "--max-discovery-cold-ms",
            "20",
            "--min-discovery-cache-hits",
            "2",
        ],
    )

    exit_code = module.main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 2
    assert any("discovery_probe.cold_ms" in item for item in payload["threshold_failures"])
    assert any("discovery_probe.cache_hit_total" in item for item in payload["threshold_failures"])
