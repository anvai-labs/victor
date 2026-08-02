# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tests for victor.runtime.reachability (FEP-0022 Phase 1).

Covers: activate/record/flush semantics, dedup, contextvars scoping + nesting,
container DI-witness integration (registered-only), env arming, and the
disarm-overhead microbenchmark that guards the production hot path.
"""

from __future__ import annotations

import json
import time

import pytest

from victor.core.container import ServiceContainer, ServiceLifetime
from victor.runtime.reachability import (
    ReachabilityRecorder,
    activate,
    current_recorder,
    is_env_armed,
    record,
    record_service_resolution,
)


class _DummyService:
    pass


class _Other:
    pass


# --- disarmed (the production default) --------------------------------------


def test_disarmed_by_default():
    assert current_recorder() is None
    assert is_env_armed() is False


def test_record_is_noop_when_disarmed():
    # No active recorder: record() must not raise and must leave nothing observable.
    record("di", "mod:A")
    record_service_resolution(_DummyService)
    assert current_recorder() is None


# --- armed: activate / record / dedup ---------------------------------------


def test_activate_records_and_dedups():
    with activate(run_id="t1") as rec:
        assert current_recorder() is rec
        record("di", "mod:A")
        record("di", "mod:B")
        record("di", "mod:A")  # duplicate -> deduped
        record_service_resolution(_DummyService)
    assert current_recorder() is None  # detached after exit

    observed = {(w["kind"], w["key"]) for w in rec.observed()}
    assert ("di", "mod:A") in observed
    assert ("di", "mod:B") in observed
    assert ("di", f"{__name__}:_DummyService") in observed
    assert len(observed) == 3  # 2 generic + 1 service-type, duplicates removed


def test_observed_is_deterministic_sorted():
    with activate(run_id="t-sort") as rec:
        record("di", "z:1")
        record("di", "a:1")
        record("di", "m:1")
    keys = [w["key"] for w in rec.observed()]
    assert keys == sorted(keys)


# --- container integration ---------------------------------------------------


def test_container_get_witnesses_registered_resolution():
    container = ServiceContainer()
    container.register(_DummyService, lambda c: _DummyService(), ServiceLifetime.SINGLETON)
    with activate(run_id="t2") as rec:
        container.get(_DummyService)
        container.get(_DummyService)  # repeated -> deduped to one witness
    observed = {(w["kind"], w["key"]) for w in rec.observed()}
    assert ("di", f"{__name__}:_DummyService") in observed
    assert len(observed) == 1


def test_unregistered_resolution_is_not_witnessed():
    # get() on an unregistered type raises in _get_descriptor, BEFORE the witness
    # fires — so unregistered types never appear in the reachability set.
    container = ServiceContainer()
    with activate(run_id="t3") as rec:
        with pytest.raises(Exception):
            container.get(_Other)
    assert rec.observed() == []


# --- sidecar flush -----------------------------------------------------------


def test_flush_writes_sidecar_jsonl(tmp_path):
    out = tmp_path / "reachability-run.jsonl"
    with activate(run_id="r1", out_path=out) as rec:
        record("di", "m:A")
        record("di", "m:B")
    assert out.exists()

    lines = out.read_text().splitlines()
    header = json.loads(lines[0])
    assert header["run_id"] == "r1"
    assert header["count"] == 2
    records = [json.loads(line) for line in lines[1:]]
    assert {"kind": "di", "key": "m:A"} in records
    assert {"kind": "di", "key": "m:B"} in records


def test_flush_without_out_path_is_noop():
    with activate(run_id="r2", out_path=None) as rec:
        record("di", "m:A")
    assert rec.flush() is None  # nothing written, no crash


# --- contextvars scoping -----------------------------------------------------


def test_nested_activate_restores_outer():
    with activate(run_id="outer") as outer:
        record("di", "o:1")
        with activate(run_id="inner") as inner:
            assert current_recorder() is inner
            record("di", "i:1")
        assert current_recorder() is outer  # restored
        record("di", "o:2")

    assert {w["key"] for w in inner.observed()} == {"i:1"}
    assert {w["key"] for w in outer.observed()} == {"o:1", "o:2"}


def test_exceptions_do_not_leak_recorder():
    with pytest.raises(RuntimeError):
        with activate(run_id="exc"):
            record("di", "x:1")
            raise RuntimeError("boom")
    assert current_recorder() is None  # contextvar reset despite the raise


# --- env arming --------------------------------------------------------------


def test_is_env_armed(monkeypatch):
    monkeypatch.setenv("VICTOR_REACHABILITY_RECORD", "1")
    assert is_env_armed() is True
    monkeypatch.setenv("VICTOR_REACHABILITY_RECORD", "false")
    assert is_env_armed() is False


def test_activate_uses_env_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VICTOR_REACHABILITY_OUT", str(tmp_path))
    monkeypatch.setenv("VICTOR_REACHABILITY_RECORD", "1")
    with activate(run_id="envrun"):
        record("di", "m:A")
    sidecar = tmp_path / "reachability-envrun.jsonl"
    assert sidecar.exists()


# --- hot-path microbenchmark (FEP-0022 mandate: disarmed = near no-op) ------


def test_disarmed_witness_is_near_noop():
    """The disarmed fast path must not regress the container hot path.

    Asserts the disarmed call is (a) faster than the armed call and (b) under a
    generous absolute cap. Machine-independent on the ratio; the cap catches
    real regressions (e.g. accidental string work or allocation on the path).
    """
    n = 100_000

    assert current_recorder() is None
    t0 = time.perf_counter()
    for _ in range(n):
        record_service_resolution(_DummyService)
    disarmed_ns = (time.perf_counter() - t0) / n * 1e9

    with activate(run_id="bench"):
        t0 = time.perf_counter()
        for _ in range(n):
            record_service_resolution(_DummyService)
        armed_ns = (time.perf_counter() - t0) / n * 1e9

    assert (
        disarmed_ns < armed_ns
    ), f"disarmed ({disarmed_ns:.0f}ns) not faster than armed ({armed_ns:.0f}ns)"
    assert disarmed_ns < 2000, f"disarmed witness too slow: {disarmed_ns:.0f}ns/op"
