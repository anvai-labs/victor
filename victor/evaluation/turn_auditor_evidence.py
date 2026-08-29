# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Produce EVR-6 evidence by replaying independently labelled HTIR prefixes.

The input is a label pack with HTIR traces and oracle fields but no model observations. The
producer invokes the production ``PerTurnAuditor`` for every prefix, records latency, and emits
the schema consumed by :mod:`victor.evaluation.turn_auditor_eval`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Protocol

from victor.evaluation.turn_auditor_eval import (
    HTIRAuditorCase,
    TurnAuditObservation,
    assess_evidence_payload,
    case_from_dict,
)
from victor.framework.per_turn_auditor import AuditSignal


class PrefixAuditor(Protocol):
    """Small protocol implemented by :class:`PerTurnAuditor`."""

    def audit_turn(self, action_result: Any, state: Optional[dict[str, Any]] = None) -> AuditSignal:
        """Judge one turn without mutating the completion decision."""


@dataclass(frozen=True)
class OllamaModelIdentity:
    """Exact Ollama tag and content digest used for an evidence run."""

    name: str
    digest: str

    @property
    def auditor_id(self) -> str:
        return f"ollama:{self.name}@sha256:{self.digest}"


def resolve_ollama_identity(
    *, base_url: str, model: str, expected_digest: Optional[str] = None
) -> OllamaModelIdentity:
    """Resolve an exact local Ollama tag to its immutable SHA-256 digest."""
    import httpx

    response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
    response.raise_for_status()
    matches = [entry for entry in response.json().get("models", []) if entry.get("name") == model]
    if len(matches) != 1:
        raise ValueError(f"Ollama model tag {model!r} did not resolve exactly once")
    digest = str(matches[0].get("digest", "")).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Ollama model tag {model!r} has no valid SHA-256 digest")
    if expected_digest is not None:
        normalized = expected_digest.removeprefix("sha256:").lower()
        if digest != normalized:
            raise ValueError(
                f"Ollama digest mismatch for {model!r}: resolved {digest}, expected {normalized}"
            )
    return OllamaModelIdentity(model, digest)


def warm_ollama_model(*, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
    """Load the pinned model before latency measurements without recording an observation."""
    import httpx

    response = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": "Reply READY.",
            "stream": False,
            "think": False,
            "keep_alive": "15m",
            "options": {"num_predict": 8, "temperature": 0},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("model") != model or not str(payload.get("response", "")).strip():
        raise RuntimeError(f"Ollama warmup for {model!r} returned an invalid response")


def _action_result(case: HTIRAuditorCase, step_index: int) -> SimpleNamespace:
    """Adapt one HTIR step to the runtime's TurnResult-shaped auditor input."""
    step = case.trace.steps[step_index]
    tool_results: list[dict[str, Any]] = []
    if step.role.value == "tool":
        tool_results.append({"name": step.tool_name, "success": not step.is_failure})
    return SimpleNamespace(content=step.summary, tool_results=tool_results)


def produce_evidence_payload(
    label_payload: Mapping[str, Any],
    *,
    auditor: PrefixAuditor,
    auditor_id: str,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Replay every labelled HTIR prefix and return a complete evidence payload.

    Existing observations are rejected so model output can never masquerade as an independent
    oracle label or silently be overwritten by a later run.
    """
    if int(label_payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported EVR-6 label-pack schema_version")
    if "auditor_id" in label_payload:
        raise ValueError("EVR-6 label packs must not contain an auditor identity")
    raw_cases = label_payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("EVR-6 label-pack cases must be a list")
    if any(not isinstance(raw_case, Mapping) for raw_case in raw_cases):
        raise ValueError("every EVR-6 label-pack case must be an object")
    if any("observations" in raw_case for raw_case in raw_cases):
        raise ValueError("EVR-6 label packs must not contain model observations")

    cases: list[HTIRAuditorCase] = []
    for raw_case in raw_cases:
        case = case_from_dict(raw_case)
        observations: list[TurnAuditObservation] = []
        for step_index in range(len(case.trace.steps)):
            state = {
                "task_id": case.task_id,
                "htir_prefix": [step.to_dict() for step in case.trace.steps[: step_index + 1]],
            }
            started = clock()
            signal = auditor.audit_turn(_action_result(case, step_index), state)
            elapsed_ms = (clock() - started) * 1000.0
            observations.append(
                TurnAuditObservation(
                    step_index=step_index,
                    verdict=signal.verdict,
                    latency_ms=elapsed_ms,
                    reason=signal.reason,
                )
            )
        cases.append(
            HTIRAuditorCase(
                trace=case.trace,
                observations=tuple(observations),
                oracle_alarm_step=case.oracle_alarm_step,
                family=case.family,
                oracle_source=case.oracle_source,
            )
        )

    canonical_labels = json.dumps(
        label_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "auditor_id": auditor_id,
        "producer": {
            "name": "victor.evaluation.turn_auditor_evidence",
            "label_pack_sha256": hashlib.sha256(canonical_labels).hexdigest(),
            "cache_enabled": False,
        },
        "cases": [case.to_dict() for case in cases],
    }


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", help="Independent HTIR label-pack JSON")
    parser.add_argument("--output", required=True, help="Destination evidence JSON")
    parser.add_argument("--report", help="Optional PASS/HOLD report JSON")
    parser.add_argument("--model", default="qwen3.5:2b", help="Exact Ollama model tag")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--expected-digest", help="Optional required sha256 digest")
    args = parser.parse_args(argv)

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    raw_cases = labels.get("cases", [])
    prefix_count = sum(
        len(raw_case.get("trace", {}).get("steps", []))
        for raw_case in raw_cases
        if isinstance(raw_case, Mapping)
    )

    identity = resolve_ollama_identity(
        base_url=args.base_url,
        model=args.model,
        expected_digest=args.expected_digest,
    )
    warm_ollama_model(base_url=args.base_url, model=args.model)

    from victor.agent.edge_model import EdgeModelConfig, create_edge_decision_service
    from victor.agent.edge_turn_judge import EdgeTurnJudge
    from victor.framework.per_turn_auditor import PerTurnAuditor, PerTurnAuditorConfig

    service = create_edge_decision_service(
        EdgeModelConfig(
            model=args.model,
            base_url=args.base_url,
            timeout_ms=args.timeout_ms,
            cache_ttl=0,
            max_tokens=128,
            micro_budget=max(20, prefix_count),
        )
    )
    if service is None:
        raise RuntimeError(f"could not create edge decision service for {args.model!r}")
    auditor = PerTurnAuditor(PerTurnAuditorConfig(enabled=True), judge=EdgeTurnJudge(service))
    evidence = produce_evidence_payload(
        labels,
        auditor=auditor,
        auditor_id=identity.auditor_id,
    )

    # A mutable tag changing during a long battery invalidates the entire artifact.
    final_identity = resolve_ollama_identity(
        base_url=args.base_url,
        model=args.model,
        expected_digest=identity.digest,
    )
    if final_identity != identity:  # defensive; expected_digest already enforces this
        raise RuntimeError("Ollama model identity changed during evidence production")

    Path(args.output).write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    report = assess_evidence_payload(evidence)
    rendered_report = json.dumps(report.to_dict(), indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(rendered_report, encoding="utf-8")
    print(rendered_report, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
