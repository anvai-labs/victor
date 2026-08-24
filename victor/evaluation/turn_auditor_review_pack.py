# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Build and finalize independently labelled EVR-6 review packs.

``export`` converts bounded real-run ``eval_manifest_*.jsonl`` records to HTIR and leaves every
case explicitly pending. ``finalize`` refuses pending cases, strips reviewer-only context, and
emits the label-pack schema consumed by :mod:`victor.evaluation.turn_auditor_evidence`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall
from victor.evaluation.htir import HTIRTrace, normalize
from victor.evaluation.turn_auditor_eval import case_from_dict

_MAX_CONTEXT_CHARS = 2000


@dataclass(frozen=True)
class ManifestSource:
    """One execution manifest and its independently assigned task family."""

    path: Path
    family: Optional[str] = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        rendered = value
    else:
        rendered = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return rendered[:_MAX_CONTEXT_CHARS]


def _messages(raw_trace: Mapping[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw_message in raw_trace.get("messages", []) or []:
        if not isinstance(raw_message, Mapping):
            continue
        messages.append(
            {
                "role": str(raw_message.get("role", "")),
                "content": str(raw_message.get("content", "")),
            }
        )
    return messages


def _tool_calls(raw_trace: Mapping[str, Any]) -> list[EvalToolCall]:
    calls: list[EvalToolCall] = []
    for raw_call in raw_trace.get("tool_calls", []) or []:
        if not isinstance(raw_call, Mapping):
            continue
        raw_arguments = raw_call.get("arguments", {})
        arguments = dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}
        try:
            timestamp = float(raw_call.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        result = raw_call.get("result")
        calls.append(
            EvalToolCall(
                name=str(raw_call.get("name", "")),
                arguments=arguments,
                result=str(result) if result is not None else None,
                success=bool(raw_call.get("success", True)),
                timestamp=timestamp,
            )
        )
    return calls


def _family(source: ManifestSource, record: Mapping[str, Any]) -> str:
    if source.family and source.family.strip():
        return source.family.strip()
    raw_trace = record.get("trace", {})
    trace = raw_trace if isinstance(raw_trace, Mapping) else {}
    inferred = str(record.get("family") or record.get("benchmark") or trace.get("benchmark") or "")
    if not inferred.strip():
        raise ValueError(f"{source.path.name}: task family is missing; use --source FAMILY=PATH")
    return inferred.strip()


def _normalized_trace(
    *,
    record: Mapping[str, Any],
    task_id: str,
    family: str,
    source_name: str,
    source_digest: str,
    line_number: int,
) -> HTIRTrace:
    raw_value = record.get("trace", {})
    raw_trace = raw_value if isinstance(raw_value, Mapping) else {}
    trace = AgenticExecutionTrace(
        task_id=task_id,
        start_time=0.0,
        benchmark=family,
        session_id=str(record.get("session_id") or raw_trace.get("session_id") or ""),
        turns=int(raw_trace.get("turns", record.get("turns", 0)) or 0),
        messages=_messages(raw_trace),
        tool_calls=_tool_calls(raw_trace),
    )
    normalized = normalize(trace)
    metadata = dict(normalized.metadata)
    metadata.update(
        {
            "source": {
                "kind": "eval_manifest",
                "manifest": source_name,
                "manifest_sha256": source_digest,
                "line": line_number,
                "original_task_id": str(record.get("task_id", "")),
            },
            "outcome": {
                "status": str(record.get("status", "")),
                "reward": str(record.get("reward", "")),
                "tests_passed": int(record.get("tests_passed", 0) or 0),
                "tests_total": int(record.get("tests_total", 0) or 0),
            },
        }
    )
    return HTIRTrace(
        task_id=normalized.task_id,
        steps=normalized.steps,
        session_id=normalized.session_id,
        benchmark=normalized.benchmark,
        metadata=metadata,
    )


def _review_context(record: Mapping[str, Any]) -> dict[str, Any]:
    raw_value = record.get("trace", {})
    raw_trace = raw_value if isinstance(raw_value, Mapping) else {}
    messages = _messages(raw_trace)
    user_message = next(
        (message["content"] for message in messages if message["role"] == "user"), ""
    )
    final_message = next(
        (message["content"] for message in reversed(messages) if message["role"] == "assistant"),
        "",
    )
    tools = []
    for index, raw_call in enumerate(raw_trace.get("tool_calls", []) or []):
        if not isinstance(raw_call, Mapping):
            continue
        tools.append(
            {
                "step_index": index,
                "tool_name": str(raw_call.get("name", "")),
                "arguments": _bounded(raw_call.get("arguments", {})),
                "success": bool(raw_call.get("success", True)),
                "result": _bounded(raw_call.get("result")),
            }
        )
    return {
        "task": _bounded(user_message),
        "tool_steps": tools,
        "final_response": _bounded(final_message),
    }


def build_review_pack(
    sources: Sequence[ManifestSource], *, include_context: bool = True
) -> dict[str, Any]:
    """Convert real-run manifests into a deterministic, entirely pending review pack."""
    cases: list[dict[str, Any]] = []
    seen_tasks: set[tuple[str, str]] = set()
    source_records: list[dict[str, Any]] = []
    skipped = {"missing_task_id": 0, "empty_trace": 0, "duplicate_task_id": 0}
    input_records = 0

    for source in sources:
        raw_bytes = source.path.read_bytes()
        digest = _sha256(raw_bytes)
        source_family = source.family.strip() if source.family else None
        source_records.append(
            {
                "manifest": source.path.name,
                "manifest_sha256": digest,
                "family": source_family,
            }
        )
        for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            input_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source.path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"{source.path.name}:{line_number}: record must be an object")
            original_task_id = str(record.get("task_id", "")).strip()
            if not original_task_id:
                skipped["missing_task_id"] += 1
                continue
            family = _family(source, record)
            dedupe_key = (family, original_task_id)
            if dedupe_key in seen_tasks:
                skipped["duplicate_task_id"] += 1
                continue
            task_id = f"{family}:{original_task_id}"
            trace = _normalized_trace(
                record=record,
                task_id=task_id,
                family=family,
                source_name=source.path.name,
                source_digest=digest,
                line_number=line_number,
            )
            if not trace.steps:
                skipped["empty_trace"] += 1
                continue
            seen_tasks.add(dedupe_key)
            case: dict[str, Any] = {
                "task_id": task_id,
                "family": family,
                "trace": trace.to_dict(),
                "review": {
                    "status": "pending",
                    "oracle_alarm_step": None,
                    "oracle_source": "",
                    "notes": "",
                },
            }
            if include_context:
                case["review_context"] = _review_context(record)
            cases.append(case)

    return {
        "schema_version": 1,
        "kind": "evr6_review_pack",
        "producer": "victor.evaluation.turn_auditor_review_pack",
        "sources": source_records,
        "selection": {
            "input_records": input_records,
            "emitted_cases": len(cases),
            "skipped": skipped,
            "dedupe_key": "family + original_task_id; first source occurrence wins",
        },
        "cases": cases,
    }


def finalize_review_pack(review_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate completed independent review and emit an evidence-ready label pack."""
    if int(review_payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported EVR-6 review-pack schema_version")
    if review_payload.get("kind") != "evr6_review_pack":
        raise ValueError("input is not an EVR-6 review pack")
    if "auditor_id" in review_payload:
        raise ValueError("EVR-6 review packs must not contain an auditor identity")
    raw_cases = review_payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("EVR-6 review-pack cases must be a list")

    label_cases: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"review case {index} must be an object")
        task_id = str(raw_case.get("task_id", ""))
        if not task_id or task_id in seen_task_ids:
            raise ValueError(f"review case {index} has a missing or duplicate task id")
        seen_task_ids.add(task_id)
        if "observations" in raw_case:
            raise ValueError(f"{task_id}: review cases must not contain model observations")
        review = raw_case.get("review", {})
        if not isinstance(review, Mapping):
            raise ValueError(f"{task_id}: review must be an object")
        status = str(review.get("status", "pending")).lower()
        if status == "excluded":
            continue
        if status != "included":
            raise ValueError(f"{task_id}: review status must be included or excluded")
        oracle_source = str(review.get("oracle_source", "")).strip()
        if not oracle_source:
            raise ValueError(f"{task_id}: included case is missing oracle_source")
        alarm_step = review.get("oracle_alarm_step")
        if alarm_step is not None and (
            isinstance(alarm_step, bool) or not isinstance(alarm_step, int)
        ):
            raise ValueError(f"{task_id}: oracle_alarm_step must be an integer or null")
        label_case = {
            "task_id": task_id,
            "family": str(raw_case.get("family", "")),
            "oracle_source": oracle_source,
            "oracle_alarm_step": alarm_step,
            "trace": raw_case.get("trace", {}),
        }
        parsed = case_from_dict(label_case)
        n_steps = len(parsed.trace.steps)
        if not parsed.family.strip() or n_steps == 0:
            raise ValueError(f"{task_id}: included case has an invalid family or empty trace")
        if [step.index for step in parsed.trace.steps] != list(range(n_steps)):
            raise ValueError(f"{task_id}: HTIR step indices are not sequential")
        if parsed.oracle_alarm_step is not None and not 0 <= parsed.oracle_alarm_step < n_steps:
            raise ValueError(f"{task_id}: oracle_alarm_step is outside the HTIR trace")
        label_cases.append(label_case)

    canonical_review = json.dumps(
        review_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "label_provenance": {
            "producer": "victor.evaluation.turn_auditor_review_pack",
            "review_pack_sha256": _sha256(canonical_review),
            "included_cases": len(label_cases),
            "excluded_cases": len(raw_cases) - len(label_cases),
        },
        "cases": label_cases,
    }


def _source(value: str) -> ManifestSource:
    family: Optional[str] = None
    path_value = value
    if "=" in value:
        family, path_value = value.split("=", 1)
        if not family.strip():
            raise argparse.ArgumentTypeError("source family cannot be empty")
    path = Path(path_value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"manifest does not exist: {path}")
    return ManifestSource(path=path, family=family)


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Build a pending review pack from manifests")
    export.add_argument(
        "--source",
        action="append",
        type=_source,
        required=True,
        help="Manifest PATH or explicit FAMILY=PATH; repeat for multiple sources",
    )
    export.add_argument("--output", required=True, help="Destination review-pack JSON")
    export.add_argument(
        "--without-context",
        action="store_true",
        help="Omit bounded task/tool/final-response reviewer context",
    )

    finalize = subparsers.add_parser("finalize", help="Validate review and emit a label pack")
    finalize.add_argument("review_pack", help="Completed review-pack JSON")
    finalize.add_argument("--output", required=True, help="Destination label-pack JSON")
    args = parser.parse_args(argv)

    if args.command == "export":
        payload = build_review_pack(args.source, include_context=not args.without_context)
    else:
        review_payload = json.loads(Path(args.review_pack).read_text(encoding="utf-8"))
        payload = finalize_review_pack(review_payload)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "cases": len(payload["cases"])}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
