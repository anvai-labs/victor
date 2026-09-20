#!/usr/bin/env python3
"""Opt-in all-formation artifact matrix using ZAI or InferFlux through Sandhi.

Run once per compared model; every failure remains evidence. The legacy six-case
battery and default runtime behavior are unchanged. Output directories are private.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

from scripts.validation.formation_matrix_cases import CASE_NAMES, MatrixCase, build_case
from scripts.validation.gateway_matrix_accounting import BUFFERED_DEADLINE_SECONDS, GatewayObserver
from victor.framework import Agent
from victor.framework.member_event_sink import MemberEventSink, current_member_sink
from victor.teams.types import TeamResult


async def check_case(
    case: MatrixCase,
    result: TeamResult | None,
    root: Path,
    python: Path,
    model: str,
    requests: list[dict[str, Any]],
    observer: GatewayObserver,
) -> dict[str, Any]:
    """Run independent checks even when the team returned failure or no result."""
    report: dict[str, Any] = {
        "failures": [],
        "pytest": [],
        "usage_reconciliation": [],
        "artifacts": {},
    }
    results = result.member_results if result is not None else {}
    members = {member.name: member for member in case.team._config.members}
    expected = {members[name].id for name in case.executed_names}
    if (
        result is None
        or not result.success
        or set(results) != expected
        or not all(member.success for member in results.values())
    ):
        report["failures"].append("member_outcomes")
    sessions = [member.metadata.get("session_id") for member in results.values()]
    if (
        not sessions
        or any(not isinstance(session, str) or not session for session in sessions)
        or len(set(sessions)) != len(expected)
        or {request["session_id"] for request in requests} != set(sessions)
    ):
        report["failures"].append("distinct_member_sessions")
    if not requests or any(
        request["model"] != model
        or request["run_id"] != request["session_id"]
        or request.get("http_status") != 200
        or request.get("elapsed_seconds", BUFFERED_DEADLINE_SECONDS) >= BUFFERED_DEADLINE_SECONDS
        for request in requests
    ):
        report["failures"].append("wire_routes_or_attempts")
    for name, files in case.artifacts.items():
        member = results.get(members[name].id)
        directory = root
        if case.isolated:
            assignment = member.metadata.get("worktree_assignment", {}) if member else {}
            path = assignment.get("worktree_path")
            if not isinstance(path, str) or not path:
                report["failures"].append("missing_worktree:" + name)
                continue
            directory = Path(path).resolve()
            if not directory.is_relative_to((root / "members").resolve()):
                report["failures"].append("unexpected_worktree:" + name)
                continue
        missing = [filename for filename in files if not (directory / filename).is_file()]
        report["artifacts"][name] = {
            "directory": str(directory),
            "required": files,
            "missing": missing,
        }
        if missing:
            report["failures"].append("missing_artifacts:" + name)
        if name in case.pytest_names:
            filename = "test_member.py" if case.isolated else f"test_{name}.py"
            entry: dict[str, Any] = {"member": name, "returncode": None}
            report["pytest"].append(entry)
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    str(directory / filename),
                    cwd=directory,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    output, _ = await asyncio.wait_for(process.communicate(), timeout=60)
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                entry.update(returncode=process.returncode, output=output.decode(errors="replace"))
            except (OSError, TimeoutError) as exc:
                entry["error_type"] = type(exc).__name__
            if entry["returncode"] != 0:
                report["failures"].append("pytest:" + name)
    for member in results.values():
        entry = {"member_id": member.member_id, "passed": False}
        report["usage_reconciliation"].append(entry)
        try:
            session = member.metadata["session_id"]
            usage = member.metadata["usage"]
            total = (await observer.admin_call("/admin/usage/run/" + session))["run"]["total"]
            entry.update(session_id=session, victor=usage, sandhi=total)
            inclusive = (
                total["tokens_in"] + total["cache_read_tokens"] + total["cache_creation_tokens"]
            )
            entry["passed"] = (
                usage["input_tokens"] == inclusive
                and usage["output_tokens"] == total["tokens_out"]
                and usage["total_tokens"] == inclusive + total["tokens_out"]
            )
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
        if not entry["passed"]:
            report["failures"].append("member_usage:" + member.member_id)
    name = case.team._config.name
    try:
        if name == "consensus" and (result is None or result.consensus_achieved is not True):
            report["failures"].append("consensus_contract")
        if name == "reflection":
            review = json.loads((root / "review.json").read_text())
            report["review"] = review
            if not (
                isinstance(review, dict)
                and set(review) == {"verdict", "feedback"}
                and review["verdict"] == "satisfied"
                and isinstance(review["feedback"], str)
            ):
                report["failures"].append("reflection_contract")
        if name.startswith("ensemble_") and not all(
            member.metadata.get("ensemble_success") is True for member in results.values()
        ):
            report["failures"].append("ensemble_contract")
        if name in {"group_chat", "debate", "handoff"}:
            termination = "judge" if name == "debate" else "done"
            if not all(
                member.metadata.get("conversation_termination") == termination
                and member.metadata.get("conversation_success") is True
                for member in results.values()
            ):
                report["failures"].append("conversation_contract")
    except (OSError, ValueError, TypeError) as exc:
        report["failures"].append("formation_contract:" + type(exc).__name__)
    if name in {"debate", "ensemble_judge", "ensemble_synthesizer"}:
        role = "synthesizer" if name == "ensemble_synthesizer" else "judge"
        try:
            path = Path(report["artifacts"][role]["directory"]) / "decision.json"
            decision = json.loads(path.read_text())
            report["decision"] = decision
            fields = {"answer"} if role == "synthesizer" else {"selected_member_id"}
            if name == "debate":
                fields.add("verdict")
            if (
                not isinstance(decision, dict)
                or set(decision) != fields
                or any(
                    not isinstance(value, str) or not value.strip() for value in decision.values()
                )
            ):
                raise ValueError("Invalid decision contract")
            if role == "judge" and decision["selected_member_id"] not in {
                member.id for member in case.team._config.members if member.name != role
            }:
                raise ValueError("Decision selected an unknown candidate")
            if name == "ensemble_judge":
                judge = results[members[role].id]
                if decision != json.loads(judge.output):
                    raise ValueError("Persisted judge selection differs from runtime output")
            if name == "debate" and (result is None or decision["verdict"] != result.final_output):
                raise ValueError("Persisted verdict differs from runtime output")
            if (
                role == "synthesizer"
                and result is not None
                and decision["answer"] != result.final_output
            ):
                raise ValueError("Persisted synthesis differs from runtime output")
        except (KeyError, OSError, ValueError, TypeError) as exc:
            report["failures"].append("decision_contract:" + type(exc).__name__)
    report["passed"] = not report["failures"]
    return report


async def run_case(
    name: str, directory: Path, args: argparse.Namespace, observer: GatewayObserver
) -> dict[str, Any]:
    directory.mkdir()
    started = time.monotonic()
    offset = len(observer.records)
    record: dict[str, Any] = {"case": name, "passed": False, "failures": []}
    agent = None
    case = None
    result = None
    sink = MemberEventSink()
    token = current_member_sink.set(sink)
    try:
        if name.startswith("ensemble_"):
            for command in (
                ["init"],
                [
                    "-c",
                    "user.name=Validation",
                    "-c",
                    "user.email=validation@example.com",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "seed",
                ],
            ):
                subprocess.run(["git", *command], cwd=directory, check=True, capture_output=True)
        agent = await Agent.create(
            provider=args.provider,
            model=args.model,
            workspace=str(directory),
            enable_observability=False,
            session_id=directory.name,
        )
        if (
            agent.get_orchestrator().provider.extra_config["gateway"]["url"]
            != f"http://127.0.0.1:{args.proxy_port}"
        ):
            raise ValueError("Agent is not using the configured observer gateway")
        case = await build_case(
            agent.get_orchestrator(),
            name,
            directory,
            WORKTREE / ".venv-codesign/bin/python",
            args.timeout,
        )
        result = await asyncio.wait_for(case.team.run(), args.timeout)
        record["result"] = result.to_dict()
    except asyncio.CancelledError:
        record["failures"].append("execution:CancelledError")
        raise
    except Exception as exc:
        record["failures"].append("execution:" + type(exc).__name__)
    finally:
        current_member_sink.reset(token)
        sink_closed = False
        sink_cancelled = None
        try:
            await sink.close()
            sink_closed = True
        except asyncio.CancelledError as exc:
            record["failures"].append("sink:CancelledError")
            sink_cancelled = exc
        except Exception as exc:
            record["failures"].append("sink:" + type(exc).__name__)
        try:
            if case is not None:
                checks = await check_case(
                    case,
                    result,
                    directory,
                    WORKTREE / ".venv-codesign/bin/python",
                    args.model,
                    observer.records[offset:],
                    observer,
                )
                record["failures"].extend(checks.pop("failures"))
                record.update(checks)
        except asyncio.CancelledError:
            record["failures"].append("acceptance:CancelledError")
            raise
        except Exception as exc:
            record["failures"].append("acceptance:" + type(exc).__name__)
        finally:
            try:
                if agent is not None:
                    await agent.close()
            except asyncio.CancelledError:
                record["failures"].append("cleanup:CancelledError")
                raise
            except Exception as exc:
                record["failures"].append("cleanup:" + type(exc).__name__)
            finally:
                record.update(
                    passed=not record["failures"],
                    events=[asdict(event) async for event in sink.drain()] if sink_closed else [],
                    events_complete=sink_closed,
                    requests=observer.records[offset:],
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
                # The shared serializer preserves supported data and explicitly fails on opaque metadata.
                from scripts.validation.multiagent_gateway_live import save_report

                save_report(directory, record)
        if sink_cancelled is not None:
            raise sink_cancelled
    return record


async def main(args: argparse.Namespace) -> int:
    from scripts.validation.multiagent_gateway_live import save_report

    if not args.cases or len(set(args.cases)) != len(args.cases) or args.timeout <= 0:
        raise ValueError("Choose unique cases and a positive timeout before making model calls")
    root = Path(args.output_dir).resolve() / ("matrix-" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True, mode=0o700)
    config_name = "client.json" if args.provider == "zai" else "inferflux.json"
    config = json.loads((args.gateway_state / config_name).read_text())
    admin = (args.gateway_state / "admin-token").read_text().strip()
    environment = {
        "SANDHI_GATEWAY_URL": f"http://127.0.0.1:{args.proxy_port}",
        "SANDHI_GATEWAY_VIRTUAL_KEY_" + args.provider.upper(): config["virtual_key"],
        args.provider.upper() + "_API_KEY": config["virtual_key"],
    }
    prior = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    observer = GatewayObserver(
        config["url"], admin, args.gateway_state / "usage.db", root, args.proxy_port, args.provider
    )
    report: dict[str, Any] = {
        "provider": args.provider,
        "model": args.model,
        "cases": [],
        "failures": [{"check": "matrix_incomplete"}],
        "case_contract_version": 1,
        "timeout_seconds": args.timeout,
        "buffered_gateway_deadline_seconds": BUFFERED_DEADLINE_SECONDS,
        "shared_cache_cleared": False,
        "inference_reruns": 0,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        await observer.start()
        report["gateway_version"] = await observer.admin_call("/admin/version")
        for name in args.cases:
            case = await run_case(name, root / (root.name + "-" + name), args, observer)
            report["cases"].append(case)
            if not case["passed"]:
                report["failures"].append({"case": name, "check": "case_failed"})
            report["requests"] = observer.records
            save_report(root, report)
            print(
                json.dumps(
                    {
                        "case": name,
                        "passed": case["passed"],
                        "elapsed_seconds": case["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    except asyncio.CancelledError:
        report["failures"].append({"check": "execution", "error_type": "CancelledError"})
        raise
    except Exception as exc:
        report["failures"].append({"check": "execution", "error_type": type(exc).__name__})
    finally:
        try:
            accounting = await observer.collect()
            report["accounting"] = accounting
            report["failures"].extend(accounting["failures"])
        except asyncio.CancelledError:
            report["failures"].append({"check": "accounting", "error_type": "CancelledError"})
            raise
        except Exception as exc:
            report["failures"].append({"check": "accounting", "error_type": type(exc).__name__})
        finally:
            for key, value in prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            try:
                await observer.close()
            except asyncio.CancelledError:
                report["failures"].append(
                    {"check": "observer_cleanup", "error_type": "CancelledError"}
                )
                raise
            except Exception as exc:
                report["failures"].append(
                    {"check": "observer_cleanup", "error_type": type(exc).__name__}
                )
            finally:
                if len(report["cases"]) == len(args.cases) and "accounting" in report:
                    report["failures"] = [
                        item
                        for item in report["failures"]
                        if item.get("check") != "matrix_incomplete"
                    ]
                report["requests"] = observer.records
                report["limitations"] = [
                    "Buffered calls only; no streaming or origin-cancellation acceptance.",
                    "Reported cache counts do not establish backend executed reuse.",
                    "Runtime identities and hardware acceptance must accompany this report.",
                    "No tokenizer-equivalence or session-lease acceptance claim.",
                    "Unsupported durable partial resume remains explicitly deferred.",
                ]
                save_report(root, report)
    print(
        json.dumps(
            {
                "evidence": str(root / "evidence.json"),
                "passed": report["passed"],
                "cases_passed": sum(case["passed"] for case in report["cases"]),
                "cases": len(args.cases),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("inferflux", "zai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gateway-state", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--proxy-port", type=int, default=18084)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
