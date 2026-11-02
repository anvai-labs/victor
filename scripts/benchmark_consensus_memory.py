#!/usr/bin/env python3
"""Measure bounded echo formations against an immutable, preview-bounded baseline.

Run from the checkout being measured. Reports Python allocation peaks, CPU and
wall time for the whole formation, and requires exact returned-result parity.
An input-size guard refuses exponential-growth regressions before large tasks
can exhaust the machine. This is a synthetic memory workload, not a model-speed
or total-process-RSS claim.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import victor.coordination.formations.consensus as candidate_module
from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.consensus import ConsensusFormation
from victor.teams.types import AgentMessage, MemberResult, MessageType

BASELINE = "22f9d5427661583fcd105194df90c6031716c9aa"


async def measure(formation: Any, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure one full 15-round run with deterministic, disagreeing echo members."""
    largest = 0
    agents = []
    for index in range(count):

        async def echo(
            message: AgentMessage, context: TeamContext, index: int = index
        ) -> MemberResult:
            nonlocal largest
            largest = max(largest, len(message.content))
            if largest >= 2_000_000:
                raise RuntimeError("Benchmark refuses unsafe task growth")
            return MemberResult(
                member_id=str(index), success=True, output=f"{index}: {message.content}"
            )

        agents.append(SimpleNamespace(id=str(index), execute=echo))
    tracemalloc.start()
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        results = await formation(max_rounds=15).execute(
            agents,
            TeamContext("benchmark", "consensus"),
            AgentMessage(sender_id="user", content="original task", message_type=MessageType.TASK),
        )
        elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if any(not result.success for result in results):
        raise RuntimeError("Benchmark refused failed or unsafe member results")
    return [item.to_dict() for item in results], {
        "wall_seconds": elapsed_wall,
        "cpu_seconds": elapsed_cpu,
        "peak_python_bytes": peak,
        "largest_task_chars": largest,
    }


async def benchmark(baseline: str) -> dict[str, Any]:
    """Compare the exact checkout owner and one explicit baseline commit."""
    if (
        Path(candidate_module.__file__).resolve()
        != ROOT / "victor/coordination/formations/consensus.py"
    ):
        raise RuntimeError("Installed editable module shadows the candidate checkout")
    source = subprocess.run(
        ["git", "show", f"{baseline}:victor/coordination/formations/consensus.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module_name = "victor.coordination.formations.consensus_baseline"
    rows = []
    with tempfile.TemporaryDirectory(prefix="victor-consensus-baseline-") as directory:
        path = Path(directory) / "consensus_baseline.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the immutable consensus baseline")
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            for count in (2, 8, 16):
                before, old = await measure(module.ConsensusFormation, count)
                after, new = await measure(ConsensusFormation, count)
                if before != after:
                    raise RuntimeError("Consensus result parity failed")
                rows.append(
                    {
                        "members": count,
                        "rounds": 15,
                        "parity": True,
                        "baseline": old,
                        "candidate": new,
                    }
                )
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
    return {
        "candidate_module": candidate_module.__file__,
        "baseline": baseline,
        "capture": False,
        "checkpoint": False,
        "runs": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args.baseline)), indent=2))
