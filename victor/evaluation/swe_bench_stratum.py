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

"""SWE-bench trajectory → judge-calibration stratum (FEP-0030 / EVR).

The judge-independence findings (benchmarks/judge_training/FINDINGS.md) hit a
wall the calibration corpus can't clear: 6 synthetic templates give
trajectory/noise generalization only, real-agent runs on that corpus are ~98%
positive (no negatives), and the eval pack has ~2 negatives/family. A judge
trained/gated there says nothing about live completion decisions on real code.

The SWE-bench runner already captures what a completion judge needs, per
instance, in its ``eval_manifest_*.jsonl``: the issue (first message), the tool
activity, the agent's **patch** (``trace.file_edits`` diffs), and **in-container
FAIL_TO_PASS gold** (``status`` = passed/failed, requires ``--eval-backend
docker`` — host verification is unsound, see PR #811). This module renders those
into the blinded judge view every calibration judge scores, labeled by the
verifier — a REAL-distribution, both-class stratum for re-gating LLM judges and
training a shippable classifier.

Note the render is DIFF-based (the agent's patch), not a workspace snapshot: for
code-fix-at-scale the patch is the completing effect. It is therefore a distinct
input distribution from the calibration corpus — which is the whole point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_TOOLS = 30
_MAX_EDITS = 20
_MAX_DIFF_CHARS = 2000


@dataclass(frozen=True)
class StratumExample:
    """One (blinded judge view, verifier gold) pair from a SWE-bench instance."""

    task_id: str
    family: str  # always "swe-bench"
    text: str
    label: int  # 1 resolved (status==passed) / 0 not

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "source": "swe-bench",
            "text": self.text,
            "label": self.label,
        }


def render_swe_bench_view(instance: dict[str, Any]) -> str:
    """Render a manifest instance into the blinded judge view.

    Sections mirror ``render_judged_content`` (TASK / TOOL ACTIVITY / FINAL
    RESPONSE) but replace WORKSPACE STATE with PATCH — the agent's file-edit
    diffs, the completing effect for a code-fix task. Verifier ``status`` is
    NEVER rendered (blinding).
    """
    trace = instance.get("trace") or {}
    messages = trace.get("messages") or []
    issue = next(
        (str(m.get("content", "")) for m in messages if m.get("role") == "user"),
        str(messages[0].get("content", "")) if messages else "",
    )
    final = next(
        (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "assistant"),
        "",
    )
    tool_lines = [
        f"- {t.get('name', '?')} {str(t.get('arguments', ''))[:120]}"
        for t in (trace.get("tool_calls") or [])[:_MAX_TOOLS]
    ]
    # Prefer the ground-truth combined patch (git diff of the workspace) — the
    # actual applied diff — falling back to the per-edit capture, which is often
    # empty even when the run modified files (and thus resolved the instance).
    patch_lines: list[str] = []
    generated = str(trace.get("generated_patch", "") or "")
    if generated.strip():
        patch_lines.append(generated[: _MAX_EDITS * _MAX_DIFF_CHARS])
    else:
        for e in (trace.get("file_edits") or [])[:_MAX_EDITS]:
            patch_lines.append(f"--- {e.get('path', '?')}")
            diff = str(e.get("diff", ""))[:_MAX_DIFF_CHARS]
            if diff:
                patch_lines.append(diff)

    return "\n".join(
        [
            "TASK:",
            issue,
            "",
            "TOOL ACTIVITY:",
            *(tool_lines or ["(none)"]),
            "",
            "FINAL RESPONSE:",
            final,
            "",
            "PATCH:",
            *(patch_lines or ["(no file edits)"]),
        ]
    )


def _instance_label(instance: dict[str, Any]) -> int:
    """Verifier gold: 1 iff the run resolved the instance (in-container pass)."""
    status = str(instance.get("status", "")).lower()
    if status in ("passed", "resolved"):
        return 1
    return 0


def manifest_to_stratum(manifest_path: Path) -> list[StratumExample]:
    """Convert a SWE-bench ``eval_manifest_*.jsonl`` into stratum examples.

    Skips instances with error/unrunnable status (no meaningful completion
    verdict); one example per resolved-or-failed instance.
    """
    out: list[StratumExample] = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        inst = json.loads(line)
        status = str(inst.get("status", "")).lower()
        if status in ("error", "running", ""):
            continue
        out.append(
            StratumExample(
                task_id=str(inst.get("task_id", "?")),
                family="swe-bench",
                text=render_swe_bench_view(inst),
                label=_instance_label(inst),
            )
        )
    return out


def write_stratum_jsonl(examples: list[StratumExample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict()) + "\n")
    return len(examples)
