# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Training-data generation for small independent completion judges (E2).

Produces (rendered judge view, verifier gold) pairs at scale with zero LLM
calls: for each corpus task, run setup → scripted executor → render the SAME
blinded view every calibration judge scores (``render_judged_content``) →
compute the label from the task's programmatic verifier (κ=1.0-validated
against independent annotation, FINDINGS run 12).

Blinding note: the rendered text is the judge view by construction — verifier
verdicts are never part of it. The label rides NEXT TO the text in the
emitted example, never inside it (a test pins this).

Known scope limitation (record wherever a trained judge is reported): the
corpus has 6 task templates with parametric surface variation, so models
trained here learn trajectory/noise generalization within those templates —
not task-type generalization. The held-out real-trajectory gate shares the
same templates; SWE-bench-stratum data is the prerequisite for any claim
beyond this corpus.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from victor.evaluation.calibration_rubric_judge import render_judged_content
from victor.evaluation.judge_calibration_harness import (
    CalibrationExecutor,
    Transcript,
    VerifiableTask,
)


@dataclass(frozen=True)
class TrainingExample:
    """One (blinded judge view, verifier gold) supervision pair."""

    task_id: str
    family: str
    source: str  # executor description, e.g. "easy-p2" | "hard"
    text: str
    label: int  # 1 complete / 0 not, from the programmatic verifier

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "source": self.source,
            "text": self.text,
            "label": self.label,
        }


def generate_training_examples(
    tasks: Sequence[VerifiableTask],
    executor: CalibrationExecutor,
    *,
    source: str,
    workspace_root: Optional[Path] = None,
) -> list[TrainingExample]:
    """Run every task through the executor and emit supervision pairs.

    Mirrors the harness flow (setup → execute → render → verify) without any
    judges. Rendering happens BEFORE verification, same as the harness's
    blinding order, so verifier side effects can never reach the text.
    """
    caller_owned = workspace_root is not None
    root = workspace_root or Path(tempfile.mkdtemp(prefix="judge_training_"))
    examples: list[TrainingExample] = []
    try:
        for index, task in enumerate(tasks):
            workspace = root / f"{index:04d}_{task.task_id}"
            workspace.mkdir(parents=True, exist_ok=False)
            task.setup(workspace)
            transcript = executor(task, workspace)
            text = render_judged_content(task.prompt, transcript, workspace)
            gold = float(task.verify(workspace, transcript))
            examples.append(
                TrainingExample(
                    task_id=task.task_id,
                    family=task.family,
                    source=source,
                    text=text,
                    label=int(gold >= 0.5),
                )
            )
        return examples
    finally:
        if not caller_owned:
            shutil.rmtree(root, ignore_errors=True)


def write_jsonl(examples: Iterable[TrainingExample], path: Path) -> int:
    """Append-free JSONL write; returns the number of examples written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict()) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[TrainingExample]:
    examples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(
            TrainingExample(
                task_id=row["task_id"],
                family=row["family"],
                source=row["source"],
                text=row["text"],
                label=int(row["label"]),
            )
        )
    return examples


def split_by_variant(
    examples: Sequence[TrainingExample], *, dev_variant_start: int
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Train/dev split by the task's variant index (the ``-NN`` id suffix).

    Splitting by variant (not randomly) keeps every (family, variant) group
    wholly on one side, so near-duplicate surface variants cannot straddle
    the split and inflate dev scores.
    """
    train: list[TrainingExample] = []
    dev: list[TrainingExample] = []
    for example in examples:
        variant = int(example.task_id.rsplit("-", 1)[1])
        (dev if variant >= dev_variant_start else train).append(example)
    return train, dev


# ----------------------------------------------------------------------------------------------------
# Real-styled negative synthesis (E2 iteration 2)
# ----------------------------------------------------------------------------------------------------
#
# Both scripted-only and real-only training collapse (FINDINGS
# benchmarks/judge_training/): scripted lacks real-styled positives; real-agent
# runs on this easy corpus are ~98% positive (no negatives to learn a boundary).
# The fix is real-styled examples of BOTH classes. Real positives come free from
# a real-agent run; real-styled negatives are synthesized here by taking a real
# positive's transcript (its real claim + real tool narration) and re-rendering
# it against the task's UNSOLVED fixture — "the agent said done, but the
# workspace has no completing effect", the ADR-010 completion-without-effect
# case, in real style. No agent re-run, no GPU.


def parse_rendered_view(text: str) -> tuple[str, "Transcript"]:
    """Inverse of ``render_judged_content`` for the (prompt, transcript) parts.

    Recovers the prompt, the tool-step contents, and the final message from a
    rendered judge view. The workspace section is intentionally dropped — the
    caller re-renders against a different workspace. Only tool steps and the
    final message are recovered (they are all ``render_judged_content`` uses);
    truncation markers (``- ... N more``) and ``(none)`` are skipped.
    """
    from victor.evaluation.judge_calibration_harness import Transcript, TranscriptStep

    def section(start: str, end: str | None) -> str:
        s = text.index(start) + len(start)
        e = text.index(end, s) if end else len(text)
        return text[s:e].strip("\n")

    prompt = section("TASK:\n", "\n\nTOOL ACTIVITY:")
    tools_block = section("TOOL ACTIVITY:\n", "\n\nFINAL RESPONSE:")
    final = section("FINAL RESPONSE:\n", "\n\nWORKSPACE STATE:")

    steps = []
    for line in tools_block.splitlines():
        line = line.strip()
        if not line.startswith("- ") or line == "- (none)" or line.startswith("- ... "):
            continue
        steps.append(TranscriptStep(kind="tool", content=line[2:]))
    return prompt, Transcript(steps=tuple(steps), final_message=final)


def synthesize_effect_removed_negative(
    task: VerifiableTask, transcript: "Transcript", *, workspace_root: Optional[Path] = None
) -> TrainingExample:
    """Render ``transcript`` against ``task``'s UNSOLVED fixture → a label-0 example.

    The fixture is materialized by ``task.setup`` alone (no solve applied), so
    the workspace shows the original, incomplete state while the transcript
    still carries the agent's real completion claim. A sanity check asserts the
    task's own verifier scores this 0 (a real fixture on which the task is
    genuinely incomplete); tasks whose fixture already verifies complete are
    skipped by the caller.
    """
    import shutil
    import tempfile

    caller_owned = workspace_root is not None
    root = workspace_root or Path(tempfile.mkdtemp(prefix="judge_realneg_"))
    try:
        ws = root / task.task_id
        ws.mkdir(parents=True, exist_ok=True)
        task.setup(ws)
        text = render_judged_content(task.prompt, transcript, ws)
        gold = float(task.verify(ws, transcript))
        if gold >= 0.5:
            raise ValueError(f"task {task.task_id} verifies complete on its own fixture")
        return TrainingExample(
            task_id=task.task_id,
            family=task.family,
            source="real-neg",
            text=text,
            label=0,
        )
    finally:
        if not caller_owned:
            shutil.rmtree(root, ignore_errors=True)
