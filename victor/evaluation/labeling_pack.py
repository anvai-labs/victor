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

"""Human-labeling pack export for judge calibration (EVR-2, ADR-011).

Serializes exactly the judge's blinded view of each calibration task — prompt,
transcript, and a read-only workspace snapshot — so a human annotator labels
from the same evidence the judge saw. Verifier verdicts and judge scores are
structurally absent from the pack (the blinding contract of
:mod:`victor.evaluation.judge_calibration_harness` extends to human raters).

The export hook runs in harness phase 1, immediately after the executor and
before any judging or verification, so the snapshot can never contain verifier
side effects.

Protocol and pre-registered thresholds:
``docs/architecture/evr2-human-validation-protocol.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from victor.evaluation.judge_calibration_harness import Transcript, VerifiableTask

# Workspace files larger than this are listed by path+size, not inlined.
MAX_INLINE_FILE_BYTES = 65_536

# Directories that are execution noise, never task content.
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}


def snapshot_workspace(workspace: Path) -> dict[str, Any]:
    """A JSON-serializable, read-only view of the workspace file tree.

    Text files up to :data:`MAX_INLINE_FILE_BYTES` are inlined; larger or
    non-UTF-8 files are recorded as ``{"omitted": reason, "size": n}`` so the
    annotator knows they exist without the pack ballooning.
    """
    files: dict[str, Any] = {}
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(workspace).as_posix()
        size = path.stat().st_size
        if size > MAX_INLINE_FILE_BYTES:
            files[rel] = {"omitted": "too large to inline", "size": size}
            continue
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            files[rel] = {"omitted": "binary", "size": size}
    return {"files": files}


def export_labeling_record(
    pack_dir: Path,
    index: int,
    task: "VerifiableTask",
    workspace: Path,
    transcript: "Transcript",
) -> Path:
    """Write one blinded task record; returns the file path.

    The record deliberately contains no ``gold``, no verifier output, and no
    judge scores — a test pins this (blinding is a contract, not a convention).
    """
    record = {
        "task_id": task.task_id,
        "family": task.family,
        "prompt": task.prompt,
        "transcript": {
            "steps": [{"kind": s.kind, "content": s.content} for s in transcript.steps],
            "final_message": transcript.final_message,
        },
        "workspace": snapshot_workspace(workspace),
    }
    pack_dir.mkdir(parents=True, exist_ok=True)
    out = pack_dir / f"{index:04d}_{task.task_id}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return out


_PACK_README = """\
# Judge-calibration labeling pack (EVR-2, ADR-011)

Each `NNNN_<task_id>.json` is the **blinded view** of one calibration task:
the instruction (`prompt`), what the executor did (`transcript`), and the
resulting workspace state (`workspace.files`). It contains no verifier verdict
and no judge score — label from this evidence alone.

## How to label

For each record, answer ONE question: **did the execution genuinely complete
the task?** `1` = complete, `0` = not complete. Judge the workspace state and
transcript against the prompt; a confident claim without a matching effect is
`0` (the completion-without-effect failure mode this program exists to catch).

Fill in `labels.template.jsonl` (one line per task) and save it as
`labels.<annotator>.jsonl`:

    {"task_id": "...", "label": 1, "annotator": "your-name", "rationale": "optional"}

Rules (see docs/architecture/evr2-human-validation-protocol.md):
- Label every record; use `rationale` whenever the call felt non-obvious.
- Do not consult verifier code, FINDINGS.md run history, or another
  annotator's labels while labeling.
- Uncertain after honest effort → label the more skeptical option (`0`) and
  say why in `rationale`.
"""


def write_pack_manifest(pack_dir: Path, task_ids: list[str]) -> None:
    """Write the pack README and an empty labels template."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "README.md").write_text(_PACK_README)
    template_lines = [
        json.dumps({"task_id": task_id, "label": None, "annotator": "", "rationale": ""})
        for task_id in task_ids
    ]
    (pack_dir / "labels.template.jsonl").write_text("\n".join(template_lines) + "\n")


def make_labeling_pack_sink(
    pack_dir: Path,
) -> Callable[[int, "VerifiableTask", Path, "Transcript"], None]:
    """A harness ``record_sink`` that exports every task and tracks the manifest."""
    task_ids: list[str] = []

    def sink(index: int, task: "VerifiableTask", workspace: Path, transcript: "Transcript") -> None:
        export_labeling_record(pack_dir, index, task, workspace, transcript)
        task_ids.append(task.task_id)
        # Rewrite the manifest each record: the pack is complete/labelable even
        # if the run dies partway.
        write_pack_manifest(pack_dir, task_ids)

    return sink
