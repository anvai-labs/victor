# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Project an execution manifest into reward-labeled classifier training rows.

The closed loop: ``eval_manifest_<run>.jsonl`` (emitted by
:mod:`victor.evaluation.manifest`) joins each task's outcome (reward) to the
decisions the agent logged during it. This module is the **pure projection**
that turns those joined records into the ``(text, label)`` samples
:func:`victor.ml.trainer.train_model` consumes.

Cost model
----------
- **Zero annotation cost** — the miner only rearranges already-captured data.
- **Zero judge cost** — labels come from ground-truth test outcomes, not an
  LLM judge.
- **Zero feature duplication** — features reuse
  :func:`victor.ml.features.extract_features` at train time.

Label semantics
---------------
The label is the **task reward** projected through the decision: for every
decision logged on a task, its label is that task's reward
(``pass``/``partial``/``fail``). For ``task_completion`` decisions this is the
premature-completion signal — a "complete" decision on a failed task is a
negative example. Other decision types get the same outcome-based reward
(noisier, refined later per DecisionType).

A head only trains when a run has ≥2 distinct labels among that decision type's
samples, so an all-fail run (no diversity) trains nothing — the loop tightens
as the pass rate improves across runs.

CLI
---
``python -m victor.ml.mining <manifest.jsonl> [-o training_rows.jsonl] [--train model.npz]``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from victor.ml.features import extract_features

logger = logging.getLogger(__name__)

# Decision types we surface training rows for. task_completion is the primary
# premature-completion head; the others get outcome-based reward labels as a
# first-pass proxy (refined per-type later).
MINED_DECISION_TYPES = frozenset({"task_completion", "stage_detection", "prompt_focus"})


def _decision_text(decision: dict[str, Any]) -> str:
    """Serialize a decision's input context to the feature-extraction text.

    ``log_decision`` stores the input as an arbitrary dict; we feed its JSON
    serialization to :func:`extract_features` so char/word n-grams capture
    message tails, error strings, response snippets, etc.
    """
    inp = decision.get("input")
    if isinstance(inp, str):
        return inp
    if inp is None:
        inp = decision.get("output", "")
    try:
        return json.dumps(inp, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(inp)


def mine_detailed(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest and return one detailed training row per decision.

    Each row carries the raw text (for ``train_model``), the derived label
    (task reward), the decision type, and the join keys (task_id/session_id)
    so rows can be audited back to their source task.
    """
    manifest_path = Path(manifest_path)
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            reward = rec.get("reward") or "fail"
            task_id = rec.get("task_id", "")
            session_id = rec.get("session_id", "")
            for decision in rec.get("decisions", []) or []:
                dtype = decision.get("type") or decision.get("decision_type")
                if dtype not in MINED_DECISION_TYPES:
                    continue
                rows.append(
                    {
                        "text": _decision_text(decision),
                        "label": reward,
                        "decision_type": dtype,
                        "task_id": task_id,
                        "session_id": session_id,
                    }
                )
    return rows


def mine(manifest_path: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Project a manifest into ``{decision_type: [(text, label), ...]}``.

    Ready to feed directly into :func:`victor.ml.trainer.train_model`.
    """
    per_type: dict[str, list[tuple[str, str]]] = {}
    for row in mine_detailed(manifest_path):
        per_type.setdefault(row["decision_type"], []).append((row["text"], row["label"]))
    return per_type


def write_training_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write detailed training rows to JSONL (for inspection / offline training)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            # Stash the feature vector alongside the text so an offline trainer
            # can consume rows without re-featurizing.
            out = dict(row)
            out["features"] = extract_features(row["text"])
            fh.write(json.dumps(out, default=str) + "\n")
    logger.info("Wrote %d training rows to %s", len(rows), path)
    return path


def train_from_manifest(
    manifest_path: str | Path,
    output_model: Optional[str | Path] = None,
    *,
    model_version: str = "manifest-1",
    threshold: float = 0.6,
) -> Optional[Any]:
    """Mine a manifest and train+save a classifier artifact (dev-only [ml] extra).

    Returns the trained :class:`~victor.ml.model.EdgeClassifierModel`, or
    ``None`` if no head could train (insufficient label diversity). Heads that
    lack ≥2 distinct labels are skipped with a log line.
    """
    per_type = mine(manifest_path)
    if not per_type:
        logger.warning("No minable decisions in %s", manifest_path)
        return None

    try:
        from victor.ml.trainer import train_head
        from victor.ml.model import EdgeClassifierModel
    except ImportError as exc:
        logger.error("Training needs the [ml] extra (scikit-learn/scipy): %s", exc)
        return None

    heads = {}
    for dtype, samples in per_type.items():
        distinct = {label for _, label in samples}
        if len(distinct) < 2:
            logger.info(
                "Skipping head %s: only %d distinct label(s) %s across %d samples",
                dtype,
                len(distinct),
                sorted(distinct),
                len(samples),
            )
            continue
        try:
            heads[dtype] = train_head(dtype, samples, threshold=threshold)
            logger.info(
                "Trained head %s on %d samples (labels=%s)",
                dtype,
                len(samples),
                sorted(distinct),
            )
        except ValueError as exc:
            logger.info("Skipping head %s: %s", dtype, exc)

    if not heads:
        logger.warning(
            "No heads trained from %s (need ≥1 decision type with ≥2 distinct "
            "outcome labels — run more tasks until some pass)",
            manifest_path,
        )
        return None

    model = EdgeClassifierModel(heads=heads, model_version=model_version)
    if output_model is not None:
        output_model = Path(output_model)
        output_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output_model))
        logger.info("Saved classifier artifact to %s", output_model)
    return model


def _main(argv: Optional[list[str]] = None) -> int:
    """``python -m victor.ml.mining <manifest> [-o rows.jsonl] [--train model.npz]``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", help="Path to eval_manifest_<run>.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write detailed training rows JSONL here (default: alongside manifest)",
    )
    parser.add_argument(
        "--train",
        default=None,
        help="Train a classifier artifact (.npz) from the manifest (needs [ml] extra)",
    )
    args = parser.parse_args(argv)

    rows = mine_detailed(args.manifest)
    if not rows:
        print(f"No minable decisions in {args.manifest}", file=sys.stderr)
        return 1

    out_path = args.output or str(Path(args.manifest).with_suffix(".training_rows.jsonl"))
    write_training_rows(rows, out_path)

    from collections import Counter

    by_type = Counter(r["decision_type"] for r in rows)
    by_label = Counter(r["label"] for r in rows)
    print(f"Rows: {len(rows)}  by_type={dict(by_type)}  by_label={dict(by_label)}")
    print(f"Wrote: {out_path}")

    if args.train:
        model = train_from_manifest(args.manifest, args.train)
        if model is None:
            print("No classifier trained (see log).", file=sys.stderr)
            return 2
        print(f"Trained artifact: {args.train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
