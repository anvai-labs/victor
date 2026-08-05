#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Train the arm-B encoder completion judge (E2 — the semantic bet).

ModernBERT-base + binary classification head on the PR-J1 dataset. Model
selection is by dev Krippendorff α (the gate metric), not accuracy. Needs the
dev-only [judge-training] deps (torch, transformers); inference for gating
loads the saved directory via transformers (CPU is fine).

Usage:
    python benchmarks/judge_training/train_encoder.py \
        --dataset ~/.victor/models/judge/dataset \
        --out ~/.victor/models/judge/modernbert_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / ".victor/models/judge/dataset"
    )
    parser.add_argument(
        "--out", type=Path, default=Path.home() / ".victor/models/judge/modernbert_v1"
    )
    parser.add_argument("--base", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from victor.evaluation.judge_calibration import evaluate_judge_agreement

    train_texts, train_labels = _load(args.dataset / "train.jsonl")
    dev_texts, dev_labels = _load(args.dataset / "dev.jsonl")
    print(f"train n={len(train_texts)}, dev n={len(dev_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)

    class DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    def metrics(pred):
        judged = pred.predictions.argmax(-1).astype(float).tolist()
        rel = evaluate_judge_agreement(
            [float(g) for g in pred.label_ids.tolist()], judged, level="nominal"
        )
        acc = float(np.mean(pred.predictions.argmax(-1) == pred.label_ids))
        return {"alpha": rel.krippendorff_alpha, "accuracy": acc}

    from transformers import DataCollatorWithPadding

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out) + "-ckpt",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch * 2,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="alpha",
            greater_is_better=True,
            logging_steps=50,
            report_to=[],
            fp16=torch.cuda.is_available(),
        ),
        train_dataset=DS(train_texts, train_labels),
        eval_dataset=DS(dev_texts, dev_labels),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    final = trainer.evaluate()
    print(f"dev: alpha={final.get('eval_alpha'):.4f} accuracy={final.get('eval_accuracy'):.4f}")
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
