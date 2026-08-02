# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""``victor ml`` — the edge-classifier ML loop, drivable from the CLI (FEP-0012).

Wraps the already-merged closed-loop functions into a command group:

- ``victor ml train``   — produce a classifier artifact (from the
  ``decision_outcome`` junction by default, or ``--from-manifest`` for the
  offline path). Writes to ``victor/models/edge_classifier_v1.npz`` by default
  so ``auto`` adopts it on next session.
- ``victor ml validate`` — run the parity gate (train/test split, holdout
  accuracy vs majority baseline). Exit 0 = ship, 2 = do-not-ship, 1 = no data.
- ``victor ml mine``     — project a benchmark manifest into training rows.

Training/validation need the dev-only ``[ml]`` extra (scikit-learn/scipy); a
missing extra is reported clearly rather than crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

ml_app = typer.Typer(name="ml", help="Edge-classifier ML: train, validate, mine (FEP-0012).")
console = Console()


def _default_artifact_path() -> Path:
    """Where ``auto`` looks for the shipped artifact (LocalClassifierDecisionService)."""
    import victor

    return Path(victor.__file__).resolve().parent / "models" / "edge_classifier_v1.npz"


def _require_sklearn() -> bool:
    """Return True if the [ml] extra (scikit-learn) is importable, else message + exit."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        console.print(
            "[red]The [ml] extra is required for training/validation.[/]\n"
            '[dim]Install it with: pip install -e ".[ml]"[/]'
        )
        raise typer.Exit(1)
    return True


@ml_app.command("train")
def ml_train(
    from_manifest: Optional[Path] = typer.Option(
        None,
        "--from-manifest",
        help="Train from a benchmark eval_manifest_*.jsonl (offline path). "
        "Default: train from the decision_outcome junction.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Artifact path (.npz). Default: victor/models/edge_classifier_v1.npz "
        "(where `auto` adopts it).",
    ),
    threshold: float = typer.Option(0.6, help="Confidence gate τ stored on each head."),
) -> None:
    """Train a classifier artifact and write it to disk."""
    _require_sklearn()
    out = output or _default_artifact_path()

    if from_manifest:
        from victor.ml.mining import train_from_manifest

        model = train_from_manifest(from_manifest, out, threshold=threshold)
    else:
        from victor.ml.outcome_training import train_from_outcomes

        model = train_from_outcomes(out, threshold=threshold)

    if model is None:
        console.print(
            "[yellow]No classifier trained.[/] Need ≥1 decision type with ≥2 distinct "
            "outcome labels — run a benchmark that produces both passing AND failing "
            "tasks, then re-run."
        )
        raise typer.Exit(2)

    heads = list(model.heads)
    console.print(f"[green]Trained artifact:[/] {out}")
    console.print(f"[dim]Heads ({len(heads)}): {', '.join(heads)}[/]")
    console.print(
        f"[dim]Drop it at {_default_artifact_path()} (or set "
        "VICTOR_EDGE_CLASSIFIER_PATH) and `auto` adopts it next session.[/]"
    )


@ml_app.command("validate")
def ml_validate(
    holdout_frac: float = typer.Option(0.2, help="Fraction held out for evaluation."),
    min_samples: int = typer.Option(20, help="Min holdout samples for a type to ship."),
    min_coverage: float = typer.Option(0.5, help="Min prediction coverage for a type to ship."),
    min_margin: float = typer.Option(0.0, help="Min calibrated-accuracy margin over baseline."),
) -> None:
    """Run the parity gate: ship a classifier only if it beats a naive baseline."""
    _require_sklearn()
    from victor.ml.parity_gate import validate_outcome_training

    verdict = validate_outcome_training(
        holdout_frac=holdout_frac,
        min_samples=min_samples,
        min_coverage=min_coverage,
        min_margin=min_margin,
    )

    if "reason" in verdict:
        console.print(f"[yellow]No verdict:[/] {verdict['reason']}")
        raise typer.Exit(1)

    table = Table(title="Parity gate (held-out decisions)")
    table.add_column("type", style="cyan")
    table.add_column("ship", justify="center")
    table.add_column("n", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("acc", justify="right")
    table.add_column("baseline", justify="right")
    table.add_column("margin", justify="right")
    for dtype, m in verdict["per_type"].items():
        table.add_row(
            dtype,
            "[green]YES[/]" if m["ship"] else "[red]no[/]",
            str(m["n"]),
            f"{m['coverage']:.2f}",
            f"{m['calibrated_accuracy']:.2f}",
            f"{m['baseline_accuracy']:.2f}",
            f"{m['margin']:+.2f}",
        )
    console.print(table)

    if verdict["ship"]:
        console.print(f"\n[bold green]SHIP[/] — key type {verdict['key_type']} cleared the bar.")
        raise typer.Exit(0)
    console.print(
        f"\n[bold red]DO NOT SHIP[/] — key type {verdict['key_type']} did not clear the bar."
    )
    raise typer.Exit(2)


@ml_app.command("mine")
def ml_mine(
    manifest: Path = typer.Argument(..., help="Path to eval_manifest_<run>.jsonl"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write detailed training rows JSONL here."
    ),
    train: Optional[Path] = typer.Option(
        None,
        "--train",
        help="Also train+save a classifier artifact (.npz) from the manifest.",
    ),
) -> None:
    """Project a benchmark manifest into reward-labeled training rows."""
    from collections import Counter

    from victor.ml.mining import mine_detailed, write_training_rows

    rows = mine_detailed(manifest)
    if not rows:
        console.print(f"[yellow]No minable decisions in {manifest}[/]")
        raise typer.Exit(1)

    out = output or manifest.with_suffix(".training_rows.jsonl")
    write_training_rows(rows, out)
    by_type = Counter(r["decision_type"] for r in rows)
    by_label = Counter(r["label"] for r in rows)
    console.print(
        f"[green]Mined {len(rows)} rows[/] " f"by_type={dict(by_type)} by_label={dict(by_label)}"
    )
    console.print(f"[dim]Wrote: {out}[/]")

    if train:
        _require_sklearn()
        from victor.ml.mining import train_from_manifest

        model = train_from_manifest(manifest, train)
        if model is None:
            console.print("[yellow]No classifier trained (need ≥2 distinct labels).[/]")
            raise typer.Exit(2)
        console.print(f"[green]Trained artifact:[/] {train}")
