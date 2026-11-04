"""Render vector-family geometry consistently across source and summary channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console


def render_vector_cohesion(
    console: Console, cohesion: Mapping[str, Any] | None, members: Sequence[Mapping[str, Any]]
) -> None:
    if cohesion is None:
        return
    by_id = {member["id"]: member for member in members}
    representative = by_id[cohesion["representative_id"]]
    console.print(
        f"   representative: {representative['file']}:{representative['line']} "
        f"{representative.get('symbol', '')}; mean member-pair cosine "
        f"{cohesion['mean_pair_similarity']:.3f}",
        markup=False,
    )
    left, right = (by_id[value] for value in cohesion["witness_ids"])
    console.print(
        f"   pair witness: {left['file']}:{left['line']} ↔ "
        f"{right['file']}:{right['line']} = {cohesion['witness_similarity']:.3f} "
        "(not a verified pairwise minimum)",
        markup=False,
    )
