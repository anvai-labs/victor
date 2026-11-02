"""Family source references remain literal and consistent in terminal output."""

from io import StringIO

from rich.console import Console

from victor.ui.rendering.clone_families import render_vector_cohesion


def test_renderer_preserves_source_markup_characters():
    output = StringIO()
    console = Console(file=output, width=200)
    members = [
        {"id": "a", "file": "[red]a.py", "symbol": "[bold]f", "line": 1},
        {"id": "b", "file": "b.py", "symbol": "f", "line": 2},
    ]
    render_vector_cohesion(
        console,
        {
            "representative_id": "a",
            "mean_pair_similarity": 0.9,
            "witness_ids": ("a", "b"),
            "witness_similarity": 0.9,
        },
        members,
    )
    text = output.getvalue()
    assert "[red]a.py:1 [bold]f" in text
    assert "mean member-pair cosine 0.900" in text
    assert "not a verified pairwise minimum" in text


def test_reports_without_vector_geometry_keep_their_existing_rendering():
    output = StringIO()
    render_vector_cohesion(Console(file=output), None, [])
    assert not output.getvalue()
