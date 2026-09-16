"""D2 rendering uses private temporary paths and cleans up failed renders."""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.workflows import visualization


@pytest.mark.parametrize("failed", [False, True])
def test_d2_paths_are_private_and_cleaned_up(monkeypatch, failed):
    visualizer = object.__new__(visualization.WorkflowVisualizer)
    monkeypatch.setattr(visualizer, "to_d2", lambda: "a -> b")
    monkeypatch.setattr(visualization, "_check_command", lambda command: True)
    paths = []

    def render(args, **kwargs):
        source, output = map(Path, args[-2:])
        paths.extend([source, output])
        assert source.parent == output.parent
        if os.name == "posix":
            assert stat.S_IMODE(source.parent.stat().st_mode) == 0o700
        assert source.read_text() == "a -> b"
        if not failed:
            output.write_text("<svg />")
        return SimpleNamespace(returncode=int(failed), stderr="render failure")

    monkeypatch.setattr(visualization.subprocess, "run", render)
    if failed:
        with pytest.raises(RuntimeError, match="render failure"):
            visualizer._to_svg_d2()
    else:
        assert visualizer._to_svg_d2() == "<svg />"
    assert paths
    assert all(not path.parent.exists() for path in paths)


def test_d2_preserves_requested_output(tmp_path, monkeypatch):
    visualizer = object.__new__(visualization.WorkflowVisualizer)
    monkeypatch.setattr(visualizer, "to_d2", lambda: "a -> b")
    monkeypatch.setattr(visualization, "_check_command", lambda command: True)
    output = tmp_path / "diagram.svg"

    def render(args, **kwargs):
        assert Path(args[-1]) == output
        output.write_text("<svg />")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(visualization.subprocess, "run", render)
    assert visualizer._to_svg_d2(str(output)) == "<svg />"
    assert output.read_text() == "<svg />"
