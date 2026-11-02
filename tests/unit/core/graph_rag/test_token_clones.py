"""Pinned external backend, source isolation and machine-readable failure contracts."""

import json
import os
import shutil
import subprocess

import pytest

from victor.core.graph_rag import token_clones
from victor.core.graph_rag.token_clones import TokenOptions, scan_tokens


@pytest.fixture
def binary():
    value = os.environ.get("JSCPD_BINARY") or shutil.which("jscpd")
    if not value:
        pytest.skip("optional pinned jscpd backend is not installed")
    if (
        subprocess.run([value, "--version"], capture_output=True, text=True, check=True)
        .stdout.strip()
        .removeprefix("jscpd ")
        != token_clones.PINNED_JSCPD_VERSION
    ):
        pytest.skip("the installed jscpd version is not the pinned evidence backend")
    return value


def source_pair(root):
    text = "def f(x):\n value=x+1\n adjusted=value*3\n return adjusted+4\n"
    for name in ("a.py", "b.py"):
        (root / name).write_text(text)


def test_real_fragment_backend_ignores_project_config(tmp_path, binary):
    source_pair(tmp_path)
    config = tmp_path / ".jscpd.json"
    config.write_text("invalid project config must never be read")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    report = scan_tokens(tmp_path, options=TokenOptions(binary=binary, min_tokens=8, min_lines=0))
    assert report["coverage_complete"], report
    assert report["backend"]["project_config_used"] is False
    assert report["fragments"]
    assert {m["file"] for m in report["fragments"][0]["members"]} == {"a.py", "b.py"}
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_partial_block_channel_finds_a_fragment_without_whole_body_match(tmp_path, binary):
    shared = "\n".join(f" value=value*{i}+{i+1}" for i in range(3, 15))
    (tmp_path / "a.py").write_text("def first(x):\n value=x\n" + shared + "\n return value\n")
    (tmp_path / "b.py").write_text(
        "def second(config):\n values=list(config)\n value=len(values)\n"
        + shared
        + "\n while values:\n  values.pop()\n return str(value)\n"
    )
    report = scan_tokens(tmp_path, options=TokenOptions(binary=binary, min_tokens=30, min_lines=1))
    assert report["coverage_complete"], report
    assert report["fragments"]
    assert all(f["relation"] == "literal_token_fragment" for f in report["fragments"])


def test_backend_is_not_downloaded_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda value: None)
    with pytest.raises(ValueError, match="Install jscpd"):
        scan_tokens(tmp_path)


def test_unpinned_version_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda value: "/fake/jscpd")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "9.0.0\n", "")
    )
    with pytest.raises(ValueError, match="expected 5.4.2"):
        scan_tokens(tmp_path)


@pytest.mark.parametrize("corrupt", ["outside", "span", "count", "kind"])
def test_invalid_backend_evidence_is_rejected(tmp_path, monkeypatch, corrupt):
    source_pair(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda value: "/fake/jscpd")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "5.4.2\n", "")
    )
    monkeypatch.setattr(
        "victor.core.graph_rag.structural_clones._source_files", lambda *a, **k: ["a.py", "b.py"]
    )

    def emit(arguments, work, timeout):
        (work / "report").mkdir()
        item = {"name": "a.py", "start": 1, "end": 4}
        duplicate = {
            "kind": "exact",
            "firstFile": item,
            "secondFile": {"name": "b.py", "start": 1, "end": 4},
            "fragment": "value=x+1",
            "tokens": 8,
        }
        if corrupt == "outside":
            item["name"] = "../secret.py"
        if corrupt == "span":
            item["end"] = 400
        if corrupt == "count":
            duplicate["tokens"] = 0
        if corrupt == "kind":
            duplicate["kind"] = "semantic"
        (work / "report" / "jscpd-report.json").write_text(
            json.dumps({"statistics": {"total": {"sources": 2}}, "duplicates": [duplicate]})
        )
        return 0, "test"

    monkeypatch.setattr(token_clones, "_run_detector", emit)
    with pytest.raises(ValueError):
        scan_tokens(tmp_path, options=TokenOptions(min_tokens=8))


def test_timeout_terminates_owned_backend(tmp_path):
    import sys

    with pytest.raises(ValueError, match="time budget"):
        token_clones._run_detector(
            [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 0.1
        )
