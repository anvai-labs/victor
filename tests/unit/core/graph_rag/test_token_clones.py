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
    required = os.environ.get("VICTOR_REQUIRE_JSCPD_TESTS") == "1"
    if not value:
        if required:
            pytest.fail("pinned jscpd qualification is required but its executable is unavailable")
        pytest.skip("optional pinned jscpd backend is not installed")
    if (
        subprocess.run([value, "--version"], capture_output=True, text=True, check=True)
        .stdout.strip()
        .removeprefix("jscpd ")
        != token_clones.PINNED_JSCPD_VERSION
    ):
        if required:
            pytest.fail("pinned jscpd qualification requires the declared backend version")
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


def test_real_fingerprints_survive_moves_and_line_shifts_and_count_a_new_copy(tmp_path, binary):
    from collections import Counter

    source_pair(tmp_path)
    options = TokenOptions(binary=binary, min_tokens=8, min_lines=0, fingerprints=True)

    def fingerprints():
        result = scan_tokens(tmp_path, options=options)
        assert result["coverage_complete"], result
        assert result["fingerprints_complete"]
        return Counter(item["fingerprint"] for item in result["fingerprints"])

    first = fingerprints()
    source = (tmp_path / "a.py").read_text()
    (tmp_path / "a.py").rename(tmp_path / "moved.py")
    (tmp_path / "moved.py").write_text("\n\n" + source)
    assert fingerprints() == first
    (tmp_path / "third.py").write_text(source)
    third = fingerprints()
    assert set(third) == set(first)
    assert sum(third.values()) > sum(first.values())


def test_real_fingerprint_carrier_preserves_literal_special_filenames(tmp_path, binary):
    names = ["a%name.py", "b space.py", "c#hash.py", "d?query.py", "e_λ.py", "f%20.py"]
    source = "def f(x):\n value=x+1\n adjusted=value*3\n return adjusted+4\n"
    for name in names:
        (tmp_path / name).write_text(source)
    result = scan_tokens(
        tmp_path, options=TokenOptions(binary=binary, min_tokens=8, min_lines=0, fingerprints=True)
    )
    assert result["fingerprints_complete"], result
    assert {m["file"] for item in result["fingerprints"] for m in item["members"]} == set(names)


@pytest.mark.parametrize(
    "suffix,source",
    [
        ("rs", "/*😊*/ fn compute(x:i32)->i32 { let value=x+1; value*3+4 }\n"),
        (
            "ts",
            "/*😊*/ export function compute(x:number):number { const value=x+1; return value*3+4; }\n",
        ),
        ("cpp", "/*😊*/ int compute(int x) { int value=x+1; return value*3+4; }\n"),
        (
            "java",
            "/*😊*/ class Example { static int compute(int x) { int value=x+1; return value*3+4; } }\n",
        ),
    ],
)
def test_real_fingerprint_spans_correlate_across_languages_and_unicode(
    tmp_path, binary, suffix, source
):
    for name in (f"a.{suffix}", f"b.{suffix}"):
        (tmp_path / name).write_text(source)
    result = scan_tokens(
        tmp_path, options=TokenOptions(binary=binary, min_tokens=8, min_lines=0, fingerprints=True)
    )
    assert result["coverage_complete"], result
    assert result["fingerprints_complete"]
    assert result["fingerprints"]
    assert all(item["kind"] == "exact" for item in result["fingerprints"])


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX group ownership")
def test_normal_parent_exit_cleans_up_its_child(tmp_path):
    import sys
    import time
    import psutil

    script = "import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']); Path('child.pid').write_text(str(p.pid))"
    code, _ = token_clones._run_detector([sys.executable, "-c", script], tmp_path, 3)
    assert code == 0
    pid = int((tmp_path / "child.pid").read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.02)
    pytest.fail("owned descendant survived detector return")


def test_fast_output_still_obeys_artifact_limit(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(token_clones, "MAX_REPORT_BYTES", 10)
    script = "from pathlib import Path;Path('backend.log').write_bytes(b'x'*11)"
    with pytest.raises(ValueError, match="artifact budget"):
        token_clones._run_detector([sys.executable, "-c", script], tmp_path, 3)


def test_cancellation_after_spawn_reaps_owned_group(tmp_path, monkeypatch):
    import sys

    created = []
    spawn = subprocess.Popen

    def observe(*args, **kwargs):
        result = spawn(*args, **kwargs)
        created.append(result)
        return result

    def interrupt(delay):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", observe)
    from victor.core.graph_rag import bounded_process

    monkeypatch.setattr(bounded_process.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        token_clones._run_detector(
            [sys.executable, "-c", "import time;time.sleep(20)"], tmp_path, 3
        )
    assert len(created) == 1 and created[0].returncode is not None


def test_relative_backend_is_resolved_before_changing_cwd(tmp_path, monkeypatch, binary):
    source_pair(tmp_path)
    monkeypatch.chdir(tmp_path)
    link = tmp_path / "jscpd"
    link.symlink_to(binary)
    report = scan_tokens(
        tmp_path, options=TokenOptions(binary="./jscpd", min_tokens=8, min_lines=0)
    )
    assert report["coverage_complete"]


def test_token_snapshot_rejects_external_source_symlink(tmp_path, monkeypatch):
    import sys

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "private.py"
    outside.write_text("def secret(x):\n return x+1\n")
    (project / "linked.py").symlink_to(outside)
    monkeypatch.setattr(shutil, "which", lambda value: sys.executable)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "jscpd 5.4.2", "")
    )
    monkeypatch.setattr(
        "victor.core.graph_rag.structural_clones._source_files", lambda *a, **k: ["linked.py"]
    )
    report = scan_tokens(project)
    assert not report["coverage_complete"]
    assert not report["source_manifest"]
    assert report["errors"]


def test_denied_process_enumeration_fails_closed(tmp_path, monkeypatch):
    import sys
    import psutil

    def denied(*args, **kwargs):
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil.Process, "children", denied)
    monkeypatch.setattr(token_clones, "MAX_BACKEND_RSS", 32 * 1024 * 1024)
    with pytest.raises(ValueError, match="process-tree monitoring is unavailable"):
        token_clones._run_detector(
            [sys.executable, "-c", "import time; payload=bytearray(80*1024*1024);time.sleep(20)"],
            tmp_path,
            3,
        )


@pytest.mark.skipif(os.name != "posix", reason="descriptor traversal race")
def test_token_inventory_root_replacement_retains_original_source(tmp_path, monkeypatch, binary):
    from victor.core.graph_rag import structural_clones

    project = tmp_path / "project"
    project.mkdir()
    source_pair(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    for name in ("a.py", "b.py"):
        (outside / name).write_text("def secret(x):\n return x-1\n")

    def replace_root(*args, **kwargs):
        project.rename(tmp_path / "original")
        project.symlink_to(outside, target_is_directory=True)
        return ["a.py", "b.py"]

    monkeypatch.setattr(structural_clones, "_source_files", replace_root)
    with pytest.raises(OSError, match="root changed"):
        scan_tokens(project, options=TokenOptions(binary=binary, min_tokens=8, min_lines=0))
