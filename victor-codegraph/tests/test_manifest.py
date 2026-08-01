"""Incremental manifest contract + encoding fallback (offline, tmp_path only)."""

from __future__ import annotations

from victor_codegraph import (
    build_manifest,
    diff_manifest,
    iter_changed_files,
    parse_path,
    read_source_text,
)


def _seed(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def a():\n    return 1\n")
    (tmp_path / "pkg" / "b.py").write_text("def b():\n    return 2\n")
    (tmp_path / "README.md").write_text("# not source\n")  # unrecognized: excluded
    return tmp_path


def test_build_manifest_relative_posix_keys(tmp_path):
    manifest = build_manifest(_seed(tmp_path))
    assert set(manifest) == {"pkg/a.py", "pkg/b.py"}
    assert all("\\" not in k for k in manifest)
    assert all(len(v) == 64 for v in manifest.values())  # sha-256 hex


def test_manifest_hash_matches_parse_path(tmp_path):
    root = _seed(tmp_path)
    manifest = build_manifest(root)
    parsed = parse_path(root / "pkg" / "a.py")
    assert parsed is not None
    assert manifest["pkg/a.py"] == parsed.content_hash


def test_diff_manifest_classifies_all_states(tmp_path):
    root = _seed(tmp_path)
    previous = build_manifest(root)

    (root / "pkg" / "a.py").write_text("def a():\n    return 42\n")  # changed
    (root / "pkg" / "b.py").unlink()  # removed
    (root / "pkg" / "c.py").write_text("def c():\n    return 3\n")  # added

    current = build_manifest(root)
    diff = diff_manifest(current, previous)
    assert diff.added == ["pkg/c.py"]
    assert diff.changed == ["pkg/a.py"]
    assert diff.removed == ["pkg/b.py"]
    assert diff.unchanged == []


def test_diff_manifest_unchanged(tmp_path):
    root = _seed(tmp_path)
    manifest = build_manifest(root)
    diff = diff_manifest(manifest, dict(manifest))
    assert diff.added == diff.changed == diff.removed == []
    assert set(diff.unchanged) == {"pkg/a.py", "pkg/b.py"}


def test_iter_changed_files_none_previous_yields_all(tmp_path):
    root = _seed(tmp_path)
    changed = list(iter_changed_files(root, None))
    assert {p.name for p, _ in changed} == {"a.py", "b.py"}
    # Yielded hashes ARE the next manifest's values.
    assert {h for _, h in changed} == set(build_manifest(root).values())


def test_iter_changed_files_skips_unchanged(tmp_path):
    root = _seed(tmp_path)
    previous = build_manifest(root)
    assert list(iter_changed_files(root, previous)) == []

    (root / "pkg" / "a.py").write_text("def a():\n    return 99\n")
    changed = list(iter_changed_files(root, previous))
    assert [p.name for p, _ in changed] == ["a.py"]
    # The yielded hash reflects the NEW content.
    assert changed[0][1] == build_manifest(root)["pkg/a.py"]


def test_single_file_root(tmp_path):
    f = tmp_path / "solo.py"
    f.write_text("def solo():\n    return 0\n")
    manifest = build_manifest(f)
    assert set(manifest) == {"solo.py"}


# ── encoding fallback ────────────────────────────────────────────────────────


def test_latin1_file_no_longer_skipped(tmp_path):
    f = tmp_path / "legacy.py"
    f.write_bytes("# caf\xe9\ndef caf():\n    return 1\n".encode("latin-1"))
    text = read_source_text(f)
    assert text is not None
    assert "caf\xe9" in text

    parsed = parse_path(f)
    assert parsed is not None
    assert [s.simple_name for s in parsed.symbols] == ["caf"]

    # And it participates in manifests deterministically.
    manifest = build_manifest(tmp_path)
    assert manifest["legacy.py"] == parsed.content_hash


def test_utf8_bom_stripped(tmp_path):
    f = tmp_path / "bom.py"
    f.write_bytes(b"\xef\xbb\xbf" + b"def bom():\n    return 1\n")
    text = read_source_text(f)
    assert text is not None
    assert not text.startswith("﻿")
    parsed = parse_path(f)
    assert parsed is not None
    assert [s.simple_name for s in parsed.symbols] == ["bom"]


def test_fallback_disabled_restores_skip_behavior(tmp_path):
    f = tmp_path / "legacy.py"
    f.write_bytes("# caf\xe9\n".encode("latin-1"))
    assert read_source_text(f, encoding_fallback=False) is None
    assert parse_path(f, encoding_fallback=False) is None


def test_unreadable_file_returns_none(tmp_path):
    assert read_source_text(tmp_path / "missing.py") is None
    assert parse_path(tmp_path / "missing.py") is None
