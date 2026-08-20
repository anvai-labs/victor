# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The docs-drift gate: docs must not contradict code-derived facts / the declared canon."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_docs_drift.py"
    spec = importlib.util.spec_from_file_location("check_docs_drift", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()
scan = mod.scan_text


def test_expected_version_matches_version_file():
    assert mod.expected_version() == (mod.ROOT / "VERSION").read_text().strip()


def test_expected_python_minimum_matches_package_metadata():
    assert mod.expected_python_minimum() == "3.11"


def test_expected_provider_count_is_positive():
    assert mod.expected_provider_count() >= 20  # 24 today; sanity floor


def test_aligned_canonical_doc_has_no_errors():
    text = "**Version**: 1.2.3 | **License**: Apache\nsupports 24 LLM providers, 34 tool modules.\n9 verticals (5 domain + 4 utility)\n"
    assert scan("docs/features.md", text, "1.2.3", 24) == []


def test_wrong_version_stamp_in_canonical_doc_flagged():
    errs = scan("docs/index.md", "**Version**: 0.7.0 | x", "0.7.1", 24)
    assert any("version stamp 0.7.0 != 0.7.1" in e for e in errs)


def test_wrong_python_minimum_flagged_anywhere():
    errs = scan("docs/anything.md", "Requires Python 3.10+", "0.7.1", 24, "3.11")
    assert any("Python minimum 3.10 != 3.11" in e for e in errs)


def test_python_test_matrix_is_not_a_minimum_claim():
    assert scan("docs/anything.md", "Test on Python 3.11, 3.12", "0.7.1", 24, "3.11") == []


def test_version_stamp_ignored_in_noncanonical_doc():
    # Spec/FEP docs carry their own independent version — must NOT be flagged.
    assert scan("docs/feps/vertical-package-spec.md", "**Version**: 1.0.0", "0.7.1", 24) == []


def test_wrong_provider_count_flagged_anywhere():
    errs = scan("docs/anything.md", "supports 21 different providers", "0.7.1", 24)
    assert any("provider count 21 != 24" in e for e in errs)


def test_wrong_tool_module_count_flagged():
    errs = scan("docs/x.md", "33 tool modules across categories", "0.7.1", 24)
    assert any("tool modules 33 != 34" in e for e in errs)


def test_canonical_vertical_claim_flagged_when_wrong():
    errs = scan("docs/x.md", "across 7 verticals (5 domain + 4 utility)", "0.7.1", 24)
    assert any("verticals 7 != 9" in e for e in errs)


def test_bare_vertical_counts_are_not_flagged():
    # Legitimate contextual counts must NOT trip the gate (only the compound canonical claim does).
    for text in [
        "5/5 verticals migrated",
        "13 vertical-specific tests",
        "TD-2 Vertical Cleanup",
        "5 external verticals",
    ]:
        assert scan("docs/migration.md", text, "0.7.1", 24) == [], text


def test_repo_docs_currently_pass():
    # The live tree must be aligned (guards against regressions in this very check).
    assert mod.main() == 0


def test_instruction_files_are_scanned():
    # F-001: root instruction files (CLAUDE.md, .victor/init.md, AGENTS.md) carry
    # canon counts and must be in the scan set so a future drift there is caught.
    #
    # These files are intentionally gitignored (machine-local instruction
    # context), so they are present in a developer checkout but ABSENT in CI's
    # fresh checkout. The scanner's contract is "scan when present"; the test
    # therefore mirrors that conditional contract: it asserts the file appears
    # in the scan set *iff* it exists on disk. This keeps the drift guard
    # active locally without producing an environment-fragile CI failure.
    # Membership in the declared scan set is unconditional, so this test is
    # never fully vacuous even where the files themselves are absent.
    assert {"CLAUDE.md", "AGENTS.md", ".victor/init.md"} <= set(mod.EXTRA_FILES)
    scanned = {str(p.relative_to(mod.ROOT)) for p in mod._doc_files()}
    assert "CLAUDE.md" in scanned or not (mod.ROOT / "CLAUDE.md").exists()
    assert ".victor/init.md" in scanned or not (mod.ROOT / ".victor/init.md").exists()
    assert "AGENTS.md" in scanned or not (mod.ROOT / "AGENTS.md").exists()


def test_instruction_files_scanned_when_present(tmp_path, monkeypatch):
    # Positive direction (runs everywhere, incl. CI where the real files are
    # absent): with instruction files present under ROOT, they enter the scan.
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "CLAUDE.md").write_text("24 LLM providers\n")
    (tmp_path / ".victor").mkdir(exist_ok=True)  # autouse fixtures may pre-create it
    (tmp_path / ".victor" / "init.md").write_text("34 tool modules\n")
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    scanned = {str(p.relative_to(tmp_path)) for p in mod._doc_files()}

    assert "CLAUDE.md" in scanned
    assert ".victor/init.md" in scanned
