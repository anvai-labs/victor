"""Copy gates compare complete compatible counts and attribute changed groups."""

import copy
import asyncio
import json
import os
import threading
from dataclasses import replace

import pytest

from victor.core.graph_rag.copy_baseline import (
    CopyBaseline,
    compare_copies,
    copy_profile,
    make_baseline,
    parse_copy_report,
)
from victor.core.graph_rag.token_clones import TokenOptions
from victor.core.graph_rag.token_fingerprints import source_evidence_id


def report(profile, pairs):
    manifest = {name: {"sha256": "a" * 64, "lines": 10} for pair in pairs for name in pair}
    raw = {
        "evidence": "token_sequence",
        "coverage_complete": True,
        "fingerprints_complete": True,
        "fingerprint_schema": "jscpdCloneHash/v1",
        "backend": {**profile.to_dict()["backend"], "project_config_used": False},
        "errors": [],
        "stats": {
            "source_files": len(manifest),
            "backend_analyzed_files": len(manifest),
            "fragment_pairs": len(pairs),
        },
        "source_manifest": manifest,
        "source_selection": profile.to_dict()["selection"],
        "fingerprints": [
            {
                "fingerprint_schema": "jscpdCloneHash/v1",
                "fingerprint": "1" * 16,
                "format": "rust",
                "kind": "exact",
                "tokens": 30,
                "members": [
                    {
                        "file": name,
                        "line": 2,
                        "end_line": 4,
                        "source_hash": "a" * 64,
                        "backend_start_column": 0,
                        "backend_end_column": 8,
                    }
                    for name in pair
                ],
            }
            for pair in pairs
        ],
    }
    for row in raw["fingerprints"]:
        row["evidence_id"] = source_evidence_id(row)
    return raw


@pytest.fixture
def profile():
    return copy_profile(TokenOptions(min_tokens=20, min_lines=0, fingerprints=True))


def test_resource_budgets_do_not_change_copy_identity_but_exclusions_do():
    lower = copy_profile(
        TokenOptions(fingerprints=True, max_rss_bytes=192 * 1024**2, max_file_bytes=2 * 1024**2)
    )
    higher = copy_profile(
        TokenOptions(
            fingerprints=True,
            max_rss_bytes=2 * 1024**3,
            max_file_bytes=8 * 1024**2,
            timeout_seconds=120,
        )
    )
    excluded = copy_profile(TokenOptions(fingerprints=True, exclude_patterns=("aliases/**",)))
    assert lower == higher
    assert lower != excluded
    assert excluded.to_dict()["selection"]["exclude_patterns"] == ["aliases/**"]


async def test_exact_v1_baseline_is_readable_and_can_be_explicitly_replaced(tmp_path, profile):
    import hashlib
    from pathlib import Path
    from collections import Counter
    from victor.core.graph_rag.copy_artifacts import load_copy_baseline, write_copy_baseline
    from victor.core.graph_rag.copy_baseline import canonical

    artifact = json.loads(
        (
            Path(__file__).resolve().parents[4]
            / "docs/architecture/evidence/copy-workflow-2026-11-02.json"
        ).read_text()
    )
    application = artifact["candidate_application"]
    old_profile = application["profile"]
    counts = Counter(
        (row["fingerprint"], row["kind"], row["format"])
        for row in application["result"]["fingerprints"]
    )
    legacy = {
        "schema": "victor-copy-baseline-v1",
        "profile": old_profile,
        "profile_digest": hashlib.sha256(canonical(old_profile).encode()).hexdigest(),
        "source_digest": application["application"]["sarif"]["runs"][0]["properties"][
            "source_digest"
        ],
        "source_count": 166,
        "counts": [
            {"fingerprint": key[0], "kind": key[1], "format": key[2], "count": count}
            for key, count in sorted(counts.items())
        ],
    }
    data = (json.dumps(legacy, indent=2) + "\n").encode()
    assert hashlib.sha256(data).hexdigest() == application["application"]["baseline_sha256"]
    destination = tmp_path / "copies.json"
    destination.write_bytes(data)
    old = await load_copy_baseline(destination)
    assert old.profile.profile_version == 1
    current = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    with pytest.raises(ValueError, match="compatible"):
        compare_copies(current, old)
    assert destination.read_bytes() == data
    await write_copy_baseline(destination, current, source_root=tmp_path)
    assert (await load_copy_baseline(destination)).profile.profile_version == 2


def test_count_delta_is_attributed_to_the_whole_changed_fingerprint_group(profile):
    baseline = make_baseline(parse_copy_report(report(profile, [("a.rs", "c.rs")]), profile))
    # A newly inserted b.rs changes the backend's pair ordering. Counting only a
    # nominal "new edge" a/c would incorrectly miss changed b.rs.
    current = parse_copy_report(report(profile, [("a.rs", "b.rs"), ("a.rs", "c.rs")]), profile)
    result = compare_copies(current, baseline, changed_files={"b.rs"})
    assert result["new_copy_pairs"] == 1
    assert result["changed_new_copy_pairs"] == 1
    assert result["regressions"][0]["touches_changed_files"]
    assert (
        compare_copies(current, baseline, changed_files={"unrelated.rs"})["changed_new_copy_pairs"]
        == 0
    )


def test_moves_and_line_shifts_do_not_create_new_copy_counts(profile):
    old = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    raw = report(profile, [("moved.rs", "b.rs")])
    raw["fingerprints"][0]["members"][0].update(line=5, end_line=7)
    raw["fingerprints"][0]["evidence_id"] = source_evidence_id(raw["fingerprints"][0])
    current = parse_copy_report(raw, profile)
    baseline = CopyBaseline.from_dict(make_baseline(old).to_dict())
    assert compare_copies(current, baseline)["new_copy_pairs"] == 0
    assert baseline.source_digest != current.source_digest


@pytest.mark.parametrize(
    "corrupt",
    [
        "coverage",
        "carrier",
        "errors",
        "backend",
        "source_hash",
        "source_count",
        "fingerprint",
        "span",
        "path",
        "pair_count",
    ],
)
def test_incomplete_or_mismatched_copy_evidence_cannot_update_a_baseline(profile, corrupt):
    raw = report(profile, [("a.rs", "b.rs")])
    row = raw["fingerprints"][0]
    if corrupt == "coverage":
        raw["coverage_complete"] = False
    elif corrupt == "carrier":
        raw["fingerprints_complete"] = False
    elif corrupt == "errors":
        raw["errors"] = [{"file": "lost.rs", "error": "unreadable"}]
    elif corrupt == "backend":
        raw["backend"]["min_tokens"] += 1
    elif corrupt == "source_hash":
        row["members"][0]["source_hash"] = "b" * 64
    elif corrupt == "source_count":
        raw["stats"]["backend_analyzed_files"] = 1
    elif corrupt == "fingerprint":
        row["fingerprint"] = "invalid"
    elif corrupt == "span":
        row["members"][0]["end_line"] = 99
    elif corrupt == "path":
        row["members"][0]["file"] = "../a.rs"
    else:
        raw["stats"]["fragment_pairs"] += 1
    with pytest.raises(ValueError):
        parse_copy_report(raw, profile)


def test_profiles_and_baselines_reject_incompatible_or_ambiguous_counts(profile):
    evidence = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    baseline = make_baseline(evidence)
    with pytest.raises(ValueError, match="compatible"):
        compare_copies(replace(evidence, profile=replace(profile, include_tests=True)), baseline)
    for value in (True, -1, 0, 100001):
        raw = copy.deepcopy(baseline.to_dict())
        raw["counts"][0]["count"] = value
        with pytest.raises(ValueError):
            CopyBaseline.from_dict(raw)
    raw = baseline.to_dict()
    raw["counts"].append(copy.deepcopy(raw["counts"][0]))
    with pytest.raises(ValueError):
        CopyBaseline.from_dict(raw)


def test_duplicate_physical_occurrences_cannot_inflate_baseline_counts(profile):
    with pytest.raises(ValueError, match="Duplicate physical"):
        parse_copy_report(report(profile, [("a.rs", "b.rs"), ("a.rs", "b.rs")]), profile)


async def test_baseline_artifact_roundtrip_and_foreign_file_protection(tmp_path, profile):
    from victor.core.graph_rag.copy_artifacts import load_copy_baseline, write_copy_baseline

    evidence = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    destination = tmp_path / ".victor" / "copies.json"
    await write_copy_baseline(destination, evidence, source_root=tmp_path)
    assert await load_copy_baseline(destination) == make_baseline(evidence)
    await write_copy_baseline(destination, evidence, source_root=tmp_path)
    destination.write_text('{"user":"data"}')
    with pytest.raises(ValueError):
        await write_copy_baseline(destination, evidence, source_root=tmp_path)
    assert destination.read_text() == '{"user":"data"}'


@pytest.mark.parametrize("alias", ["source", "database", "symlink", "hardlink", "parent_symlink"])
async def test_baseline_output_cannot_clobber_inputs_or_follow_links(tmp_path, profile, alias):
    from victor.core.graph_rag.copy_artifacts import write_copy_baseline

    evidence = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    original = tmp_path / "owned.json"
    original.write_text(json.dumps(make_baseline(evidence).to_dict()))
    if alias == "source":
        destination = tmp_path / "a.rs"
    elif alias == "database":
        destination = tmp_path / ".victor" / "project.db"
    elif alias == "symlink":
        destination = tmp_path / "link.json"
        destination.symlink_to(original)
    elif alias == "hardlink":
        destination = tmp_path / "link.json"
        os.link(original, destination)
    else:
        (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
        destination = tmp_path / "linked" / "owned.json"
    before = original.read_bytes()
    with pytest.raises((OSError, ValueError)):
        await write_copy_baseline(destination, evidence, source_root=tmp_path)
    assert original.read_bytes() == before


def test_sarif_encodes_raw_names_once_and_preserves_actual_pair_evidence(tmp_path, profile):
    from victor.core.graph_rag.copy_artifacts import copy_sarif

    evidence = parse_copy_report(report(profile, [("a%20 #?.rs", "é.rs")]), profile)
    sarif = copy_sarif(tmp_path, evidence)
    run = sarif["runs"][0]
    assert run["invocations"][0]["executionSuccessful"]
    assert run["artifacts"][0]["location"]["uri"] == "a%2520%20%23%3F.rs"
    result = run["results"][0]
    assert result["properties"]["pairs"][0]["evidence_id"] == evidence.occurrences[0].evidence_id
    assert not result["properties"]["equivalence_verified"]
    assert "baselineState" not in result
    assert (
        copy_sarif(tmp_path, None, error="Incomplete coverage")["runs"][0]["invocations"][0][
            "executionSuccessful"
        ]
        is False
    )


async def test_baseline_protects_source_root_parent_aliases(tmp_path, profile):
    from victor.core.graph_rag.copy_artifacts import write_copy_baseline

    project = tmp_path / "project"
    project.mkdir()
    evidence = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    source = project / "a.rs"
    source.write_text(json.dumps(make_baseline(evidence).to_dict()))
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="source inputs"):
        await write_copy_baseline(source, evidence, source_root=alias / "project")
    assert source.read_bytes() == before


@pytest.mark.parametrize("change", ["destination", "directory"])
async def test_baseline_publication_rejects_replaced_destinations(
    tmp_path, profile, monkeypatch, change
):
    from victor.core.graph_rag.copy_artifacts import write_copy_baseline

    evidence = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    parent = tmp_path / "reports"
    parent.mkdir()
    destination = parent / "copies.json"
    destination.write_text(json.dumps(make_baseline(evidence).to_dict()))
    fsync = os.fsync

    def replaced(descriptor):
        if change == "destination":
            destination.write_text("new unrelated data")
        else:
            parent.rename(tmp_path / "saved")
            parent.mkdir()
        fsync(descriptor)

    monkeypatch.setattr(os, "fsync", replaced)
    with pytest.raises(ValueError, match="changed"):
        await write_copy_baseline(destination, evidence, source_root=tmp_path)
    assert not list(tmp_path.rglob("*.tmp"))
    if change == "destination":
        assert destination.read_text() == "new unrelated data"
    else:
        assert not destination.exists()


async def test_cancelled_baseline_publication_settles_once_before_returning(
    tmp_path, profile, monkeypatch
):
    from victor.core.graph_rag.copy_artifacts import load_copy_baseline, write_copy_baseline

    evidence = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    destination = tmp_path / "copies.json"
    entered, release = threading.Event(), threading.Event()
    replace_file = os.replace
    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(3)
        return replace_file(*args, **kwargs)

    monkeypatch.setattr(os, "replace", blocked)
    task = asyncio.create_task(write_copy_baseline(destination, evidence, source_root=tmp_path))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == [1]
    assert await load_copy_baseline(destination) == make_baseline(evidence)


async def test_baseline_loading_rejects_duplicate_json_and_size_overruns(
    tmp_path, profile, monkeypatch
):
    from victor.core.graph_rag import copy_artifacts

    evidence = parse_copy_report(report(profile, [("a.rs", "b.rs")]), profile)
    destination = tmp_path / "copies.json"
    value = json.dumps(make_baseline(evidence).to_dict())
    destination.write_text(value.replace('"schema":', '"schema":"duplicate", "schema":', 1))
    with pytest.raises(ValueError, match="Duplicate"):
        await copy_artifacts.load_copy_baseline(destination)
    destination.write_text(value)
    monkeypatch.setattr(copy_artifacts, "MAX_BASELINE_BYTES", 64)
    with pytest.raises(ValueError, match="byte budget"):
        await copy_artifacts.load_copy_baseline(destination)
    with pytest.raises(ValueError, match="byte budget"):
        await copy_artifacts.write_copy_baseline(
            tmp_path / "new.json", evidence, source_root=tmp_path
        )
    assert not (tmp_path / "new.json").exists()


async def test_workflow_refuses_incomplete_or_changed_producer_evidence_before_writing(
    tmp_path, profile, monkeypatch
):
    from victor.core.graph_rag.copy_artifacts import complete_copy_workflow
    from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES

    raw = report(profile, [("a.rs", "b.rs")])
    destination = tmp_path / "copies.json"
    raw["coverage_complete"] = False
    with pytest.raises(ValueError, match="complete"):
        await complete_copy_workflow(tmp_path, raw, profile, write_path=destination)
    raw["coverage_complete"] = True
    monkeypatch.setitem(SOURCE_LANGUAGES, ".new-source", "rust")
    with pytest.raises(ValueError, match="producer sources changed"):
        await complete_copy_workflow(tmp_path, raw, profile, write_path=destination)
    assert not destination.exists()


def test_forged_ids_and_duplicate_physical_signatures_are_rejected(profile):
    raw = report(profile, [("a.rs", "b.rs")])
    raw["fingerprints"][0]["evidence_id"] = "f" * 64
    with pytest.raises(ValueError, match="identity is not bound"):
        parse_copy_report(raw, profile)
    raw = report(profile, [("a.rs", "b.rs"), ("a.rs", "b.rs")])
    raw["fingerprints"][1]["fingerprint"] = "2" * 16
    raw["fingerprints"][1]["evidence_id"] = source_evidence_id(raw["fingerprints"][1])
    with pytest.raises(ValueError, match="physical copy signature"):
        parse_copy_report(raw, profile)


@pytest.mark.parametrize(
    "selection", ["scope", "include_tests", "include_docs", "include_same_file"]
)
def test_actual_source_selection_contract_must_match_frozen_profile(profile, selection):
    raw = report(profile, [("a.rs", "b.rs")])
    raw["source_selection"][selection] = "src/" if selection == "scope" else True
    with pytest.raises(ValueError, match="source selection"):
        parse_copy_report(raw, profile)


async def test_unicode_normalization_alias_cannot_overwrite_selected_source(tmp_path, profile):
    from victor.core.graph_rag.copy_artifacts import write_copy_baseline

    evidence = parse_copy_report(report(profile, [("é.rs", "b.rs")]), profile)
    source = tmp_path / "é.rs"
    old = make_baseline(evidence).to_dict()
    old["counts"][0]["count"] += 1
    source.write_text(json.dumps(old))
    destination = tmp_path / "e\u0301.rs"
    if not destination.exists() or not destination.samefile(source):
        pytest.skip("Filesystem distinguishes NFC/NFD filenames")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="filesystem alias"):
        await write_copy_baseline(destination, evidence, source_root=tmp_path)
    assert source.read_bytes() == before
