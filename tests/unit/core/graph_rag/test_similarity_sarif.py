"""SARIF preserves evidence channels, family boundaries and incomplete analysis."""

import copy
from pathlib import Path

import pytest

from victor.core.graph_rag.similarity_sarif import similarity_sarif


def family():
    return {
        "tier": "near",
        "score": 0.98,
        "score_basis": "maximum_matching_edge",
        "matching_edges": 2,
        "min_edge_similarity": 0.82,
        "members": [
            {"file": name, "line": 2, "end_line": 5, "source_hash": "a" * 64}
            for name in ("a #?.rs", "b%20.rs", "nested/λ.rs")
        ],
    }


def payload():
    return {
        "evidence": "normalized_syntax",
        "coverage_complete": True,
        "equivalence_verified": False,
        "total_families": 1,
        "families": [family()],
        "suppressed": [],
        "stats": {"source_files": 3},
        "source_manifest": {
            member["file"]: member["source_hash"] for member in family()["members"]
        },
    }


def body_payload():
    from dataclasses import asdict
    from victor.core.graph_rag.body_clones import BodyOptions
    from victor.core.graph_rag.body_provenance import producer_profile

    raw = payload()
    options = BodyOptions()
    producer = producer_profile(options.identity, component_hash="b" * 64, custom_id="sarif-test")
    raw["evidence"] = "function_body_embeddings"
    raw["profile"] = {
        **asdict(options),
        "identity": options.identity,
        "local_files_only": True,
        "quality_calibrated": False,
        "producer": {"identity": producer.identity, **producer.manifest},
    }
    raw["source_manifest"] = {
        member["file"]: member["source_hash"] for member in family()["members"]
    }
    return raw


def test_three_member_family_is_one_finding_with_related_members(tmp_path):
    report = similarity_sarif(tmp_path, payload(), mode="structural")
    run = report["runs"][0]
    assert run["invocations"][0]["executionSuccessful"]
    assert len(run["results"]) == 1
    result = run["results"][0]
    assert result["ruleId"] == "victor/structural-near"
    assert result["properties"]["matching_edges"] == 2
    assert result["properties"]["member_count"] == 3
    assert result["properties"]["equivalence_verified"] is False
    assert len(result["locations"]) == 1
    assert len(result["relatedLocations"]) == 2
    assert "baselineState" not in result and "fixes" not in result
    assert [a["location"]["uri"] for a in run["artifacts"]] == [
        "a%20%23%3F.rs",
        "b%2520.rs",
        "nested/%CE%BB.rs",
    ]
    assert all(a["hashes"]["sha-256"] == "a" * 64 for a in run["artifacts"])


def test_partial_hybrid_keeps_successful_channel_and_snapshot_conflicts(tmp_path):
    body = {"coverage_complete": False, "error": "local model unavailable"}
    raw = {
        "coverage_complete": False,
        "snapshot_conflicts": ["a #?.rs"],
        "channels": {"structural": payload(), "body": body},
    }
    run = similarity_sarif(tmp_path, raw, mode="hybrid")["runs"][0]
    assert len(run["results"]) == 1
    assert not run["invocations"][0]["executionSuccessful"]
    assert run["properties"]["snapshot_conflicts"] == ["a #?.rs"]
    assert run["properties"]["channels"]["body"]["error"] == "local model unavailable"


def test_body_profile_cohesion_and_explicit_truncation_survive_export(tmp_path):
    raw = body_payload()
    raw["families"][0].update(
        relation="body_similarity_candidate", cohesion={"minimum_is_verified": False}
    )
    raw["total_families"] = 8
    run = similarity_sarif(tmp_path, raw, mode="body")["runs"][0]
    assert run["properties"]["channels"]["body"]["profile"] == raw["profile"]
    assert run["properties"]["report_truncated"]
    assert run["results"][0]["properties"]["cohesion"] == {"minimum_is_verified": False}
    assert run["invocations"][0]["executionSuccessful"]


def test_summary_freshness_is_unknown_and_missing_line_is_not_invented(tmp_path):
    raw = {
        "evidence": "symbol-summary-v1",
        "analysis_complete": True,
        "total_families": 1,
        "families": [
            {
                "similarity": 1.0,
                "matching_pairs": 1,
                "members": [{"file": "a.py", "line": 0}, {"file": "b.py", "line": 7}],
            }
        ],
    }
    run = similarity_sarif(tmp_path, raw, mode="semantic")["runs"][0]
    assert run["properties"]["source_freshness"] == "unverified_index_snapshot"
    assert run["properties"]["coverage_complete"] is None
    assert run["invocations"][0]["executionSuccessful"]
    assert "region" not in run["results"][0]["locations"][0]["physicalLocation"]
    assert "hashes" not in run["artifacts"][0]
    assert run["results"][0]["properties"]["score_basis"] == "maximum_matching_edge"


@pytest.mark.parametrize("bad", ["../outside.rs", "/outside.rs", "x/../y.rs", ""])
def test_invalid_paths_fail_execution_without_losing_other_channels(tmp_path, bad):
    raw = payload()
    raw["families"][0]["members"][0]["file"] = bad
    hybrid = {"coverage_complete": True, "channels": {"structural": raw, "body": body_payload()}}
    run = similarity_sarif(tmp_path, hybrid, mode="hybrid")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert len(run["results"]) == 1
    assert run["results"][0]["ruleId"] == "victor/body-similarity"


@pytest.mark.parametrize("field,value", [("score", float("nan")), ("matching_edges", -1)])
def test_malformed_evidence_cannot_produce_success(tmp_path, field, value):
    raw = payload()
    raw["families"][0][field] = value
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert not run["results"]


def test_suppression_and_ffi_are_retained_separately_from_duplicate_families(tmp_path):
    raw = payload()
    raw["suppressed"] = [copy.deepcopy(family())]
    raw["suppressed"][0]["suppressed"] = "abstract_stub"
    raw["ffi_links"] = [{"evidence": "jni_declaration_export", "abi_verified": False}]
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert len(run["results"]) == 2
    assert run["results"][1]["properties"]["suppressed"] == "abstract_stub"
    assert run["properties"]["channels"]["structural"]["ffi_links"] == raw["ffi_links"]


def test_report_bounds_and_unknown_channel_are_unsuccessful(tmp_path):
    raw = {"coverage_complete": True, "channels": {"mystery": payload()}}
    run = similarity_sarif(tmp_path, raw, mode="hybrid")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    raw = payload()
    raw["families"] = [family()] * 201
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]


def test_report_errors_override_a_contradictory_complete_flag(tmp_path):
    raw = payload()
    raw["errors"] = [{"file": "bad.rs", "error": "cannot read"}]
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert run["results"]


def test_byte_limit_fails_without_corrupting_artifact_references(tmp_path, monkeypatch):
    from victor.core.graph_rag import similarity_sarif as module

    monkeypatch.setattr(module, "_MAX_BYTES", 2200)
    raw = payload()
    raw["families"] = [family()] * 2
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    for result in run["results"]:
        for location in result["locations"] + result["relatedLocations"]:
            index = location["physicalLocation"]["artifactLocation"]["index"]
            assert index < len(run["artifacts"])


def test_snapshot_versions_keep_distinct_hashes_for_one_uri(tmp_path):
    structural = payload()
    body = body_payload()
    for member in body["families"][0]["members"]:
        member["source_hash"] = "b" * 64
        body["source_manifest"][member["file"]] = "b" * 64
    raw = {
        "coverage_complete": True,
        "snapshot_conflicts": [],
        "channels": {
            "structural": structural,
            "body": body,
            "tokens": {"coverage_complete": True, "evidence": "token_sequence", "fragments": []},
        },
    }
    run = similarity_sarif(tmp_path, raw, mode="hybrid")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert len(run["artifacts"]) == 6
    assert sorted(run["properties"]["snapshot_conflicts"]) == sorted(
        member["file"] for member in family()["members"]
    )
    assert {a["hashes"]["sha-256"] for a in run["artifacts"]} == {"a" * 64, "b" * 64}


@pytest.mark.parametrize(
    "mode,raw",
    [
        ("structural", {"coverage_complete": True}),
        (
            "body",
            {"coverage_complete": True, "evidence": "function_body_embeddings", "families": []},
        ),
        ("semantic", {"analysis_complete": True, "evidence": "unknown", "families": []}),
        (
            "semantic",
            {
                "analysis_complete": True,
                "evidence": "symbol-summary-v1",
                "families": [],
                "embedding_profile_ids": ["garbage"],
            },
        ),
    ],
)
def test_unknown_or_missing_success_envelopes_cannot_pass(tmp_path, mode, raw):
    run = similarity_sarif(tmp_path, raw, mode=mode)["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]


def test_producer_identity_and_member_manifest_binding_are_verified(tmp_path):
    raw = body_payload()
    raw["profile"]["producer"]["identity"] = "0" * 64
    assert not similarity_sarif(tmp_path, raw, mode="body")["runs"][0]["invocations"][0][
        "executionSuccessful"
    ]
    raw = body_payload()
    raw["source_manifest"]["a #?.rs"] = "b" * 64
    run = similarity_sarif(tmp_path, raw, mode="body")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert not run["results"]


def test_deep_metadata_fails_with_valid_channel_preserved(tmp_path):
    nested = {}
    for _ in range(1100):
        nested = {"nested": nested}
    bad = payload()
    bad["stats"] = nested
    raw = {"coverage_complete": True, "channels": {"structural": bad, "body": body_payload()}}
    run = similarity_sarif(tmp_path, raw, mode="hybrid")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert len(run["results"]) == 1


@pytest.mark.parametrize("mode", ["body", "semantic"])
def test_malformed_partial_profile_cannot_publish_findings(tmp_path, mode):
    raw = (
        body_payload()
        if mode == "body"
        else {
            "analysis_complete": False,
            "evidence": "symbol-summary-v1",
            "families": [family()],
            "embedding_profile_ids": ["garbage"],
        }
    )
    if mode == "body":
        raw["coverage_complete"] = False
        raw["profile"] = {"identity": "not-valid"}
    run = similarity_sarif(tmp_path, raw, mode=mode)["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert not run["results"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence", "unknown-proof"),
        ("score_basis", "all_pairs_equivalent"),
        ("cohesion", {"pairwise_minimum_verified": True}),
    ],
)
def test_unknown_or_stronger_family_claims_are_rejected(tmp_path, field, value):
    raw = payload()
    raw["families"][0][field] = value
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert not run["results"]


def test_read_only_export_does_not_resolve_source_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("must not resolve sources"))
    assert similarity_sarif(tmp_path, payload(), mode="structural")["runs"][0]["results"]


@pytest.mark.parametrize(
    "field,value",
    [("score", None), ("matching_edges", 4), ("matching_edges", 0), ("min_edge_similarity", 0.99)],
)
def test_family_metrics_must_match_the_reported_component(tmp_path, field, value):
    raw = payload()
    raw["families"][0][field] = value
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert not run["results"]


def test_reported_score_rounding_does_not_reject_a_real_matching_minimum(tmp_path):
    raw = payload()
    raw["families"][0].update(score=0.9812, min_edge_similarity=0.981234)
    run = similarity_sarif(tmp_path, raw, mode="structural")["runs"][0]
    assert run["invocations"][0]["executionSuccessful"]


def test_token_regions_are_bound_to_acquired_line_counts(tmp_path):
    raw = {
        "coverage_complete": True,
        "evidence": "token_sequence",
        "backend": {"name": "jscpd", "version": "5.4.2"},
        "stats": {"source_files": 2},
        "source_manifest": {name: {"sha256": "a" * 64, "lines": 4} for name in ("a.py", "b.py")},
        "fragments": [
            {
                "kind": "exact",
                "tokens": 30,
                "format": "python",
                "fragment_hash": "b" * 64,
                "members": [
                    {"file": name, "line": 999, "end_line": 1000, "source_hash": "a" * 64}
                    for name in ("a.py", "b.py")
                ],
            }
        ],
    }
    run = similarity_sarif(tmp_path, raw, mode="tokens")["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert not run["results"]


def test_oversized_scalar_is_refused_before_whole_string_serialization():
    import tracemalloc
    from victor.core.graph_rag.similarity_sarif import _json_size

    raw = {"stats": {"note": "x" * (1024**2)}}
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="scalar exceeds"):
            _json_size(raw, 1024)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024


@pytest.mark.parametrize(
    "raw", ['λ😀\n"\\', {"items": [True, None, 3, -0.25, ()]}, {"control": "\0\t"}]
)
def test_bounded_json_size_matches_actual_escaped_utf8_bytes(raw):
    import json
    from victor.core.graph_rag.similarity_sarif import _json_size

    assert _json_size(raw, 10000) == len(json.dumps(raw, allow_nan=False).encode())


def test_final_writer_budgets_any_report_and_returns_a_bounded_failure(tmp_path, monkeypatch):
    import json
    from victor.core.graph_rag import similarity_sarif as module

    report = similarity_sarif(tmp_path, payload(), mode="structural")
    report["runs"][0]["properties"]["oversized"] = "x" * 10000
    monkeypatch.setattr(module, "_MAX_BYTES", 3000)
    text, successful = module.bounded_sarif_text(report)
    assert not successful
    assert len(text.encode()) + 2 <= 3000
    assert not json.loads(text)["runs"][0]["invocations"][0]["executionSuccessful"]
