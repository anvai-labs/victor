"""The pinned fingerprint carrier must match the token report's source evidence."""

import copy

import pytest

from victor.core.graph_rag.token_fingerprints import parse_fingerprints


def reports(root):
    def json_location(name):
        return {
            "name": name,
            "start": 2,
            "end": 4,
            "startLoc": {"line": 2, "column": 0},
            "endLoc": {"line": 4, "column": 8},
        }

    def location(name):
        return {
            "physicalLocation": {
                "artifactLocation": {"uri": name, "uriBaseId": "%SRCROOT%"},
                "region": {"startLine": 2, "endLine": 4, "startColumn": 1, "endColumn": 9},
            }
        }

    duplicate = {
        "kind": "exact",
        "format": "python",
        "tokens": 30,
        "firstFile": json_location("a%name.py"),
        "secondFile": json_location("b.py"),
    }
    result = {
        "ruleId": "jscpd/duplicate-code",
        "locations": [location("a%name.py")],
        "relatedLocations": [location("b.py")],
        "partialFingerprints": {"jscpdCloneHash/v1": "1" * 16},
        "properties": {"clone_hash": "1" * 16, "token_count": 30},
    }
    sarif = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "jscpd", "version": "5.4.2"}},
                "originalUriBaseIds": {"%SRCROOT%": {"uri": root.as_uri() + "/"}},
                "results": [result],
            }
        ],
    }
    manifest = {name: {"sha256": "a" * 64, "lines": 5} for name in ("a%name.py", "b.py")}
    return sarif, [duplicate], manifest


def test_fingerprints_retain_raw_paths_and_verified_source_references(tmp_path):
    sarif, duplicates, manifest = reports(tmp_path)
    result = parse_fingerprints(sarif, duplicates, manifest, tmp_path, expected_version="5.4.2")
    assert result[0]["fingerprint"] == "1" * 16
    assert result[0]["format"] == "python"
    assert result[0]["tokens"] == 30
    assert result[0]["members"][0]["file"] == "a%name.py"
    assert result[0]["members"][0]["source_hash"] == "a" * 64


@pytest.mark.parametrize(
    "corrupt",
    [
        "hash",
        "version",
        "producer",
        "base",
        "outside",
        "span",
        "column",
        "tokens",
        "missing",
        "extra",
        "uri",
        "kind",
    ],
)
def test_mismatched_or_unsafe_fingerprint_carriers_are_rejected(tmp_path, corrupt):
    sarif, duplicates, manifest = reports(tmp_path)
    run = sarif["runs"][0]
    result = run["results"][0]
    if corrupt == "hash":
        result["partialFingerprints"]["jscpdCloneHash/v1"] = "invalid"
    elif corrupt == "version":
        sarif["version"] = "2.0.0"
    elif corrupt == "producer":
        run["tool"]["driver"]["version"] = "future"
    elif corrupt == "base":
        run["originalUriBaseIds"]["%SRCROOT%"]["uri"] = tmp_path.parent.as_uri() + "/"
    elif corrupt == "outside":
        result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] = "../secret.py"
    elif corrupt == "span":
        result["locations"][0]["physicalLocation"]["region"]["endLine"] = 100
    elif corrupt == "column":
        result["locations"][0]["physicalLocation"]["region"]["endColumn"] += 1
    elif corrupt == "tokens":
        result["properties"]["token_count"] = 31
    elif corrupt == "missing":
        run["results"] = []
    elif corrupt == "extra":
        run["results"].append(copy.deepcopy(result))
    elif corrupt == "uri":
        result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] = "a%25name.py"
    elif corrupt == "kind":
        duplicates[0]["kind"] = "semantic"
    with pytest.raises(ValueError):
        parse_fingerprints(sarif, duplicates, manifest, tmp_path, expected_version="5.4.2")


def test_valid_empty_reports_are_distinct_from_missing_fingerprint_evidence(tmp_path):
    sarif, _, manifest = reports(tmp_path)
    sarif["runs"][0]["results"] = []
    assert parse_fingerprints(sarif, [], manifest, tmp_path, expected_version="5.4.2") == []


@pytest.mark.parametrize("hash_", ["1" * 16, "2" * 16])
def test_identical_physical_evidence_cannot_inflate_fingerprint_counts(tmp_path, hash_):
    sarif, duplicates, manifest = reports(tmp_path)
    duplicates.append(copy.deepcopy(duplicates[0]))
    extra = copy.deepcopy(sarif["runs"][0]["results"][0])
    extra["partialFingerprints"]["jscpdCloneHash/v1"] = hash_
    extra["properties"]["clone_hash"] = hash_
    sarif["runs"][0]["results"].append(extra)
    with pytest.raises(ValueError, match="physical token evidence"):
        parse_fingerprints(sarif, duplicates, manifest, tmp_path, expected_version="5.4.2")
