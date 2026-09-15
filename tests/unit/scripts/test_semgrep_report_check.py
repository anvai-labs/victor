"""Advisory findings cannot hide failed or incomplete Semgrep execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/ci/semgrep_report_check.py"
SPEC = importlib.util.spec_from_file_location("semgrep_report_check", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def reports():
    result = {
        "check_id": "python.test.rule",
        "path": "victor/example.py",
        "start": {"line": 1},
        "end": {"line": 2},
        "extra": {"message": "Advisory finding"},
    }
    report = {
        "version": "1.177.0",
        "paths": {"scanned": ["victor/example.py"]},
        "skipped_rules": [],
        "errors": [],
        "results": [result],
    }
    sarif = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "Semgrep OSS", "semanticVersion": "1.177.0"}},
                "invocations": [{"executionSuccessful": True, "toolExecutionNotifications": []}],
                "results": [
                    {
                        "ruleId": "python.test.rule",
                        "message": {"text": "Advisory finding"},
                        "fingerprints": {"matchBasedId/v1": "preserve-me"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "victor/example.py"},
                                    "region": {"startLine": 1, "endLine": 2},
                                }
                            }
                        ],
                    }
                ],
            }
        ],
    }
    return report, sarif


@pytest.mark.parametrize("status", [1, 2, 3, 7, 13, 130])
def test_nonzero_status_is_operational_failure_even_with_findings(status):
    with pytest.raises(ValueError, match="operational failure"):
        gate.validate(*reports(), status)


def test_findings_are_advisory_and_raw_report_is_unchanged():
    report, sarif = reports()
    suppressed = deepcopy(sarif["runs"][0]["results"][0])
    suppressed["suppressions"] = [{"kind": "inSource"}]
    sarif["runs"][0]["results"].append(suppressed)
    original = deepcopy(sarif)
    output, warnings, summary = gate.validate(report, sarif, 0)
    assert sarif == original
    assert len(output["runs"][0]["results"]) == 1
    assert output["runs"][0]["results"][0]["fingerprints"] == {"matchBasedId/v1": "preserve-me"}
    assert not warnings
    assert "1 advisory findings" in summary
    assert "1 in-source suppressions" in summary


@pytest.mark.parametrize(
    "suppression",
    [
        {"kind": "external"},
        {"kind": "inSource", "status": "rejected"},
        {"kind": "inSource", "status": "underReview"},
    ],
)
def test_unaccepted_or_external_suppressions_do_not_hide_findings(suppression):
    report, sarif = reports()
    sarif["runs"][0]["results"][0]["suppressions"] = [suppression]
    output, _, _ = gate.validate(report, sarif, 0)
    assert len(output["runs"][0]["results"]) == 1


def test_partial_parsing_is_reported_as_a_coverage_gap():
    report, sarif = reports()
    report["errors"] = [
        {
            "code": 3,
            "level": "warn",
            "type": ["PartialParsing", []],
            "message": "Template Bash could not be fully parsed",
        }
    ]
    sarif["runs"][0]["invocations"][0]["toolExecutionNotifications"] = [
        {"level": "warning", "message": {"text": report["errors"][0]["message"]}}
    ]
    _, warnings, summary = gate.validate(report, sarif, 0)
    assert warnings == ["PartialParsing coverage gap: Template Bash could not be fully parsed"]
    assert "1 partial-parsing coverage warnings" in summary


@pytest.mark.parametrize(
    "kind,level", [("Timeout", "warn"), ("ParseError", "error"), ("PartialParsing", "error")]
)
def test_operational_diagnostics_fail_even_if_exit_status_is_zero(kind, level):
    report, sarif = reports()
    report["errors"] = [{"type": kind, "level": level, "message": "Incomplete scan"}]
    with pytest.raises(ValueError, match="scan incomplete"):
        gate.validate(report, sarif, 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", None),
        ("errors", None),
        ("results", None),
        ("paths", {}),
        ("skipped_rules", ["missing-rule"]),
    ],
)
def test_missing_or_incomplete_json_evidence_fails(field, value):
    report, sarif = reports()
    report[field] = value
    with pytest.raises(ValueError):
        gate.validate(report, sarif, 0)


def test_zero_scanned_files_is_not_clean_scan():
    report, sarif = reports()
    report["paths"]["scanned"] = []
    with pytest.raises(ValueError, match="no files"):
        gate.validate(report, sarif, 0)


@pytest.mark.parametrize("field,value", [("results", None), ("invocations", []), ("tool", {})])
def test_missing_sarif_evidence_fails(field, value):
    report, sarif = reports()
    sarif["runs"][0][field] = value
    with pytest.raises(ValueError):
        gate.validate(report, sarif, 0)


def test_false_sarif_execution_and_missing_findings_fail():
    report, sarif = reports()
    sarif["runs"][0]["invocations"][0]["executionSuccessful"] = False
    with pytest.raises(ValueError, match="Unsuccessful"):
        gate.validate(report, sarif, 0)
    _, sarif = reports()
    sarif["runs"][0]["results"] = []
    with pytest.raises(ValueError, match="inventory mismatch"):
        gate.validate(report, sarif, 0)


@pytest.mark.parametrize("level", ["error", "warning"])
def test_sarif_operational_diagnostics_cannot_hide_behind_success(level):
    report, sarif = reports()
    sarif["runs"][0]["invocations"][0]["toolExecutionNotifications"] = [
        {"level": level, "message": {"text": "scanner did not finish"}}
    ]
    with pytest.raises(ValueError, match="SARIF"):
        gate.validate(report, sarif, 0)


def test_partial_parsing_does_not_mask_an_unrelated_sarif_warning():
    report, sarif = reports()
    report["errors"] = [{"type": "PartialParsing", "level": "warn", "message": "template parsing"}]
    sarif["runs"][0]["invocations"][0]["toolExecutionNotifications"] = [
        {"level": "warning", "message": {"text": "template parsing"}},
        {"level": "warning", "message": {"text": "analysis timed out"}},
    ]
    with pytest.raises(ValueError, match="diagnostic inventory mismatch"):
        gate.validate(report, sarif, 0)


def test_json_and_sarif_match_url_encoded_paths():
    report, sarif = reports()
    report["results"][0]["path"] = "victor/a b.py"
    sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"][
        "uri"
    ] = "victor/a%20b.py"
    gate.validate(report, sarif, 0)


@pytest.mark.parametrize("bad_contents", [None, "{", "{}", "[]"])
def test_cli_fails_missing_or_malformed_reports(tmp_path, bad_contents):
    report, sarif = reports()
    report_path, sarif_path, output_path = (
        tmp_path / name for name in ("raw.json", "raw.sarif", "upload.sarif")
    )
    if bad_contents is not None:
        report_path.write_text(bad_contents)
    sarif_path.write_text(json.dumps(sarif))
    assert (
        gate.main(
            [str(report_path), str(sarif_path), "--status", "0", "--output", str(output_path)]
        )
        == 1
    )
    assert not output_path.exists()


def test_cli_retains_raw_evidence_and_escapes_annotations(tmp_path, monkeypatch, capsys):
    report, sarif = reports()
    report["errors"] = [
        {"type": "PartialParsing", "level": "warn", "message": "unparsed\n::error::injected%"}
    ]
    sarif["runs"][0]["invocations"][0]["toolExecutionNotifications"] = [
        {"level": "warning", "message": {"text": report["errors"][0]["message"]}}
    ]
    report_path, sarif_path, output_path, summary_path = (
        tmp_path / name for name in ("raw.json", "raw.sarif", "upload.sarif", "summary.md")
    )
    report_path.write_text(json.dumps(report))
    sarif_path.write_text(json.dumps(sarif))
    before = (report_path.read_bytes(), sarif_path.read_bytes())
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    assert (
        gate.main(
            [str(report_path), str(sarif_path), "--status", "0", "--output", str(output_path)]
        )
        == 0
    )
    assert (report_path.read_bytes(), sarif_path.read_bytes()) == before
    assert "unparsed%0A::error::injected%25" in capsys.readouterr().out
    assert "1 partial-parsing coverage warnings" in summary_path.read_text()
    assert (
        gate.main([str(report_path), str(sarif_path), "--status", "0", "--output", str(sarif_path)])
        == 1
    )
    assert sarif_path.read_bytes() == before[1]
