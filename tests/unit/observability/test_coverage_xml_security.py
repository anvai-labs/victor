"""Coverage parsing accepts JaCoCo doctypes but rejects active XML content."""

import pytest

from victor.observability.pipeline import analyzers


@pytest.mark.parametrize("analyzer", [analyzers.CoberturaAnalyzer, analyzers.JaCoCoAnalyzer])
async def test_entity_report_is_not_used_for_coverage(tmp_path, analyzer):
    report = tmp_path / "coverage.xml"
    report.write_text(
        '<!DOCTYPE coverage [<!ENTITY value "99">]><coverage lines-covered="&value;"/>'
    )
    with pytest.raises(ValueError, match="entity declarations"):
        analyzers._parse_coverage_xml(report)
    result = await analyzer().parse_report(report)
    assert result.report_path == report


def test_jacoco_external_doctype_is_not_loaded(tmp_path):
    report = tmp_path / "jacoco.xml"
    report.write_text(
        '<!DOCTYPE report PUBLIC "-//JACOCO//DTD Report 1.1//EN" '
        '"http://127.0.0.1:1/never-fetch.dtd"><report name="test"/>'
    )
    root = analyzers._parse_coverage_xml(report)
    assert root.tag == "report"
    assert root.get("name") == "test"


def test_external_entity_is_rejected(tmp_path):
    secret = tmp_path / "sentinel.txt"
    secret.write_text("must not enter coverage output")
    report = tmp_path / "coverage.xml"
    report.write_text(
        f'<!DOCTYPE coverage [<!ENTITY external SYSTEM "{secret.as_uri()}">]>'
        "<coverage>&external;</coverage>"
    )
    with pytest.raises(ValueError, match="entity declarations"):
        analyzers._parse_coverage_xml(report)


def test_oversized_report_is_rejected_before_xml_parsing(tmp_path, monkeypatch):
    monkeypatch.setattr(analyzers, "_MAX_COVERAGE_XML_BYTES", 32)
    report = tmp_path / "coverage.xml"
    report.write_text("<coverage>" + " " * 100 + "</coverage>")
    with pytest.raises(ValueError, match="limit"):
        analyzers._parse_coverage_xml(report)
