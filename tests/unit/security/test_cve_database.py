"""Real advisory ratings and incomplete scans must reach the policy boundary."""

import json
import sqlite3
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from victor.security.cve_database import CVEDatabaseError, LocalCVECache, OSVDatabase
from victor.security.manager import SecurityManager
from victor.security.protocol import (
    SecurityDependency,
    SecurityPolicy,
    SecurityScanResult,
    Severity,
    Vulnerability,
)
from victor.security.scanner import SecurityScanner

CRITICAL = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
HIGH = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N"


def advisory(entries):
    return {"id": "CVE-2026-12345", "severity": entries, "summary": "test advisory"}


@pytest.mark.parametrize(
    "kind,vector,score,severity",
    [
        ("CVSS_V3", CRITICAL, 9.8, Severity.CRITICAL),
        ("CVSS_V3", HIGH, 7.5, Severity.HIGH),
        ("CVSS_V3", "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0, Severity.NONE),
        ("CVSS_V2", "AV:N/AC:L/Au:N/C:C/I:C/A:C", 10.0, Severity.CRITICAL),
        (
            "CVSS_V4",
            "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:N",
            9.9,
            Severity.CRITICAL,
        ),
    ],
)
def test_standard_vectors_produce_actual_base_scores(kind, vector, score, severity):
    cve = OSVDatabase()._parse_osv_vuln(advisory([{"type": kind, "score": vector}]))
    assert cve.severity is severity
    assert cve.cvss.score == score
    assert cve.cvss.vector == vector


@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_ratings_cannot_downgrade_the_most_severe_finding(reverse):
    entries = [{"type": "CVSS_V3", "score": s} for s in (CRITICAL, HIGH)]
    cve = OSVDatabase()._parse_osv_vuln(advisory(entries[::-1] if reverse else entries))
    assert cve.cvss.score == 9.8
    assert not SecurityPolicy().check(scan_result(cve))[0]


def test_package_specific_ratings_are_not_dropped():
    data = advisory([])
    data["affected"] = [{"severity": [{"type": "CVSS_V3", "score": CRITICAL}]}]
    assert OSVDatabase()._parse_osv_vuln(data).severity is Severity.CRITICAL


@pytest.mark.parametrize(
    "entries",
    [
        [],
        None,
        {},
        [None],
        [{"type": "CVSS_V3", "score": "9.8"}],
        [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N"}],
        [{"type": "CVSS_V3", "score": "x" * 4097}],
        [{"type": "CVSS_V5", "score": CRITICAL}],
        [{"type": [], "score": CRITICAL}],
        [{"type": "CVSS_V3", "score": HIGH}, {"type": "CVSS_V3", "score": "invalid"}],
    ],
)
def test_missing_or_unusable_ratings_remain_visible_and_fail_policy(entries):
    cve = OSVDatabase()._parse_osv_vuln(advisory(entries))
    assert cve.severity is Severity.UNKNOWN
    assert cve.cvss is None
    result = scan_result(cve)
    assert result.unknown_count == 1
    passed, reasons = SecurityPolicy().check(result)
    assert not passed
    assert any("unknown severity" in reason for reason in reasons)


def scan_result(cve):
    return SecurityScanResult(
        vulnerabilities=[Vulnerability(cve=cve, dependency=SecurityDependency("demo", "1", "pypi"))]
    )


def test_unknown_rating_is_preserved_in_all_report_formats(tmp_path):
    cve = OSVDatabase()._parse_osv_vuln(advisory([]))
    result = scan_result(cve)
    with patch("victor.security.manager.get_scanner"):
        manager = SecurityManager(project_root=tmp_path)
    data = json.loads(manager._generate_json_report(result))
    assert data["summary"]["by_severity"]["unknown"] == 1
    assert data["vulnerabilities"][0]["cvss_score"] is None
    assert data["policy"]["passed"] is False
    assert "UNKNOWN VULNERABILITIES" in manager._generate_text_report(result)
    assert "### Unknown" in manager._generate_markdown_report(result)


def test_legacy_cache_scores_are_invalidated_and_new_scores_round_trip(tmp_path):
    cache = LocalCVECache(tmp_path / "cve.db")
    cve = OSVDatabase()._parse_osv_vuln(advisory([{"type": "CVSS_V3", "score": CRITICAL}]))
    cache.set_cve(cve)
    assert cache.get_cve(cve.cve_id).cvss.score == 9.8
    data = cache._serialize_cve(cve)
    data.pop("cache_format")
    data["cvss"]["score"] = 5.0
    with sqlite3.connect(cache.cache_path) as conn:
        conn.execute("UPDATE cves SET data = ?", (json.dumps(data),))
    assert cache.get_cve(cve.cve_id) is None


@pytest.mark.parametrize("lookup", [False, True])
@respx.mock
async def test_failed_osv_requests_cannot_return_clean_results(lookup):
    respx.route().mock(return_value=httpx.Response(503))
    database = OSVDatabase(rate_limit_delay=0)
    with pytest.raises(CVEDatabaseError):
        if lookup:
            await database.lookup_cve("CVE-2026-12345")
        else:
            await database.search_by_package("demo", "pypi", "1")


async def test_single_file_scan_records_backend_failure_and_fails_policy(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("demo==1\n")
    backend = AsyncMock()
    backend.search_by_package.side_effect = CVEDatabaseError("unavailable")
    result = await SecurityScanner(cve_db=backend).scan_file(path)
    assert result.errors
    assert not SecurityPolicy().check(result)[0]


@pytest.mark.parametrize(
    "kind,vector,score",
    [
        (
            "CVSS_V4",
            "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:N/MVC:N/MVI:N/MVA:N/MSC:N/MSI:N/MSA:N",
            9.9,
        ),
        ("CVSS_V4", "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:H/SI:H/SA:N/E:U", 9.9),
        ("CVSS_V3", CRITICAL + "/E:U/RL:O/RC:U/MC:N/MI:N/MA:N", 9.8),
        ("CVSS_V2", "AV:N/AC:L/Au:N/C:C/I:C/A:C/E:U/RL:OF/RC:UC", 10.0),
    ],
)
def test_optional_metrics_cannot_lower_advisory_base_severity(kind, vector, score):
    cve = OSVDatabase()._parse_osv_vuln(advisory([{"type": kind, "score": vector}]))
    assert cve.cvss.score == score
    assert cve.cvss.vector == vector
    assert not SecurityPolicy().check(scan_result(cve))[0]


@respx.mock
async def test_token_only_page_follows_continuation_to_critical_finding():
    route = respx.post("https://api.osv.dev/v1/query").mock(
        side_effect=[
            httpx.Response(200, json={"next_page_token": "page2"}),
            httpx.Response(
                200, json={"vulns": [advisory([{"type": "CVSS_V3", "score": CRITICAL}])]}
            ),
        ]
    )
    cves = await OSVDatabase(rate_limit_delay=0).search_by_package("demo", "pypi", "2")
    assert len(cves) == 1
    assert not SecurityPolicy().check(scan_result(cves[0]))[0]
    assert json.loads(route.calls[1].request.content) == {
        "package": {"name": "demo", "ecosystem": "PyPI"},
        "version": "2",
        "page_token": "page2",
    }


@pytest.mark.parametrize(
    "body",
    [
        [],
        None,
        {"vulns": {}},
        {"vulns": None},
        {"vulns": [None]},
        {"vulns": [{}]},
        {"vulns": [{"id": ""}]},
        {"next_page_token": None},
        {"next_page_token": ""},
        {"next_page_token": 42},
    ],
)
@respx.mock
async def test_malformed_success_responses_fail_instead_of_clearing_findings(body):
    respx.post("https://api.osv.dev/v1/query").respond(200, json=body)
    with pytest.raises(CVEDatabaseError):
        await OSVDatabase(rate_limit_delay=0).search_by_package("demo", "pypi", "1")


@pytest.mark.parametrize(
    "last_response",
    [
        httpx.Response(503),
        httpx.Response(200, json={"next_page_token": "page2"}),
    ],
)
@respx.mock
async def test_incomplete_pagination_does_not_publish_partial_cache(tmp_path, last_response):
    cache = LocalCVECache(tmp_path / "cve.db")
    respx.post("https://api.osv.dev/v1/query").mock(
        side_effect=[
            httpx.Response(200, json={"vulns": [advisory([])], "next_page_token": "page2"}),
            last_response,
        ]
    )
    with pytest.raises(CVEDatabaseError):
        await OSVDatabase(cache=cache, rate_limit_delay=0).search_by_package("demo", "pypi", "1")
    assert cache.get_cve("CVE-2026-12345") is None


async def test_cache_isolates_package_versions_and_caches_complete_empty_queries(tmp_path):
    from victor.security.cve_database import CachingCVEDatabase

    cache = LocalCVECache(tmp_path / "cve.db")
    backend = AsyncMock()
    critical = OSVDatabase()._parse_osv_vuln(advisory([{"type": "CVSS_V3", "score": CRITICAL}]))
    backend.search_by_package.side_effect = [[], [critical], [], []]
    database = CachingCVEDatabase(backend, cache)
    assert await database.search_by_package("demo", "pypi", "1") == []
    assert await database.search_by_package("demo", "pypi", "1") == []
    findings = await database.search_by_package("demo", "pypi", "2")
    assert findings[0].severity is Severity.CRITICAL
    assert await database.search_by_package("demo", "pypi", None) == []
    assert await database.search_by_package("demo", "pypi", "") == []
    assert backend.search_by_package.await_count == 4


async def test_missing_cached_advisory_refreshes_whole_query_and_cannot_return_partial(tmp_path):
    from victor.security.cve_database import CachingCVEDatabase

    cache = LocalCVECache(tmp_path / "cve.db")
    cve = OSVDatabase()._parse_osv_vuln(advisory([]))
    cache.set_cve(cve)
    cache.set_package_cves("demo", "pypi", [(cve.cve_id, "", ""), ("missing", "", "")], version="1")
    backend = AsyncMock()
    backend.search_by_package.side_effect = CVEDatabaseError("unavailable")
    with pytest.raises(CVEDatabaseError):
        await CachingCVEDatabase(backend, cache).search_by_package("demo", "pypi", "1")
    backend.search_by_package.assert_awaited_once_with("demo", "pypi", "1")


@pytest.mark.parametrize("invalidity", ["missing", "legacy", "expired"])
async def test_offline_index_cannot_silently_drop_unavailable_advisories(tmp_path, invalidity):
    from victor.security.cve_database import OfflineCVEDatabase

    directory = tmp_path / "advisories"
    directory.mkdir()
    (directory / "demo.json").write_text(
        json.dumps({"advisories": [{"package": "pypi:demo", "cve_id": "CVE-2026-12345"}]})
    )
    database = OfflineCVEDatabase(tmp_path)
    cve = OSVDatabase()._parse_osv_vuln(advisory([]))
    if invalidity != "missing":
        database._cache.set_cve(cve)
        with sqlite3.connect(database._cache.cache_path) as conn:
            if invalidity == "legacy":
                data = database._cache._serialize_cve(cve)
                data.pop("cache_format")
                conn.execute("UPDATE cves SET data = ?", (json.dumps(data),))
            else:
                conn.execute("UPDATE cves SET cached_at = '2000-01-01 00:00:00'")
    with pytest.raises(CVEDatabaseError, match="unavailable or expired"):
        await database.search_by_package("demo", "pypi", "1")


async def test_offline_clean_requires_exact_completed_cached_query(tmp_path):
    from victor.security.cve_database import OfflineCVEDatabase

    database = OfflineCVEDatabase(tmp_path)
    with pytest.raises(CVEDatabaseError, match="No offline advisory coverage"):
        await database.search_by_package("demo", "pypi", "1")
    database._cache.set_package_cves("demo", "pypi", [], version="1")
    assert await database.search_by_package("demo", "pypi", "1") == []
    with pytest.raises(CVEDatabaseError):
        await database.search_by_package("demo", "pypi", "2")


async def test_legacy_versionless_query_cache_is_not_reused(tmp_path):
    from victor.security.cve_database import CachingCVEDatabase

    cache = LocalCVECache(tmp_path / "cve.db")
    with sqlite3.connect(cache.cache_path) as conn:
        conn.execute("CREATE TABLE package_cves (package_name TEXT, ecosystem TEXT, cve_id TEXT)")
        conn.execute("INSERT INTO package_cves VALUES ('demo', 'pypi', 'old')")
    backend = AsyncMock()
    backend.search_by_package.side_effect = CVEDatabaseError("unavailable")
    with pytest.raises(CVEDatabaseError):
        await CachingCVEDatabase(backend, cache).search_by_package("demo", "pypi", "2")


@pytest.mark.parametrize("reverse", [False, True])
@respx.mock
async def test_conflicting_duplicate_advisories_cannot_downgrade_cached_findings(tmp_path, reverse):
    from victor.security.cve_database import CachingCVEDatabase

    cache = LocalCVECache(tmp_path / "cve.db")
    records = [advisory([{"type": "CVSS_V3", "score": vector}]) for vector in [CRITICAL, HIGH]]
    respx.post("https://api.osv.dev/v1/query").respond(
        200, json={"vulns": records[::-1] if reverse else records}
    )
    database = CachingCVEDatabase(OSVDatabase(cache=cache, rate_limit_delay=0), cache)
    for _ in range(2):
        with pytest.raises(CVEDatabaseError, match="Conflicting advisory records"):
            await database.search_by_package("demo", "pypi", "1")
    assert cache.get_cve(records[0]["id"]) is None
    assert cache.get_package_cves("demo", "pypi", version="1") is None


@respx.mock
async def test_identical_duplicate_advisories_are_deduplicated():
    record = advisory([{"type": "CVSS_V3", "score": CRITICAL}])
    respx.post("https://api.osv.dev/v1/query").respond(200, json={"vulns": [record, record]})
    cves = await OSVDatabase(rate_limit_delay=0).search_by_package("demo", "pypi", "1")
    assert len(cves) == 1
    assert cves[0].severity is Severity.CRITICAL


@pytest.mark.parametrize(
    "payload", ["{}", "null", '""', "[[]]", '[["", "", ""]]', '[["id", 0, ""]]', "{"]
)
async def test_invalid_query_snapshots_cannot_claim_offline_clean_and_refresh_online(
    tmp_path, payload
):
    from victor.security.cve_database import CachingCVEDatabase, OfflineCVEDatabase

    offline = OfflineCVEDatabase(tmp_path)
    cache = offline._cache
    cache.set_package_cves("demo", "pypi", [], version="1")
    with sqlite3.connect(cache.cache_path) as conn:
        conn.execute("UPDATE package_queries_v2 SET data = ?", (payload,))
    with pytest.raises(CVEDatabaseError):
        await offline.search_by_package("demo", "pypi", "1")
    critical = OSVDatabase()._parse_osv_vuln(advisory([{"type": "CVSS_V3", "score": CRITICAL}]))
    backend = AsyncMock()
    backend.search_by_package.return_value = [critical]
    assert await CachingCVEDatabase(backend, cache).search_by_package("demo", "pypi", "1") == [
        critical
    ]
    backend.search_by_package.assert_awaited_once_with("demo", "pypi", "1")


@pytest.mark.parametrize(
    "name,content",
    [
        ("Pipfile.lock", "{"),
        ("poetry.lock", "["),
        ("pyproject.toml", "["),
        ("package.json", "{"),
        ("package-lock.json", "{"),
        ("Cargo.toml", "["),
        ("Cargo.lock", "["),
        ("setup.py", "from setuptools import setup"),
    ],
)
@pytest.mark.parametrize("whole_project", [False, True])
async def test_unparseable_dependencies_fail_policy_in_file_and_project_scans(
    tmp_path, name, content, whole_project
):
    path = tmp_path / name
    path.write_text(content)
    scanner = SecurityScanner(cve_db=AsyncMock())
    result = await scanner.scan(tmp_path) if whole_project else await scanner.scan_file(path)
    assert result.errors
    assert not SecurityPolicy().check(result)[0]


async def test_unreadable_dependency_file_cannot_report_clean(tmp_path):
    scanner = SecurityScanner(cve_db=AsyncMock())
    result = await scanner.scan_file(tmp_path / "requirements.txt")
    assert result.errors
    assert not SecurityPolicy().check(result)[0]
