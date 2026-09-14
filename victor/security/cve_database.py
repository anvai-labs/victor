# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CVE database client for vulnerability lookups.

Supports:
- NVD (National Vulnerability Database) API
- OSV (Open Source Vulnerabilities) API
- Local cache for offline/air-gapped mode
"""

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from victor.security.protocol import (
    CVE,
    CVSSMetrics,
    Severity,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class CVEDatabase(Protocol):
    """Protocol for CVE database clients."""

    async def lookup_cve(self, cve_id: str) -> Optional[CVE]:
        """Look up a CVE by ID."""
        ...

    async def search_by_package(
        self,
        package_name: str,
        ecosystem: str,
        version: Optional[str] = None,
    ) -> list[CVE]:
        """Search for CVEs affecting a package."""
        ...

    async def get_affected_versions(
        self,
        cve_id: str,
        package_name: str,
        ecosystem: str,
    ) -> list[str]:
        """Get affected versions for a CVE and package."""
        ...


class CVEDatabaseError(RuntimeError):
    """An advisory lookup failed; it cannot establish a clean result."""


def _unique_advisories(cves: list[CVE]) -> list[CVE]:
    """Reject inconsistent snapshots before global advisory cache publication."""
    records: dict[str, CVE] = {}
    for cve in cves:
        previous = records.get(cve.cve_id)
        if previous is not None and previous != cve:
            raise CVEDatabaseError(f"Conflicting advisory records: {cve.cve_id}")
        records[cve.cve_id] = cve
    return list(records.values())


class BaseCVEDatabase(ABC):
    """Abstract base class for CVE database clients."""

    @abstractmethod
    async def lookup_cve(self, cve_id: str) -> Optional[CVE]:
        """Look up a CVE by ID."""
        ...

    @abstractmethod
    async def search_by_package(
        self,
        package_name: str,
        ecosystem: str,
        version: Optional[str] = None,
    ) -> list[CVE]:
        """Search for CVEs affecting a package."""
        ...

    async def get_affected_versions(
        self,
        cve_id: str,
        package_name: str,
        ecosystem: str,
    ) -> list[str]:
        """Get affected versions. Default returns empty list."""
        return []


class LocalCVECache:
    """Local SQLite cache for CVE data."""

    def __init__(self, cache_path: Path):
        """Initialize the cache.

        Args:
            cache_path: Path to SQLite database file
        """
        self.cache_path = cache_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cves (
                    cve_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Query identity includes the exact version. A separate generation
            # deliberately ignores old versionless entries, including clean results.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS package_queries_v2 (
                    package_name TEXT NOT NULL,
                    ecosystem TEXT NOT NULL,
                    requested_version TEXT NOT NULL,
                    data TEXT NOT NULL,
                    cached_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (package_name, ecosystem, requested_version)
                )
            """)
            conn.commit()

    def get_cve(self, cve_id: str, max_age_hours: int = 24) -> Optional[CVE]:
        """Get a CVE from cache if fresh enough.

        Args:
            cve_id: CVE ID to look up
            max_age_hours: Maximum age of cached data

        Returns:
            CVE or None if not cached or stale
        """
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                "SELECT data, cached_at FROM cves WHERE cve_id = ?",
                (cve_id,),
            ).fetchone()

            if row is None:
                return None

            data_str, cached_at = row
            cached_time = datetime.fromisoformat(cached_at)

            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None  # Stale

            data = json.loads(data_str)
            if data.get("cache_format") != 2:
                return None  # Discard legacy entries with fabricated CVSS scores.
            return self._deserialize_cve(data)

    def set_cve(self, cve: CVE) -> None:
        """Cache a CVE.

        Args:
            cve: CVE to cache
        """
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cves (cve_id, data, cached_at)
                VALUES (?, ?, ?)
                """,
                (cve.cve_id, json.dumps(self._serialize_cve(cve)), datetime.now()),
            )
            conn.commit()

    def get_package_cves(
        self,
        package_name: str,
        ecosystem: str,
        max_age_hours: int = 24,
        *,
        version: Optional[str] = None,
    ) -> Optional[list[tuple[str, str, str]]]:
        """Return a complete query snapshot, or None when absent or expired.

        An empty list is a cached clean query, distinct from missing coverage.
        JSON encoding distinguishes an unspecified version from any string.
        """
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                """
                SELECT data FROM package_queries_v2
                WHERE package_name = ? AND ecosystem = ? AND requested_version = ?
                    AND cached_at > ?
                """,
                (
                    package_name,
                    ecosystem,
                    json.dumps(version),
                    datetime.now() - timedelta(hours=max_age_hours),
                ),
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
        except (ValueError, TypeError):
            return None
        if not isinstance(data, list) or any(
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(value, str) for value in item)
            or not item[0].strip()
            for item in data
        ):
            return None
        return [(item[0], item[1], item[2]) for item in data]

    def set_package_cves(
        self,
        package_name: str,
        ecosystem: str,
        cve_data: list[tuple[str, str, str]],
        *,
        version: Optional[str] = None,
    ) -> None:
        """Publish a completed package/version query, including clean results."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO package_queries_v2
                (package_name, ecosystem, requested_version, data, cached_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    package_name,
                    ecosystem,
                    json.dumps(version),
                    json.dumps(cve_data),
                    datetime.now(),
                ),
            )

    def _serialize_cve(self, cve: CVE) -> dict:
        """Serialize CVE to dict."""
        return {
            "cache_format": 2,
            "cve_id": cve.cve_id,
            "description": cve.description,
            "severity": cve.severity.value,
            "cvss": (
                {
                    "score": cve.cvss.score,
                    "vector": cve.cvss.vector,
                }
                if cve.cvss
                else None
            ),
            "published_date": (cve.published_date.isoformat() if cve.published_date else None),
            "modified_date": (cve.modified_date.isoformat() if cve.modified_date else None),
            "references": cve.references,
            "cwe_ids": cve.cwe_ids,
            "affected_products": cve.affected_products,
        }

    def _deserialize_cve(self, data: dict) -> CVE:
        """Deserialize CVE from dict."""
        cvss = None
        if data.get("cvss"):
            cvss = CVSSMetrics(
                score=data["cvss"]["score"],
                vector=data["cvss"].get("vector", ""),
            )

        return CVE(
            cve_id=data["cve_id"],
            description=data["description"],
            severity=Severity(data["severity"]),
            cvss=cvss,
            published_date=(
                datetime.fromisoformat(data["published_date"])
                if data.get("published_date")
                else None
            ),
            modified_date=(
                datetime.fromisoformat(data["modified_date"]) if data.get("modified_date") else None
            ),
            references=data.get("references", []),
            cwe_ids=data.get("cwe_ids", []),
            affected_products=data.get("affected_products", []),
        )


class OSVDatabase(BaseCVEDatabase):
    """Client for the OSV (Open Source Vulnerabilities) database.

    OSV is a distributed vulnerability database for open source software.
    Free to use, no API key required.
    """

    API_URL = "https://api.osv.dev/v1"

    def __init__(
        self,
        cache: Optional[LocalCVECache] = None,
        rate_limit_delay: float = 0.1,
    ):
        """Initialize the OSV client.

        Args:
            cache: Optional local cache
            rate_limit_delay: Delay between API calls in seconds
        """
        self._cache = cache
        self._rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

    async def lookup_cve(self, cve_id: str) -> Optional[CVE]:
        """Look up a CVE by ID."""
        # Check cache first
        if self._cache:
            cached = self._cache.get_cve(cve_id)
            if cached:
                return cached

        # Query OSV API
        try:
            import httpx

            await self._rate_limit()

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.API_URL}/vulns/{cve_id}",
                    timeout=30.0,
                )

                if response.status_code == 404:
                    return None

                response.raise_for_status()
                data = response.json()

                cve = self._parse_osv_vuln(data)
                if cve and self._cache:
                    self._cache.set_cve(cve)

                return cve

        except ImportError:
            raise CVEDatabaseError("OSV lookup dependencies are unavailable") from None
        except Exception as e:
            raise CVEDatabaseError(f"OSV lookup failed: {e}") from e

    async def search_by_package(
        self,
        package_name: str,
        ecosystem: str,
        version: Optional[str] = None,
    ) -> list[CVE]:
        """Search for CVEs affecting a package."""
        # Map ecosystem names
        osv_ecosystem = self._map_ecosystem(ecosystem)

        try:
            import httpx

            query: dict = {"package": {"name": package_name, "ecosystem": osv_ecosystem}}
            if version is not None:
                query["version"] = version

            cves = []
            seen_tokens: set[str] = set()
            async with httpx.AsyncClient() as client:
                # Bound a broken server's continuation chain; exhaustion is an
                # incomplete scan, never a partial successful result.
                for _ in range(1000):
                    await self._rate_limit()
                    response = await client.post(
                        f"{self.API_URL}/query",
                        json=query,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict) or not isinstance(data.get("vulns", []), list):
                        raise CVEDatabaseError("Malformed OSV query response")
                    for vuln in data.get("vulns", []):
                        cves.append(self._parse_osv_vuln(vuln))
                    if "next_page_token" not in data:
                        break
                    token = data["next_page_token"]
                    if not isinstance(token, str) or not token or token in seen_tokens:
                        raise CVEDatabaseError("Invalid or repeated OSV continuation token")
                    seen_tokens.add(token)
                    query["page_token"] = token
                else:
                    raise CVEDatabaseError("OSV pagination limit exceeded")

            cves = _unique_advisories(cves)
            # Only publish findings once every page is valid and complete.
            if self._cache:
                for cve in cves:
                    self._cache.set_cve(cve)
            return cves

        except ImportError:
            raise CVEDatabaseError("OSV lookup dependencies are unavailable") from None
        except Exception as e:
            raise CVEDatabaseError(f"OSV search failed: {e}") from e

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            import asyncio

            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _map_ecosystem(self, ecosystem: str) -> str:
        """Map ecosystem name to OSV format."""
        mapping = {
            "pypi": "PyPI",
            "npm": "npm",
            "cargo": "crates.io",
            "maven": "Maven",
            "go": "Go",
            "nuget": "NuGet",
            "rubygems": "RubyGems",
            "packagist": "Packagist",
        }
        return mapping.get(ecosystem.lower(), ecosystem)

    @staticmethod
    def _parse_osv_cvss(data: dict) -> Optional[CVSSMetrics]:
        """Use the highest reported base score; incomplete ratings remain unknown."""
        from cvss import CVSS2, CVSS3, CVSS4
        from cvss.exceptions import CVSSError

        parsers = {"CVSS_V2": CVSS2, "CVSS_V3": CVSS3, "CVSS_V4": CVSS4}
        entries = data.get("severity", [])
        affected = data.get("affected", [])
        if not isinstance(entries, list) or not isinstance(affected, list):
            return None
        entries = list(entries)
        for package in affected:
            if not isinstance(package, dict) or not isinstance(package.get("severity", []), list):
                return None
            entries.extend(package.get("severity", []))
        scores = []
        for entry in entries:
            if not isinstance(entry, dict):
                return None
            kind = entry.get("type")
            parser = parsers.get(kind) if isinstance(kind, str) else None
            vector = entry.get("score")
            if parser is None or not isinstance(vector, str) or len(vector) > 4096:
                return None
            try:
                parsed = parser(vector)  # Validate optional metrics before excluding them.
                if kind == "CVSS_V4":
                    # CVSS4.scores() includes threat/environmental modifiers.
                    # Advisory policy uses the base vector, not a reporter's environment.
                    base_keys = {"AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA"}
                    base_vector = "/".join(
                        part
                        for part in vector.split("/")
                        if part.split(":", 1)[0] in base_keys | {"CVSS"}
                    )
                    parsed = parser(base_vector)
                score = float(parsed.scores()[0])
            except CVSSError:
                return None
            scores.append(CVSSMetrics(score=score, vector=vector))
        return max(scores, key=lambda item: item.score) if scores else None

    def _parse_osv_vuln(self, data: dict) -> CVE:
        """Parse OSV vulnerability to CVE format; reject unidentifiable records."""
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("id"), str)
            or not data["id"].strip()
        ):
            raise CVEDatabaseError("OSV advisory is missing a valid identifier")
        vuln_id = data["id"]

        # Get description
        description = data.get("summary", "") or data.get("details", "")

        cvss = self._parse_osv_cvss(data)
        severity = Severity.from_cvss(cvss.score) if cvss is not None else Severity.UNKNOWN

        # Parse dates
        published = None
        modified = None
        if data.get("published"):
            try:
                published = datetime.fromisoformat(data["published"].replace("Z", "+00:00"))
            except Exception:
                pass
        if data.get("modified"):
            try:
                modified = datetime.fromisoformat(data["modified"].replace("Z", "+00:00"))
            except Exception:
                pass

        # Parse references
        references = [ref.get("url", "") for ref in data.get("references", []) if ref.get("url")]

        # Parse CWEs
        cwe_ids = []
        for cwe in data.get("database_specific", {}).get("cwe_ids", []):
            cwe_ids.append(cwe)

        # Get affected products
        affected_products = []
        for affected in data.get("affected", []):
            pkg = affected.get("package", {})
            name = pkg.get("name", "")
            ecosystem = pkg.get("ecosystem", "")
            if name:
                affected_products.append(f"{ecosystem}:{name}")

        return CVE(
            cve_id=vuln_id,
            description=description,
            severity=severity,
            cvss=cvss,
            published_date=published,
            modified_date=modified,
            references=references,
            cwe_ids=cwe_ids,
            affected_products=affected_products,
        )


class OfflineCVEDatabase(BaseCVEDatabase):
    """Offline CVE database using bundled/cached data.

    For air-gapped environments where API access is not available.
    """

    def __init__(self, data_dir: Path):
        """Initialize with local data directory.

        Args:
            data_dir: Directory containing offline CVE data
        """
        self.data_dir = data_dir
        self._cache = LocalCVECache(data_dir / "cve.db")
        self._load_errors: list[str] = []
        self._advisory_index: dict[str, list[dict]] = {}
        self._load_advisories()

    def _load_advisories(self) -> None:
        """Load advisory data from JSON files."""
        if not self.data_dir.exists():
            return

        for json_file in self.data_dir.glob("advisories/*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    for advisory in data.get("advisories", []):
                        package = advisory.get("package", "")
                        if package not in self._advisory_index:
                            self._advisory_index[package] = []
                        self._advisory_index[package].append(advisory)
            except Exception as e:
                self._load_errors.append(f"Failed to load advisory file {json_file}: {e}")

    async def lookup_cve(self, cve_id: str) -> Optional[CVE]:
        """Look up CVE from local cache."""
        return self._cache.get_cve(cve_id, max_age_hours=24 * 365)  # 1 year cache

    async def search_by_package(
        self,
        package_name: str,
        ecosystem: str,
        version: Optional[str] = None,
    ) -> list[CVE]:
        """Search local advisory index."""
        if self._load_errors:
            raise CVEDatabaseError("; ".join(self._load_errors))
        cached = self._cache.get_package_cves(package_name, ecosystem, version=version)
        if cached is not None:
            advisory_ids = [item[0] for item in cached]
        else:
            key = f"{ecosystem}:{package_name}"
            advisories = self._advisory_index.get(key)
            if not advisories:
                raise CVEDatabaseError(f"No offline advisory coverage for {key}@{version}")
            # Bundled entries lack a complete version-query snapshot: report all
            # indexed advisories conservatively instead of guessing affected ranges.
            advisory_ids = [item.get("cve_id", "") for item in advisories]
        cves = []
        for cve_id in advisory_ids:
            cve = await self.lookup_cve(cve_id)
            if cve is None:
                raise CVEDatabaseError(f"Offline advisory unavailable or expired: {cve_id}")
            cves.append(cve)
        return cves


class CachingCVEDatabase(BaseCVEDatabase):
    """CVE database with automatic caching layer.

    Wraps another database and adds caching.
    """

    def __init__(
        self,
        backend: BaseCVEDatabase,
        cache: LocalCVECache,
        cache_hours: int = 24,
    ):
        """Initialize caching wrapper.

        Args:
            backend: Backend database to wrap
            cache: Local cache
            cache_hours: Cache TTL in hours
        """
        self._backend = backend
        self._cache = cache
        self._cache_hours = cache_hours

    async def lookup_cve(self, cve_id: str) -> Optional[CVE]:
        """Look up CVE with caching."""
        # Check cache
        cached = self._cache.get_cve(cve_id, max_age_hours=self._cache_hours)
        if cached:
            return cached

        # Query backend
        cve = await self._backend.lookup_cve(cve_id)
        if cve:
            self._cache.set_cve(cve)

        return cve

    async def search_by_package(
        self,
        package_name: str,
        ecosystem: str,
        version: Optional[str] = None,
    ) -> list[CVE]:
        """Search with caching."""
        cached_cves = self._cache.get_package_cves(
            package_name, ecosystem, max_age_hours=self._cache_hours, version=version
        )
        if cached_cves is not None:
            cves = []
            for cve_id, _, _ in cached_cves:
                cve = self._cache.get_cve(cve_id, max_age_hours=self._cache_hours)
                if cve is None:
                    break  # Refresh the entire query rather than return partial findings.
                cves.append(cve)
            else:
                return cves

        cves = _unique_advisories(
            await self._backend.search_by_package(package_name, ecosystem, version)
        )
        for cve in cves:
            self._cache.set_cve(cve)
        self._cache.set_package_cves(
            package_name, ecosystem, [(cve.cve_id, "", "") for cve in cves], version=version
        )
        return cves


def get_cve_database(
    offline: bool = False,
    cache_dir: Optional[Path] = None,
) -> BaseCVEDatabase:
    """Get a configured CVE database instance.

    Args:
        offline: Whether to use offline mode only
        cache_dir: Directory for caching

    Returns:
        Configured CVE database
    """
    if cache_dir is None:
        try:
            from victor.config.secure_paths import get_victor_dir

            cache_dir = get_victor_dir() / "cve_cache"
        except ImportError:
            cache_dir = Path.home() / ".victor" / "cve_cache"

    cache = LocalCVECache(cache_dir / "cve.db")

    if offline:
        return OfflineCVEDatabase(cache_dir)

    backend = OSVDatabase(cache=cache)
    return CachingCVEDatabase(backend, cache)
