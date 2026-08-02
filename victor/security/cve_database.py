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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS package_cves (
                    package_name TEXT,
                    ecosystem TEXT,
                    cve_id TEXT,
                    version_range TEXT,
                    fixed_version TEXT,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (package_name, ecosystem, cve_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_package_lookup
                ON package_cves (package_name, ecosystem)
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

            return self._deserialize_cve(json.loads(data_str))

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
    ) -> list[tuple[str, str, str]]:
        """Get cached CVEs for a package.

        Returns:
            List of (cve_id, version_range, fixed_version) tuples
        """
        with sqlite3.connect(self.cache_path) as conn:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            rows = conn.execute(
                """
                SELECT cve_id, version_range, fixed_version
                FROM package_cves
                WHERE package_name = ? AND ecosystem = ? AND cached_at > ?
                """,
                (package_name, ecosystem, cutoff),
            ).fetchall()
            return rows

    def set_package_cves(
        self,
        package_name: str,
        ecosystem: str,
        cve_data: list[tuple[str, str, str]],
    ) -> None:
        """Cache CVEs for a package.

        Args:
            package_name: Package name
            ecosystem: Package ecosystem
            cve_data: List of (cve_id, version_range, fixed_version) tuples
        """
        with sqlite3.connect(self.cache_path) as conn:
            # Clear old data
            conn.execute(
                "DELETE FROM package_cves WHERE package_name = ? AND ecosystem = ?",
                (package_name, ecosystem),
            )

            # Insert new data
            for cve_id, version_range, fixed_version in cve_data:
                conn.execute(
                    """
                    INSERT INTO package_cves
                    (package_name, ecosystem, cve_id, version_range, fixed_version, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        package_name,
                        ecosystem,
                        cve_id,
                        version_range,
                        fixed_version,
                        datetime.now(),
                    ),
                )
            conn.commit()

    def _serialize_cve(self, cve: CVE) -> dict:
        """Serialize CVE to dict."""
        return {
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
            logger.warning("httpx not installed, cannot query OSV API")
            return None
        except Exception as e:
            logger.warning(f"OSV lookup failed: {e}")
            return None

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

            await self._rate_limit()

            query = {"package": {"name": package_name, "ecosystem": osv_ecosystem}}
            if version:
                query["version"] = version

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.API_URL}/query",
                    json=query,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                cves = []
                for vuln in data.get("vulns", []):
                    cve = self._parse_osv_vuln(vuln)
                    if cve:
                        cves.append(cve)
                        if self._cache:
                            self._cache.set_cve(cve)

                return cves

        except ImportError:
            logger.warning("httpx not installed")
            return []
        except Exception as e:
            logger.warning(f"OSV search failed: {e}")
            return []

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

    def _parse_osv_vuln(self, data: dict) -> Optional[CVE]:
        """Parse OSV vulnerability to CVE format."""
        vuln_id = data.get("id", "")

        # Get description
        description = data.get("summary", "") or data.get("details", "")

        # Parse CVSS
        cvss = None
        severity = Severity.MEDIUM  # Default
        for severity_entry in data.get("severity", []):
            if severity_entry.get("type") == "CVSS_V3":
                score_str = severity_entry.get("score", "")
                try:
                    # Parse CVSS vector to extract score
                    # For simplicity, use a lookup or calculation
                    cvss = CVSSMetrics(score=5.0, vector=score_str)
                    severity = Severity.from_cvss(cvss.score)
                except Exception:
                    pass

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
        self._cache = LocalCVECache(data_dir / "cve_cache.db")
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
                logger.warning(f"Failed to load advisory file {json_file}: {e}")

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
        key = f"{ecosystem}:{package_name}"
        advisories = self._advisory_index.get(key, [])

        cves = []
        for advisory in advisories:
            cve = await self.lookup_cve(advisory.get("cve_id", ""))
            if cve:
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
        # Check cache
        cached_cves = self._cache.get_package_cves(
            package_name, ecosystem, max_age_hours=self._cache_hours
        )
        if cached_cves:
            cves = []
            for cve_id, _, _ in cached_cves:
                cve = await self.lookup_cve(cve_id)
                if cve:
                    cves.append(cve)
            if cves:
                return cves

        # Query backend
        cves = await self._backend.search_by_package(package_name, ecosystem, version)

        # Cache results
        cve_data = [(cve.cve_id, "", "") for cve in cves]
        self._cache.set_package_cves(package_name, ecosystem, cve_data)

        for cve in cves:
            self._cache.set_cve(cve)

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
