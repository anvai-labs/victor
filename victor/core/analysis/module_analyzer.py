"""Module-level graph analysis: coupling, cohesion, PageRank, hotspots.

Computes and persists module-level metrics using the graph_module_metric
table in .victor/project.db.
"""

from __future__ import annotations

import logging
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModuleMetrics:
    """Metrics for a single module (file or package)."""

    module_path: str
    pagerank_score: float = 0.0
    betweenness: float = 0.0
    afferent_coupling: int = 0  # Ca: incoming deps
    efferent_coupling: int = 0  # Ce: outgoing deps
    instability: float = 0.0  # I = Ce / (Ca + Ce)
    abstractness: float = 0.0  # A = abstract / total types
    distance_main_seq: float = 0.0  # D = |A + I - 1|
    cohesion_lcom4: float = 0.0
    hotspot_score: float = 0.0
    symbol_count: int = 0
    change_frequency: int = 0
    tdd_priority: float = 0.0


class ModuleAnalyzer:
    """Compute and persist module-level metrics from the code graph."""

    #: Max history rows retained per module. Every persist() appends one history row per
    #: module; without a cap graph_module_metric_history grows unbounded (observed at 360k+
    #: rows / hundreds of MB on a long-lived project DB). The latest N snapshots are enough
    #: for trend/hotspot analysis.
    _HISTORY_RETENTION_PER_MODULE = 50

    #: Pure-Python Brandes betweenness is O(V·E) with per-source bookkeeping;
    #: past this module count it runs for hours (observed: a 59k-module
    #: vendored tree burned 99 CPU-minutes before being killed). The native
    #: backend handles large graphs; without it betweenness is skipped so the
    #: rest of the metric refresh still completes.
    _BETWEENNESS_PYTHON_MAX_MODULES = 5000

    def __init__(self, db=None, project_path: Optional[Path] = None):
        self._db = db
        self._project_path = project_path or Path.cwd()
        # Parsed coverage data (path -> fraction), built lazily once per
        # analyzer instead of re-reading coverage.json for every module.
        self._coverage_index: Optional[dict[str, float]] = None

    def _get_db(self):
        """Get or create database connection."""
        if self._db is not None:
            return self._db
        from victor.core.database import ProjectDatabaseManager

        return ProjectDatabaseManager(self._project_path)

    def compute_all(self) -> list[ModuleMetrics]:
        """Compute all module metrics from the graph."""
        modules, adj_out, adj_in = self._load_module_graph()
        if not modules:
            return []

        pageranks = self._compute_pagerank(modules, adj_out)
        betweenness = self._compute_betweenness(modules, adj_out)
        module_to_nodes = self._load_module_to_nodes()

        results = []
        # Normalize change frequencies for hotspot calc
        change_freqs = self._get_change_frequencies(modules)
        max_change = max(change_freqs.values()) if change_freqs else 1
        if max_change == 0:
            max_change = 1

        for mod in modules:
            ca, ce, instability = self._compute_coupling(mod, adj_out, adj_in)
            abstractness = self._compute_abstractness(mod, module_to_nodes)
            distance = abs(abstractness + instability - 1.0)
            cohesion = self._compute_cohesion(mod, module_to_nodes)
            change_freq = change_freqs.get(mod, 0)

            m = ModuleMetrics(
                module_path=mod,
                pagerank_score=pageranks.get(mod, 0.0),
                betweenness=betweenness.get(mod, 0.0),
                afferent_coupling=ca,
                efferent_coupling=ce,
                instability=instability,
                abstractness=abstractness,
                distance_main_seq=distance,
                cohesion_lcom4=cohesion,
                symbol_count=len(module_to_nodes.get(mod, [])),
                change_frequency=change_freq,
            )
            m.hotspot_score = self._compute_hotspot(m, max_change)
            m.tdd_priority = self._compute_tdd_priority(m, max_change)
            results.append(m)

        return results

    def persist(self, metrics: list[ModuleMetrics]) -> None:
        """UPSERT metrics into graph_module_metric table."""
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        for m in metrics:
            conn.execute(
                """INSERT INTO graph_module_metric
                   (module_path, pagerank_score, betweenness, afferent_coupling,
                    efferent_coupling, instability, abstractness, distance_main_seq,
                    cohesion_lcom4, hotspot_score, symbol_count, change_frequency,
                    tdd_priority, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(module_path) DO UPDATE SET
                    pagerank_score=excluded.pagerank_score,
                    betweenness=excluded.betweenness,
                    afferent_coupling=excluded.afferent_coupling,
                    efferent_coupling=excluded.efferent_coupling,
                    instability=excluded.instability,
                    abstractness=excluded.abstractness,
                    distance_main_seq=excluded.distance_main_seq,
                    cohesion_lcom4=excluded.cohesion_lcom4,
                    hotspot_score=excluded.hotspot_score,
                    symbol_count=excluded.symbol_count,
                    change_frequency=excluded.change_frequency,
                    tdd_priority=excluded.tdd_priority,
                    computed_at=datetime('now')
                """,
                (
                    m.module_path,
                    m.pagerank_score,
                    m.betweenness,
                    m.afferent_coupling,
                    m.efferent_coupling,
                    m.instability,
                    m.abstractness,
                    m.distance_main_seq,
                    m.cohesion_lcom4,
                    m.hotspot_score,
                    m.symbol_count,
                    m.change_frequency,
                    m.tdd_priority,
                ),
            )
            # Also insert history
            conn.execute(
                """INSERT INTO graph_module_metric_history
                   (module_path, hotspot_score, tdd_priority)
                   VALUES (?, ?, ?)""",
                (m.module_path, m.hotspot_score, m.tdd_priority),
            )
            # Bound history growth: keep only the most recent N snapshots per module.
            conn.execute(
                """DELETE FROM graph_module_metric_history
                   WHERE module_path = ?
                     AND id NOT IN (
                         SELECT id FROM graph_module_metric_history
                         WHERE module_path = ?
                         ORDER BY id DESC
                         LIMIT ?
                     )""",
                (m.module_path, m.module_path, self._HISTORY_RETENTION_PER_MODULE),
            )
        conn.commit()

    def get_cached(self, order_by: str = "hotspot_score", limit: int = 50) -> list[dict]:
        """Get cached metrics from the database."""
        allowed_cols = {
            "hotspot_score",
            "tdd_priority",
            "pagerank_score",
            "instability",
            "betweenness",
        }
        if order_by not in allowed_cols:
            order_by = "hotspot_score"
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        cursor = conn.execute(
            f"SELECT * FROM graph_module_metric ORDER BY {order_by} DESC LIMIT ?",
            (limit,),
        )
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def has_cached_metrics(self) -> bool:
        """Check if cached metrics exist."""
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM graph_module_metric")
            return cursor.fetchone()[0] > 0
        except Exception:
            return False

    # === Internal methods ===

    def _load_module_graph(
        self,
    ) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
        """Load module-level dependency adjacency from graph_edge/graph_node.

        Martin coupling (Ca/Ce), PageRank, and betweenness are module-level
        *dependency* metrics, so the adjacency uses IMPORTS edges only.
        CALLS (and CFG/CDG/DDG) edges are deliberately excluded: cross-file
        CALLS are resolved by leaf name with heuristic fanout and were
        observed inflating Ca 10-18x over real use-statement fan-in.
        Projects with no IMPORTS edges at all (language without an import
        resolver yet) fall back to the legacy all-edges adjacency so their
        metrics don't vanish.

        The module universe is every indexed file (graph_node.file), not
        just edge endpoints — modules with no import relationships get
        honest zero-coupling rows, which also overwrites stale inflated
        values persisted by earlier runs.
        """
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        edge_query = """SELECT DISTINCT n1.file, n2.file
                   FROM graph_edge e
                   JOIN graph_node n1 ON e.src = n1.node_id
                   JOIN graph_node n2 ON e.dst = n2.node_id
                   WHERE n1.file IS NOT NULL AND n2.file IS NOT NULL
                     AND n1.file != n2.file"""
        try:
            module_rows = conn.execute(
                "SELECT DISTINCT file FROM graph_node WHERE file IS NOT NULL"
            ).fetchall()
            rows = conn.execute(edge_query + " AND e.type = 'IMPORTS'").fetchall()
            if not rows:
                logger.debug(
                    "No IMPORTS edges in graph — falling back to all-edge "
                    "adjacency for module metrics"
                )
                rows = conn.execute(edge_query).fetchall()
        except Exception:
            return set(), {}, {}

        modules: set[str] = {row[0] for row in module_rows}
        adj_out: dict[str, set[str]] = {}
        adj_in: dict[str, set[str]] = {}
        for src_file, dst_file in rows:
            modules.add(src_file)
            modules.add(dst_file)
            adj_out.setdefault(src_file, set()).add(dst_file)
            adj_in.setdefault(dst_file, set()).add(src_file)
        return modules, adj_out, adj_in

    def _load_module_to_nodes(self) -> dict[str, list[dict]]:
        """Load mapping of modules to their node info."""
        db = self._get_db()
        conn = db.connection if hasattr(db, "connection") else db._get_raw_connection()
        try:
            rows = conn.execute(
                "SELECT node_id, type, name, file FROM graph_node WHERE file IS NOT NULL"
            ).fetchall()
        except Exception:
            return {}
        result: dict[str, list[dict]] = {}
        for node_id, ntype, name, fpath in rows:
            result.setdefault(fpath, []).append({"node_id": node_id, "type": ntype, "name": name})
        return result

    def _compute_coupling(self, module: str, adj_out: dict, adj_in: dict) -> tuple[int, int, float]:
        """Compute afferent (Ca) and efferent (Ce) coupling + instability."""
        ca = len(adj_in.get(module, set()))
        ce = len(adj_out.get(module, set()))
        total = ca + ce
        instability = ce / total if total > 0 else 0.0
        return ca, ce, instability

    def _compute_abstractness(self, module: str, module_to_nodes: dict) -> float:
        """Compute abstractness: ratio of abstract types to total types."""
        nodes = module_to_nodes.get(module, [])
        if not nodes:
            return 0.0
        type_nodes = [n for n in nodes if n["type"] in ("class", "interface", "protocol")]
        if not type_nodes:
            return 0.0
        # Heuristic: classes with "Base", "Abstract", "Protocol", "Interface" in name
        abstract_count = sum(
            1
            for n in type_nodes
            if any(
                kw in n["name"]
                for kw in ("Base", "Abstract", "Protocol", "Interface", "ABC", "Mixin")
            )
        )
        return abstract_count / len(type_nodes)

    @staticmethod
    def _to_list_adjacency(modules: set[str], adj: dict[str, set[str]]) -> dict[str, list[str]]:
        """Set adjacency → list adjacency covering every module (loader interface)."""
        adj_dict = {k: list(v) for k, v in adj.items()}
        for m in modules:
            if m not in adj_dict:
                adj_dict[m] = []
        return adj_dict

    def _compute_pagerank(
        self,
        modules: set[str],
        adj: dict[str, set[str]],
        damping: float = 0.85,
        iterations: int = 100,
    ) -> dict[str, float]:
        """Compute PageRank via the graph-algo loader (Rust or pure Python).

        The loader import always succeeds — it falls back to
        victor.native.python.graph_algo internally — so no inline
        re-implementation is kept here. The Python PageRank is edge-based
        power iteration with a convergence early-exit, fine at any module
        count this analyzer sees.
        """
        if not modules:
            return {}
        from victor.native.graph_algo_loader import pagerank

        return pagerank(self._to_list_adjacency(modules, adj), damping, iterations)

    def _compute_betweenness(self, modules: set[str], adj: dict[str, set[str]]) -> dict[str, float]:
        """Compute betweenness centrality via the graph-algo loader.

        Brandes is O(V·E) regardless of backend; the pure-Python
        implementation additionally allocates per-source bookkeeping, so
        above _BETWEENNESS_PYTHON_MAX_MODULES it is skipped (zeroes) rather
        than stalling the whole metric refresh.
        """
        if not modules:
            return {}
        from victor.native.graph_algo_loader import (
            GRAPH_ALGO_BACKEND,
            betweenness_centrality,
        )

        if GRAPH_ALGO_BACKEND != "rust" and len(modules) > self._BETWEENNESS_PYTHON_MAX_MODULES:
            logger.info(
                "Skipping betweenness centrality for %d modules (pure-Python cap %d; "
                "build the victor_native extension to compute it on large graphs)",
                len(modules),
                self._BETWEENNESS_PYTHON_MAX_MODULES,
            )
            return dict.fromkeys(modules, 0.0)
        return betweenness_centrality(self._to_list_adjacency(modules, adj))

    def _compute_cohesion(self, module: str, module_to_nodes: dict) -> float:
        """Compute LCOM4 cohesion (0=low cohesion, 1=high cohesion)."""
        nodes = module_to_nodes.get(module, [])
        if len(nodes) <= 1:
            return 1.0
        # Simplified: connected component count over total as proxy
        # Lower component count = higher cohesion
        # For now, use 1/symbol_count as a simple heuristic
        return min(1.0, 1.0 / max(1, len(nodes) - 1))

    def _get_change_frequencies(self, modules: set[str]) -> dict[str, int]:
        """Git change frequency per module (last 90 days), in ONE git pass.

        The previous implementation spawned one ``git log -- <path>``
        subprocess per module — ~20-50ms each, so a 4k-module repo paid
        minutes and a 59k-module tree effectively never finished. A single
        ``--name-only`` log emits every touched path once per commit;
        counting those gives the same per-file commit counts.
        """
        counts: Counter[str] = Counter()
        try:
            result = subprocess.run(
                ["git", "log", "--since=90 days ago", "--name-only", "--pretty=format:"],
                capture_output=True,
                text=True,
                cwd=str(self._project_path),
                timeout=120,
            )
            if result.returncode == 0:
                counts.update(line.strip() for line in result.stdout.splitlines() if line.strip())
        except Exception:
            pass
        return {mod: counts.get(mod, 0) for mod in modules}

    def _compute_hotspot(self, m: ModuleMetrics, max_change: int) -> float:
        """Compute hotspot score: 0.4*pagerank + 0.3*coupling_ratio + 0.3*change_freq_norm."""
        coupling_ratio = m.instability  # Ce / (Ca + Ce), already [0, 1]
        change_norm = m.change_frequency / max_change if max_change > 0 else 0.0
        return 0.4 * m.pagerank_score + 0.3 * coupling_ratio + 0.3 * change_norm

    def _compute_tdd_priority(self, m: ModuleMetrics, max_change: int) -> float:
        """Compute TDD priority score.

        Formula: 0.25*pagerank + 0.20*coupling + 0.20*change_freq
                 + 0.20*(1-coverage) + 0.15*(1-cohesion)
        """
        change_norm = m.change_frequency / max_change if max_change > 0 else 0.0
        coverage = self._get_test_coverage(m.module_path)
        return (
            0.25 * m.pagerank_score
            + 0.20 * m.instability
            + 0.20 * change_norm
            + 0.20 * (1.0 - coverage)
            + 0.15 * (1.0 - m.cohesion_lcom4)
        )

    def _get_test_coverage(self, module: str) -> float:
        """Look up test coverage for a module from the parsed coverage index.

        The index is built once per analyzer — previously coverage.json was
        re-read and re-parsed for every module (O(modules × file size)).
        """
        if self._coverage_index is None:
            self._coverage_index = self._load_coverage_index()
        for path, covered in self._coverage_index.items():
            if module in path:
                return covered
        return 0.0

    def _load_coverage_index(self) -> dict[str, float]:
        """Parse coverage.json (preferred) or htmlcov/status.json once."""
        import json

        coverage_json = self._project_path / "coverage.json"
        if coverage_json.exists():
            try:
                data = json.loads(coverage_json.read_text())
                return {
                    path: info.get("summary", {}).get("percent_covered", 0.0) / 100.0
                    for path, info in data.get("files", {}).items()
                }
            except Exception:
                pass

        status_json = self._project_path / "htmlcov" / "status.json"
        if status_json.exists():
            try:
                data = json.loads(status_json.read_text())
                return {
                    path: info.get("index", {}).get("pc_covered", 0.0) / 100.0
                    for path, info in data.get("files", {}).items()
                }
            except Exception:
                pass

        return {}
