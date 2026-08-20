# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Migration helpers for moving Victor code storage to ProximaDB."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from victor.config.settings import get_project_paths
from victor.core.schema import Tables
from victor.storage.vector_stores._lancedb_compat import get_table_names
from victor.storage.vector_stores.proximadb_multi import ProximaDBMultiModelProvider

logger = logging.getLogger(__name__)


@dataclass
class MigrationSummary:
    """Summary of a SQLite + LanceDB to ProximaDB migration run."""

    graph_nodes_migrated: int = 0
    graph_edges_migrated: int = 0
    vector_records_migrated: int = 0
    document_records_backfilled: int = 0
    metric_records_backfilled: int = 0
    files_backfilled: int = 0
    files_missing: int = 0
    files_scanned: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "graph_nodes_migrated": self.graph_nodes_migrated,
            "graph_edges_migrated": self.graph_edges_migrated,
            "vector_records_migrated": self.vector_records_migrated,
            "document_records_backfilled": self.document_records_backfilled,
            "metric_records_backfilled": self.metric_records_backfilled,
            "files_backfilled": self.files_backfilled,
            "files_missing": self.files_missing,
            "files_scanned": self.files_scanned,
        }


class SqliteLanceDBMigration:
    """One-shot migration from Victor's SQLite + LanceDB layout to ProximaDB."""

    def __init__(
        self,
        provider: ProximaDBMultiModelProvider,
        repo_root: Path | str,
        graph_db_path: Optional[Path | str] = None,
        lancedb_dir: Optional[Path | str] = None,
        lancedb_table: Optional[str] = None,
        symbol_store_db_path: Optional[Path | str] = None,
    ) -> None:
        self.provider = provider
        self.repo_root = Path(repo_root).expanduser().resolve()

        project_paths = get_project_paths(self.repo_root)
        self.graph_db_path = (
            Path(graph_db_path).expanduser()
            if graph_db_path
            else (project_paths.project_victor_dir / "project.db")
        )
        self.lancedb_dir = (
            Path(lancedb_dir).expanduser() if lancedb_dir else project_paths.embeddings_dir
        )
        self.lancedb_table = lancedb_table or "embeddings"
        self.symbol_store_db_path = (
            Path(symbol_store_db_path).expanduser()
            if symbol_store_db_path
            else project_paths.project_db
        )

    async def migrate(self, clear_target: bool = False) -> Dict[str, int]:
        """Execute the migration and return summary counts."""
        await self.provider.initialize()
        if clear_target:
            await self.provider.clear_index()

        summary = MigrationSummary()
        file_languages: Dict[str, str] = {}

        graph_counts, graph_files = self._migrate_graph()
        summary.graph_nodes_migrated = graph_counts[0]
        summary.graph_edges_migrated = graph_counts[1]
        file_languages.update(graph_files)

        vector_count, vector_files = self._migrate_vectors()
        summary.vector_records_migrated = vector_count
        file_languages.update({path: lang for path, lang in vector_files.items() if lang})

        symbol_files = self._load_symbol_store_files()
        file_languages.update({path: lang for path, lang in symbol_files.items() if lang})

        backfill_counts = self._backfill_documents_and_metrics(file_languages)
        summary.files_scanned = backfill_counts[0]
        summary.files_backfilled = backfill_counts[1]
        summary.files_missing = backfill_counts[2]
        summary.document_records_backfilled = backfill_counts[3]
        summary.metric_records_backfilled = backfill_counts[4]
        return summary.to_dict()

    def _migrate_graph(self) -> Tuple[Tuple[int, int], Dict[str, str]]:
        if not self.graph_db_path.exists():
            logger.info(
                "Graph database not found at %s; skipping graph migration",
                self.graph_db_path,
            )
            return (0, 0), {}

        conn = sqlite3.connect(str(self.graph_db_path))
        conn.row_factory = sqlite3.Row
        try:
            if not self._table_exists(conn, Tables.GRAPH_NODE):
                return (0, 0), {}

            node_count = 0
            edge_count = 0
            file_languages: Dict[str, str] = {}

            node_rows = conn.execute(f"""
                SELECT node_id, type, name, file, line, end_line, lang, signature,
                       docstring, parent_id, embedding_ref, metadata
                FROM {Tables.GRAPH_NODE}
                """).fetchall()
            for row in node_rows:
                metadata = self._parse_json_dict(row["metadata"])
                properties = {
                    "legacy_node_id": row["node_id"],
                    "name": row["name"],
                    "node_type": row["type"],
                    "file_path": row["file"],
                    "line_start": row["line"],
                    "line_end": row["end_line"],
                    "language": row["lang"],
                    "signature": row["signature"],
                    "docstring": row["docstring"],
                    "parent_id": row["parent_id"],
                    "embedding_ref": row["embedding_ref"],
                    "migration_source": "sqlite_graph",
                    **metadata,
                }
                self.provider._create_graph_node(
                    node_id=row["node_id"],
                    labels=self._labels_for_node_type(row["type"]),
                    properties=properties,
                )
                if row["file"] and row["lang"]:
                    file_languages[row["file"]] = row["lang"]
                node_count += 1

            if self._table_exists(conn, Tables.GRAPH_EDGE):
                edge_rows = conn.execute(
                    f"SELECT src, dst, type, weight, metadata FROM {Tables.GRAPH_EDGE}"
                ).fetchall()
                for row in edge_rows:
                    metadata = self._parse_json_dict(row["metadata"])
                    edge_id = self._legacy_edge_id(
                        edge_type=row["type"],
                        src=row["src"],
                        dst=row["dst"],
                        metadata=metadata,
                    )
                    self.provider._create_graph_edge(
                        edge_id=edge_id,
                        from_node_id=row["src"],
                        to_node_id=row["dst"],
                        edge_type=row["type"],
                        properties={
                            **metadata,
                            "weight": row["weight"],
                            "migration_source": "sqlite_graph",
                        },
                    )
                    edge_count += 1

            return (node_count, edge_count), file_languages
        finally:
            conn.close()

    def _migrate_vectors(self) -> Tuple[int, Dict[str, str]]:
        if not self.lancedb_dir.exists():
            logger.info(
                "LanceDB directory not found at %s; skipping vector migration",
                self.lancedb_dir,
            )
            return 0, {}

        try:
            import lancedb
        except ImportError as exc:
            raise ImportError("lancedb is required for vector migration") from exc

        db = lancedb.connect(str(self.lancedb_dir))
        table_name = self.lancedb_table
        available = get_table_names(db)
        if table_name not in available:
            if len(available) == 1:
                table_name = available[0]
            else:
                logger.info(
                    "LanceDB table %s not found in %s; available tables: %s",
                    table_name,
                    self.lancedb_dir,
                    available,
                )
                return 0, {}

        table = db.open_table(table_name)
        try:
            row_count = table.count_rows()
        except Exception:
            row_count = 0

        if row_count == 0:
            return 0, {}

        rows = table.head(row_count).to_list()
        file_languages: Dict[str, str] = {}
        migrated = 0
        batch: List[Any] = []
        batch_size = 500

        for row in rows:
            metadata = self._extract_vector_metadata(row)
            source = str(row.get("content", ""))
            record_id = str(row.get("id"))
            vector = row.get("vector") or self.provider._zero_vector()

            if metadata.get("file_path") and metadata.get("language"):
                file_languages[str(metadata["file_path"])] = str(metadata["language"])

            batch.append(
                self.provider._vector_record(
                    record_id=record_id,
                    vector=list(vector),
                    source=source,
                    metadata={**metadata, "migration_source": "lancedb"},
                )
            )

            if len(batch) >= batch_size:
                self.provider._client.insert_vectors(
                    self.provider._vector_collection, records=batch
                )
                migrated += len(batch)
                batch = []

        if batch:
            self.provider._client.insert_vectors(self.provider._vector_collection, records=batch)
            migrated += len(batch)

        return migrated, file_languages

    def _backfill_documents_and_metrics(
        self, file_languages: Dict[str, str]
    ) -> Tuple[int, int, int, int, int]:
        scanned = 0
        backfilled = 0
        missing = 0
        document_records = 0
        metric_records = 0

        for relative_path, language in sorted(file_languages.items()):
            scanned += 1
            absolute_path = self.repo_root / relative_path
            if not absolute_path.exists() or not absolute_path.is_file():
                missing += 1
                continue

            try:
                content = absolute_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                missing += 1
                continue

            resolved_language = language or self.provider._resolve_language(
                relative_path, content, None
            )
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            indexed_at = datetime.now(timezone.utc)
            base_metadata = {
                "workspace": self.provider._workspace,
                "file_path": relative_path,
                "language": resolved_language,
                "file_hash": file_hash,
                "indexed_at": indexed_at.isoformat(),
                "migration_source": "sqlite_lancedb",
            }

            if self.provider._document_enabled:
                self.provider._store_document_snapshot(
                    file_path=relative_path,
                    content=content,
                    language=resolved_language,
                    base_metadata=base_metadata,
                )
                document_records += 1

            if self.provider._metrics_enabled:
                snapshot = self.provider._extract_graph_snapshot(
                    relative_path, content, resolved_language
                )
                metric_records += self.provider._store_metrics_snapshot(
                    file_path=relative_path,
                    content=content,
                    language=resolved_language,
                    symbols=snapshot.symbols,
                    recorded_at=indexed_at,
                    base_metadata=base_metadata,
                )

            backfilled += 1

        return scanned, backfilled, missing, document_records, metric_records

    def _load_symbol_store_files(self) -> Dict[str, str]:
        if not self.symbol_store_db_path.exists():
            return {}

        conn = sqlite3.connect(str(self.symbol_store_db_path))
        conn.row_factory = sqlite3.Row
        try:
            if not self._table_exists(conn, "files"):
                return {}
            rows = conn.execute("SELECT path, language FROM files").fetchall()
            return {str(row["path"]): str(row["language"]) for row in rows if row["path"]}
        except Exception:
            logger.debug("Failed to read file metadata from %s", self.symbol_store_db_path)
            return {}
        finally:
            conn.close()

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _parse_json_dict(self, raw_value: Any) -> Dict[str, Any]:
        if not raw_value:
            return {}
        if isinstance(raw_value, dict):
            return raw_value
        try:
            parsed = json.loads(raw_value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _labels_for_node_type(self, node_type: Optional[str]) -> List[str]:
        normalized = (node_type or "symbol").lower()
        if normalized in {"file", "module", "package"}:
            return ["Module"]
        if normalized in {"class", "interface", "struct", "enum", "trait"}:
            return ["Class"]
        if normalized in {"function", "method", "constructor"}:
            return ["Function"]
        return ["Symbol", normalized.title()]

    def _extract_vector_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"id", "vector", "content", "_distance"}
        }
        if "metadata" in metadata and isinstance(metadata["metadata"], dict):
            nested = metadata.pop("metadata")
            metadata.update(nested)
        return metadata

    def _legacy_edge_id(
        self,
        edge_type: str,
        src: str,
        dst: str,
        metadata: Dict[str, Any],
    ) -> str:
        line_hint = metadata.get("line_number") or metadata.get("line")
        material = f"legacy|{edge_type}|{src}|{dst}|{line_hint or 0}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return f"{self.provider._workspace}:legacy_edge:{digest}"


async def migrate_sqlite_lancedb_to_proximadb(
    provider: ProximaDBMultiModelProvider,
    repo_root: Path | str,
    graph_db_path: Optional[Path | str] = None,
    lancedb_dir: Optional[Path | str] = None,
    lancedb_table: Optional[str] = None,
    symbol_store_db_path: Optional[Path | str] = None,
    clear_target: bool = False,
) -> Dict[str, int]:
    """Convenience wrapper around :class:`SqliteLanceDBMigration`."""
    migration = SqliteLanceDBMigration(
        provider=provider,
        repo_root=repo_root,
        graph_db_path=graph_db_path,
        lancedb_dir=lancedb_dir,
        lancedb_table=lancedb_table,
        symbol_store_db_path=symbol_store_db_path,
    )
    return await migration.migrate(clear_target=clear_target)


__all__ = [
    "MigrationSummary",
    "SqliteLanceDBMigration",
    "migrate_sqlite_lancedb_to_proximadb",
]
