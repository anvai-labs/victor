"""Offline function-body vectors with a separate input profile and derived cache.

Summary vectors are never read or overwritten. Exact cosine retrieval is tiled
and bounded by an explicit function limit; scores are review evidence, not proof.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import re
from pathlib import Path
import sqlite3
import stat
import uuid
from typing import Any

from victor.core.graph_rag.structural_clones import StructuralReport
from victor.core.graph_rag.body_provenance import (
    ProducerProfile,
    callable_code_fingerprint,
    library_versions,
    producer_profile,
)


@dataclass(frozen=True)
class BodyOptions:
    model_id: str = "BAAI/bge-small-en-v1.5"
    revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    dimension: int = 384
    max_tokens: int = 512
    overlap: int = 64
    max_functions: int = 5000
    max_windows: int = 20_000
    comparison_scope: str = "all"
    text_contract: str = "function-source-token-windows-v1"
    pooling: str = "mean-unit-windows-then-unit-v1"

    def __post_init__(self) -> None:
        try:
            from victor_codegraph.embeddings import EmbeddingProfile
        except ImportError as exc:
            raise ImportError("Body analysis requires the graph and embeddings extras") from exc

        EmbeddingProfile(self.model_id, self.revision, self.dimension, max_tokens=self.max_tokens)
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.model_id) or any(
            p in {".", ".."} for p in self.model_id.split("/")
        ):
            raise ValueError(
                "Body models require a Hub model ID and immutable revision, not a mutable local path"
            )
        if self.max_tokens < 8 or not 0 <= self.overlap < self.max_tokens - 4:
            raise ValueError("Body windows require a positive token budget and smaller overlap")
        if not 1 <= self.max_functions <= 50_000 or not 1 <= self.max_windows <= 100_000:
            raise ValueError("Invalid body analysis budget")
        if self.comparison_scope not in {"all", "same", "cross"}:
            raise ValueError("Invalid body language scope")
        if (
            self.text_contract != "function-source-token-windows-v1"
            or self.pooling != "mean-unit-windows-then-unit-v1"
        ):
            raise ValueError("Unsupported body input/pooling contract")

    @property
    def identity(self) -> str:
        # Retrieval budgets do not change an individual vector's compatibility.
        profile = asdict(self)
        profile.pop("max_functions")
        profile.pop("max_windows")
        profile.pop("comparison_scope")
        return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


class LocalBodyEncoder:
    """One pinned local model per analysis; failed loads are not replayed.

    Create a fresh encoder after repairing a profile or local model installation.
    Warm vector-cache reads do not call this loader.
    """

    def __init__(
        self, options: BodyOptions, *, model_factory: Any = None, producer_id: str | None = None
    ) -> None:
        self._options = options
        self._model_factory = model_factory
        self._model: Any = None
        self._load_failure: tuple[type[Exception], str] | None = None
        self._load_attempts = 0
        self._producer_id = producer_id
        self._producer_nonce = uuid.uuid4().hex

    @property
    def options(self) -> BodyOptions:
        return self._options

    @property
    def model_factory(self) -> Any:
        return self._model_factory

    @property
    def model(self) -> Any:
        return self._model

    @property
    def cache_producer(self) -> ProducerProfile:
        """Describe actual producer generation without constructing model weights."""
        from victor_codegraph.embeddings import validate_vectors

        nonce = (
            self._producer_nonce
            if (
                self.model_factory is not None
                or type(self) is not LocalBodyEncoder
                or any(name in self.__dict__ for name in ("_load", "windows", "encode"))
            )
            and self._producer_id is None
            else None
        )
        components = (
            self._load,
            self.windows,
            self.encode,
            _pool_body_windows,
            analyze_bodies,
            _BodyCache.get,
            _BodyCache.put,
            validate_vectors,
            library_versions,
            producer_profile,
        )
        if any(hasattr(method, "__wrapped__") for method in components):
            # Wrappers may carry runtime state absent from their source text.
            nonce = self._producer_nonce
        try:
            source = "\n".join(
                inspect.getsource(method) + callable_code_fingerprint(method)
                for method in components
            )
        except (AttributeError, OSError, TypeError, ValueError):
            source = "producer-source-unavailable"
            # A declaration covers the custom model, not unknown framework code.
            nonce = self._producer_nonce
        return producer_profile(
            self.options.identity,
            component_hash=hashlib.sha256(source.encode()).hexdigest(),
            custom_id=self._producer_id,
            instance_nonce=nonce,
        )

    @property
    def load_attempts(self) -> int:
        return self._load_attempts

    @property
    def load_failed(self) -> bool:
        return self._load_failure is not None

    def _load(self) -> Any:
        if self._load_failure is not None:
            error_type, message = self._load_failure
            raise error_type(message)
        if self.model is None:
            self._load_attempts += 1
            try:
                factory = self.model_factory
                if factory is None:
                    from sentence_transformers import SentenceTransformer

                    factory = SentenceTransformer
                model = factory(
                    self.options.model_id,
                    revision=self.options.revision,
                    local_files_only=True,
                    trust_remote_code=False,
                    device="cpu",
                )
                dimension = (
                    getattr(model, "get_embedding_dimension", None)
                    or model.get_sentence_embedding_dimension
                )
                if dimension() != self.options.dimension:
                    raise ValueError("Body model dimension does not match its profile")
                if int(model.max_seq_length) < self.options.max_tokens:
                    raise ValueError("Body token budget exceeds this model's context")
                model.max_seq_length = self.options.max_tokens
                self._model = model
            except (AttributeError, TypeError) as exc:
                message = f"Unsupported local body model metadata: {exc}"[:1500]
                self._load_failure = (ValueError, message)
                raise ValueError(message) from exc
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                # Keep only a bounded message and base exception class, never a
                # traceback retaining partially constructed model weights.
                error_type = next(
                    kind
                    for kind in (ImportError, OSError, RuntimeError, ValueError)
                    if isinstance(exc, kind)
                )
                self._load_failure = (error_type, str(exc)[:1500])
                raise
        return self.model

    def windows(self, source: str) -> tuple[list[str], list[tuple[int, int]]]:
        tokenizer = self._load().tokenizer
        encoded = tokenizer(
            source,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        offsets = encoded.get("offset_mapping")
        if not offsets or len(offsets) != len(encoded["input_ids"]):
            raise ValueError("Body analysis requires token-to-source offsets")
        if len(offsets) > 65_536:
            raise ValueError("Function body exceeds the token budget")
        budget = self.options.max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
        if budget <= self.options.overlap:
            raise ValueError("Model special tokens exhaust the body window budget")
        result: list[str] = []
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(offsets):
            end = min(len(offsets), start + budget)
            while end > start:
                text = source[offsets[start][0] : offsets[end - 1][1]]
                # Retokenizing a slice can split a boundary word differently.
                if (
                    len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
                    <= self.options.max_tokens
                ):
                    break
                end -= 1
            if end <= start or len(result) >= 128:
                raise ValueError("Function cannot be fully represented within body window limits")
            result.append(text)
            spans.append((start, end))
            if end == len(offsets):
                break
            start = max(start + 1, end - self.options.overlap)
        return result, spans

    def encode(self, texts: list[str]) -> Any:
        return self._load().encode(
            texts,
            batch_size=32,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt="",
        )


def _pool_body_windows(windows: Any) -> Any:
    """Reference pooling is also part of the producer compatibility contract."""
    import numpy as np

    normalized = windows / np.linalg.norm(windows, axis=1)[:, None]
    mean = normalized.mean(axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or not norm:
        raise ValueError("Body window pooling produced an invalid vector")
    return mean / norm


class _BodyCache:
    def __init__(self, path: Path | None, profile: BodyOptions, producer: ProducerProfile) -> None:
        self.profile = profile
        self.identity = producer.identity
        self.db: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            original = path.lstat() if path.exists() or path.is_symlink() else None
            if original is not None and (
                not stat.S_ISREG(original.st_mode) or original.st_nlink != 1
            ):
                raise ValueError("Body cache must be an exclusive regular file, without links")
            self.db = sqlite3.connect(path, timeout=5)
            try:
                opened = path.lstat()
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or original is not None
                    and (original.st_dev, original.st_ino) != (opened.st_dev, opened.st_ino)
                ):
                    raise ValueError("Body cache file identity changed while opening")
                objects = self.db.execute(
                    "SELECT type,name FROM sqlite_master WHERE name NOT GLOB 'sqlite_*'"
                ).fetchall()
                if any(kind != "table" or name != "body_vectors_v1" for kind, name in objects):
                    raise ValueError("Body cache cannot repurpose an unrelated SQLite database")
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS body_vectors_v1 (profile TEXT, input_hash TEXT, vector BLOB NOT NULL, spans TEXT NOT NULL, checksum TEXT NOT NULL, PRIMARY KEY(profile,input_hash))"
                )
            except (sqlite3.Error, OSError, ValueError):
                self.db.close()
                self.db = None
                raise

    def get(self, digest: str) -> tuple[Any, list[Any]] | None:
        import numpy as np

        if self.db is None:
            return None
        row = self.db.execute(
            "SELECT vector,spans,checksum FROM body_vectors_v1 WHERE profile=? AND input_hash=?",
            (self.identity, digest),
        ).fetchone()
        if row is None:
            return None
        vector, raw_spans, checksum = row
        if (
            not isinstance(vector, bytes)
            or not isinstance(raw_spans, str)
            or not isinstance(checksum, str)
        ):
            raise ValueError("Invalid cached body record types")
        if hashlib.sha256(vector + raw_spans.encode()).hexdigest() != checksum:
            raise ValueError("Corrupt body embedding cache entry")
        values = np.frombuffer(vector, dtype="<f4").copy()
        spans = json.loads(raw_spans)
        if (
            values.shape != (self.profile.dimension,)
            or not np.isfinite(values).all()
            or not np.linalg.norm(values)
        ):
            raise ValueError("Invalid cached body vector")
        if (
            not isinstance(spans, list)
            or not 1 <= len(spans) <= 128
            or any(
                not isinstance(span, list)
                or len(span) != 2
                or any(type(v) is not int for v in span)
                or not 0 <= span[0] < span[1] <= 65_536
                for span in spans
            )
        ):
            raise ValueError("Invalid cached body window manifest")
        return values, spans

    def put(self, digest: str, vector: Any, spans: list[Any]) -> None:
        if self.db is None:
            return
        values = vector.astype("<f4", copy=False).tobytes()
        raw_spans = json.dumps(spans, separators=(",", ":"))
        checksum = hashlib.sha256(values + raw_spans.encode()).hexdigest()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO body_vectors_v1 VALUES (?,?,?,?,?)",
                (self.identity, digest, values, raw_spans, checksum),
            )

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


def analyze_bodies(
    report: StructuralReport,
    *,
    options: BodyOptions = BodyOptions(),
    threshold: float = 0.9,
    top_k: int | None = None,
    include_same_file: bool = False,
    cache_path: Path | None = None,
    encoder: LocalBodyEncoder | None = None,
    requested_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Encode complete sources; retrieve families or score explicit diagnostic pairs.

    Requested pairs bypass retrieval/grouping and preserve low/negative cosines.
    Their indices refer to report.units, including unavailable source entries.
    """
    import numpy as np
    from numpy.typing import NDArray
    from victor_codegraph.embeddings import validate_vectors
    from victor.core.graph_rag.semantic_clones import SymbolRecord, find_clone_families

    if len(report.units) > options.max_functions:
        raise ValueError(f"Body analysis exceeds {options.max_functions} functions; narrow --scope")
    if requested_pairs is not None and (
        len(requested_pairs) > 50_000
        or any(
            len(pair) != 2
            or any(type(i) is not int or not 0 <= i < len(report.units) for i in pair)
            for pair in requested_pairs
        )
    ):
        raise ValueError("Invalid requested body pair indices or pair budget")
    encoder = encoder or LocalBodyEncoder(options)
    if encoder.options != options:
        raise ValueError("Body encoder profile does not match the requested evidence")
    load_attempts_before = getattr(encoder, "load_attempts", None)
    producer = getattr(encoder, "cache_producer", None)
    declared_producer = isinstance(producer, ProducerProfile)
    if not isinstance(producer, ProducerProfile):
        producer = producer_profile(
            options.identity,
            component_hash=hashlib.sha256(b"undeclared-custom-encoder").hexdigest(),
            instance_nonce=uuid.uuid4().hex,
        )
    if producer.manifest["input_profile"] != options.identity:
        raise ValueError("Body producer input profile does not match the requested evidence")
    cache = _BodyCache(cache_path, options, producer)
    records: list[SymbolRecord] = []
    vectors: list[Any] = []
    metadata: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    # Duplicate exact source inputs are inferred once, independently of paths.
    documents: dict[str, tuple[str, list[Any]]] = {}
    ready: dict[str, Any] = {}
    pending: dict[str, tuple[int, int]] = {}
    texts: list[str] = []
    cache_hits = 0
    row_by_unit: dict[int, int] = {}
    try:
        for unit in report.units:
            digest = hashlib.sha256(unit.source_code.encode()).hexdigest()
            if not unit.source_code:
                errors.append(
                    {"file": unit.file_path, "line": unit.line, "reason": "source-not-retained"}
                )
                continue
            if digest in documents:
                continue
            cached = cache.get(digest)
            if cached is not None:
                ready[digest], spans = cached
                documents[digest] = (unit.source_code, spans)
                cache_hits += 1
                continue
            try:
                windows, spans = encoder.windows(unit.source_code)
            except ValueError as exc:
                errors.append({"file": unit.file_path, "line": unit.line, "reason": str(exc)})
                continue
            if len(texts) + len(windows) > options.max_windows:
                raise ValueError("Body analysis exceeds its total window budget; narrow --scope")
            documents[digest] = (unit.source_code, spans)
            pending[digest] = (len(texts), len(texts) + len(windows))
            texts.extend(windows)
        matrix: NDArray[np.float32] = np.empty((len(texts), options.dimension), dtype=np.float32)
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            values = validate_vectors(
                encoder.encode(batch), count=len(batch), dimension=options.dimension
            )
            matrix[start : start + len(batch)] = values
        for digest, (start, end) in pending.items():
            window_vectors = matrix[start:end]
            ready[digest] = _pool_body_windows(window_vectors)
        if declared_producer and getattr(encoder, "cache_producer", None) != producer:
            raise ValueError("Body producer changed during analysis")
        for digest in pending:
            cache.put(digest, ready[digest], documents[digest][1])
        for unit_index, unit in enumerate(report.units):
            digest = hashlib.sha256(unit.source_code.encode()).hexdigest()
            if digest not in ready:
                continue
            identity = hashlib.sha256(
                json.dumps(
                    [unit.file_path, unit.line, unit.region_index, unit.source_start, digest]
                ).encode()
            ).hexdigest()
            row_by_unit[unit_index] = len(records)
            records.append(
                SymbolRecord(identity, unit.file_path, unit.name, "function", unit.line, digest)
            )
            vectors.append(ready[digest])
            metadata[identity] = {
                "id": identity,
                "file": unit.file_path,
                "symbol": unit.name,
                "line": unit.line,
                "end_line": unit.end_line,
                "language": unit.language,
                "source_kind": unit.source_kind,
                "recovered_from_partial_file": unit.recovered_from_partial_file,
                "source_hash": unit.source_hash,
                "input_hash": digest,
                "window_token_spans": documents[digest][1],
                "truncated": False,
            }
        if requested_pairs is None:
            families = find_clone_families(
                records,
                np.asarray(vectors).reshape(len(vectors), options.dimension),
                threshold=threshold,
                top_k=top_k,
                include_same_file=include_same_file,
                language_ids=[metadata[r.id]["language"] for r in records],
                language_scope=options.comparison_scope,
            )
            pair_scores = None
        else:
            families = []
            pair_scores = _requested_cosines(requested_pairs, row_by_unit, vectors, report, options)
        if declared_producer and getattr(encoder, "cache_producer", None) != producer:
            raise ValueError("Body producer changed during analysis")
    finally:
        cache.close()
    # Portable projection is independent of body-vector extraction coverage.
    source_complete = not any(
        report.stats.get(key)
        for key in (
            "parse_error_files",
            "binding_error_files",
            "unreadable_files",
            "oversized_files",
            "unavailable_languages",
            "unsupported_regions",
            "unresolved_external_regions",
            "malformed_regions",
            "region_limit_files",
            "unavailable_region_extractors",
        )
    )
    return {
        "evidence": "function_body_embeddings",
        "equivalence_verified": False,
        "coverage_complete": bool(report.stats.get("source_files"))
        and bool(report.stats.get("selected_regions", 1))
        and source_complete
        and not errors,
        "profile": {
            **asdict(options),
            "identity": options.identity,
            "producer": {"identity": producer.identity, **producer.manifest},
            "local_files_only": True,
            "quality_calibrated": False,
        },
        "stats": {
            "functions": len(records),
            "model_load_attempts": (
                encoder.load_attempts - load_attempts_before
                if type(load_attempts_before) is int
                else None
            ),
            "model_load_failed": getattr(encoder, "load_failed", None),
            "unique_inputs": len(documents),
            "inferred_windows": len(texts),
            "cache_hits": cache_hits,
            "retrieval": (
                "tiled_exact_cosine" if requested_pairs is None else "requested_pair_cosine"
            ),
            "neighbors": top_k if requested_pairs is None else None,
        },
        "errors": errors,
        "source_manifest": report.source_manifest,
        "source_coverage": {
            key: report.stats.get(key)
            for key in (
                "source_files",
                "selected_regions",
                "parse_error_files",
                "binding_error_files",
                "unreadable_files",
                "oversized_files",
                "unavailable_languages",
                "unsupported_regions",
                "unresolved_external_regions",
                "malformed_regions",
                "region_limit_files",
                "unavailable_region_extractors",
                "source_read_errors",
            )
        },
        "inputs": list(metadata.values()),
        "near_grouping": "connected_components" if requested_pairs is None else None,
        **({"pair_scores": pair_scores} if pair_scores is not None else {}),
        "families": [
            {
                "relation": "body_similarity_candidate",
                "evidence": "function_body_embeddings",
                "score": family.similarity,
                "score_basis": "maximum_matching_edge",
                "min_edge_similarity": family.min_edge_similarity,
                "matching_edges": family.matching_pairs,
                "cohesion": asdict(family.cohesion) if family.cohesion is not None else None,
                "members": [metadata[m.id] for m in family.members],
            }
            for family in families
        ],
    }


def _requested_cosines(
    pairs: Sequence[tuple[int, int]],
    rows: dict[int, int],
    vectors: list[Any],
    report: StructuralReport,
    options: BodyOptions,
) -> list[dict[str, Any]]:
    """Batch selected dot products with bounded workspace, never an N-by-N matrix."""
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32).reshape(len(vectors), options.dimension)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Body vectors must have finite nonzero norms")
    normalized = matrix / norms[:, None]
    result: list[dict[str, Any]] = []
    selected: list[tuple[int, int, int]] = []
    for left, right in pairs:
        item: dict[str, Any] = {"left": left, "right": right, "score": None}
        same = report.units[left].language == report.units[right].language
        if left not in rows or right not in rows:
            item["error"] = "body_unavailable"
        elif options.comparison_scope != "all" and same != (options.comparison_scope == "same"):
            item["error"] = "language_scope_filtered"
        else:
            selected.append((len(result), rows[left], rows[right]))
        result.append(item)
    for start in range(0, len(selected), 1024):
        batch = selected[start : start + 1024]
        lefts = np.array([item[1] for item in batch], dtype=np.int64)
        rights = np.array([item[2] for item in batch], dtype=np.int64)
        values = np.clip(np.sum(normalized[lefts] * normalized[rights], axis=1), -1, 1)
        for selected_item, score in zip(batch, values):
            result[selected_item[0]]["score"] = float(score)
    return result
