"""Offline function-body vectors with a separate input profile and derived cache.

Summary vectors are never read or overwritten. Exact cosine retrieval is tiled
and bounded by an explicit function limit; scores are review evidence, not proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from pathlib import Path
import sqlite3
from typing import Any

from victor.core.graph_rag.structural_clones import StructuralReport


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
    """One explicitly pinned upstream model; local cache only, no remote code."""

    def __init__(self, options: BodyOptions, *, model_factory: Any = None) -> None:
        self.options = options
        self.model_factory = model_factory
        self.model: Any = None

    def _load(self) -> Any:
        if self.model is None:
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
            self.model = model
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
        result, spans = [], []
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


class _BodyCache:
    def __init__(self, path: Path | None, profile: BodyOptions) -> None:
        self.profile = profile
        self.db: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=5)
            try:
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS body_vectors_v1 (profile TEXT, input_hash TEXT, vector BLOB NOT NULL, spans TEXT NOT NULL, checksum TEXT NOT NULL, PRIMARY KEY(profile,input_hash))"
                )
            except sqlite3.Error:
                self.db.close()
                self.db = None
                raise

    def get(self, digest: str) -> tuple[Any, list[Any]] | None:
        import numpy as np

        if self.db is None:
            return None
        row = self.db.execute(
            "SELECT vector,spans,checksum FROM body_vectors_v1 WHERE profile=? AND input_hash=?",
            (self.profile.identity, digest),
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
                (self.profile.identity, digest, values, raw_spans, checksum),
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
) -> dict[str, Any]:
    """Batch complete captured function sources, cache by input/profile, retrieve."""
    import numpy as np
    from victor_codegraph.embeddings import validate_vectors
    from victor.core.graph_rag.semantic_clones import SymbolRecord, find_clone_families

    if len(report.units) > options.max_functions:
        raise ValueError(f"Body analysis exceeds {options.max_functions} functions; narrow --scope")
    encoder = encoder or LocalBodyEncoder(options)
    if encoder.options != options:
        raise ValueError("Body encoder profile does not match the requested evidence")
    cache = _BodyCache(cache_path, options)
    records, vectors, metadata, errors = [], [], {}, []
    # Duplicate exact source inputs are inferred once, independently of paths.
    documents: dict[str, tuple[str, list[Any]]] = {}
    ready: dict[str, Any] = {}
    pending: dict[str, tuple[int, int]] = {}
    texts: list[str] = []
    cache_hits = 0
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
        matrix = np.empty((len(texts), options.dimension), dtype=np.float32)
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            values = validate_vectors(
                encoder.encode(batch), count=len(batch), dimension=options.dimension
            )
            matrix[start : start + len(batch)] = values
        for digest, (start, end) in pending.items():
            windows = matrix[start:end]
            normalized = windows / np.linalg.norm(windows, axis=1)[:, None]
            mean = normalized.mean(axis=0)
            norm = np.linalg.norm(mean)
            if not np.isfinite(norm) or not norm:
                raise ValueError("Body window pooling produced an invalid vector")
            ready[digest] = mean / norm
            cache.put(digest, ready[digest], documents[digest][1])
        for unit in report.units:
            digest = hashlib.sha256(unit.source_code.encode()).hexdigest()
            if digest not in ready:
                continue
            identity = hashlib.sha256(
                json.dumps(
                    [unit.file_path, unit.line, unit.region_index, unit.source_start, digest]
                ).encode()
            ).hexdigest()
            records.append(
                SymbolRecord(identity, unit.file_path, unit.name, "function", unit.line, digest)
            )
            vectors.append(ready[digest])
            metadata[identity] = {
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
        families = find_clone_families(
            records,
            np.asarray(vectors).reshape(len(vectors), options.dimension),
            threshold=threshold,
            top_k=top_k,
            include_same_file=include_same_file,
            language_ids=[metadata[r.id]["language"] for r in records],
            language_scope=options.comparison_scope,
        )
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
            "local_files_only": True,
            "quality_calibrated": False,
        },
        "stats": {
            "functions": len(records),
            "unique_inputs": len(documents),
            "inferred_windows": len(texts),
            "cache_hits": cache_hits,
            "retrieval": "tiled_exact_cosine",
            "neighbors": top_k,
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
        "near_grouping": "connected_components",
        "families": [
            {
                "relation": "body_similarity_candidate",
                "evidence": "function_body_embeddings",
                "score": family.similarity,
                "score_basis": "maximum_matching_edge",
                "min_edge_similarity": family.min_edge_similarity,
                "matching_edges": family.matching_pairs,
                "members": [metadata[m.id] for m in family.members],
            }
            for family in families
        ],
    }
