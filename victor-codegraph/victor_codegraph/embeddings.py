"""Store-neutral semantic embedding evidence; model dependencies are optional.

Inference and publication failures propagate. Graph identity is supplied by the
compiler; this module neither derives replacement IDs nor writes to a database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import islice
import math
from numbers import Real
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .model import CodeSymbol


def embedding_text(name: str, signature: str | None, documentation: str | None) -> str:
    """The symbol-summary-v1 input, identical to Victor's existing node text."""
    return " ".join(filter(None, [name, signature or "", documentation or ""]))


@dataclass(frozen=True)
class EmbeddingProfile:
    """Complete compatibility key for document and query vectors."""

    model_id: str
    revision: str
    dimension: int
    text_contract: str = "symbol-summary-v1"
    max_tokens: int = 512
    normalize: bool = False
    query_prefix: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id is required")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("model revision must be an immutable 40-hex commit")
        if type(self.dimension) is not int or not 1 <= self.dimension <= 65536:
            raise ValueError("invalid embedding dimension")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 32768:
            raise ValueError("invalid embedding token limit")
        if self.text_contract not in _TEXT_CONTRACTS:
            raise ValueError("unsupported embedding text contract")
        if type(self.normalize) is not bool or not isinstance(self.query_prefix, str):
            raise ValueError("invalid embedding preprocessing")

    @property
    def identity(self) -> str:
        return _digest(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))


_TEXT_CONTRACTS = frozenset({"symbol-summary-v1", "chunk-source-v1"})

BGE_SMALL_PROFILE = EmbeddingProfile(
    model_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
)

# Intent tier: embeds SOURCE text of code chunks (chunk-source-v1). Same
# pinned model/revision as the symbol-summary tier; the distinct
# text_contract gives it a distinct profile identity, so the two tiers can
# never be mixed in one store.
CHUNK_SOURCE_PROFILE = EmbeddingProfile(
    model_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
    text_contract="chunk-source-v1",
)


@dataclass(frozen=True)
class SymbolEmbedding:
    symbol_id: str
    profile_id: str
    input_hash: str
    source_hash: str
    values: tuple[float, ...]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_vectors(
    vectors: Iterable[Iterable[float]], *, count: int, dimension: int
) -> list[list[float]]:
    """Reject failed, truncated or malformed inference before any publication."""
    if type(count) is not int or not 0 <= count <= 512:
        raise ValueError("embedding batch must contain at most 512 inputs")
    if type(dimension) is not int or not 1 <= dimension <= 65536:
        raise ValueError("invalid embedding dimension")
    rows = [list(islice(row, dimension + 1)) for row in islice(vectors, count + 1)]
    if len(rows) != count:
        raise ValueError("embedding batch must return exactly one vector per input")
    for row in rows:
        if len(row) != dimension:
            raise ValueError("embedding dimension mismatch")
        if any(
            isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v))
            for v in row
        ):
            raise ValueError("embedding values must be finite numeric values")
        if not any(float(v) != 0.0 for v in row):
            raise ValueError("zero vector is not a semantic embedding")
    return [[float(v) for v in row] for row in rows]


def embed_symbols(
    symbols: Sequence[CodeSymbol],
    encode: Callable[[list[str]], Iterable[Iterable[float]]],
    profile: EmbeddingProfile,
    *,
    batch_size: int = 32,
) -> list[SymbolEmbedding]:
    """Prepare, batch and validate compiler symbols without a store dependency.

    ``encode`` is trusted configuration: it must implement the supplied profile.
    Backend identity and real-model quality require separate deployment acceptance.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 512:
        raise ValueError("embedding batch size must be between 1 and 512")
    ids = [s.id for s in symbols]
    if len(set(ids)) != len(ids) or any(not sid for sid in ids):
        raise ValueError("empty or duplicate canonical symbol identity")
    eligible = [s for s in symbols if not s.metadata.get("embedding_excluded")]
    result = []
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start : start + batch_size]
        texts = [embedding_text(s.simple_name, s.signature, s.documentation) for s in batch]
        vectors = validate_vectors(encode(texts), count=len(texts), dimension=profile.dimension)
        for symbol, text, vector in zip(batch, texts, vectors, strict=True):
            result.append(
                SymbolEmbedding(
                    symbol.id,
                    profile.identity,
                    _digest(text),
                    _digest(symbol.source_code),
                    tuple(vector),
                )
            )
    return result


@dataclass(frozen=True)
class ChunkEmbedding:
    chunk_id: str
    symbol_id: str
    profile_id: str
    input_hash: str
    source_hash: str
    values: tuple[float, ...]


def embed_chunks(
    chunks: Sequence[Any],
    encode: Callable[[list[str]], Iterable[Iterable[float]]],
    profile: EmbeddingProfile,
    *,
    source_hashes: Mapping[str, str] | None = None,
    batch_size: int = 32,
) -> list[ChunkEmbedding]:
    """Prepare, batch and validate chunk-source embeddings for the intent tier.

    Mirrors :func:`embed_symbols` exactly: batch cap, vector validation,
    duplicate/empty identity rejection. ``chunks`` are ``CodeChunk``-shaped
    objects (``chunk_id``, ``text``, ``symbol_id``, ``metadata``).
    ``source_hashes`` optionally maps file path -> manifest content hash;
    the file path is read from ``chunk.metadata["file_path"]`` when present,
    falling back to the chunk's own source digest discipline.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 512:
        raise ValueError("embedding batch size must be between 1 and 512")
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids) or any(not cid for cid in ids):
        raise ValueError("empty or duplicate canonical chunk identity")
    result = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        vectors = validate_vectors(encode(texts), count=len(texts), dimension=profile.dimension)
        for chunk, vector in zip(batch, vectors, strict=True):
            file_path = chunk.metadata.get("file_path", "")
            manifest_hash = (source_hashes or {}).get(file_path)
            # Verbatim manifest hash when supplied (it is already the
            # sha256 content contract); otherwise digest the chunk text.
            source_hash = manifest_hash or _digest(chunk.text)
            result.append(
                ChunkEmbedding(
                    chunk.chunk_id,
                    chunk.symbol_id,
                    profile.identity,
                    _digest(chunk.text),
                    source_hash,
                    tuple(vector),
                )
            )
    return result


def detect_best_device() -> str:
    """Auto-detect the best available compute device for embedding inference.

    Priority: NVIDIA CUDA > AMD ROCm/HIP (both surface through
    ``torch.cuda``) > Apple MPS > CPU. NVIDIA is preferred over AMD when
    both exist because CUDA attention kernels are more mature for the
    encoder architecture. Falls back to CPU when no accelerator is found.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            # No CUDA/ROCm device; try Apple MPS.
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        # GPU present — prefer NVIDIA (mature CUDA attention kernels).
        name = torch.cuda.get_device_name(0)
        if "NVIDIA" in name:
            return "cuda"
        # AMD ROCm/HIP — still fast for inference, just less tuned kernels.
        if "AMD" in name or "Radeon" in name:
            return "cuda"
        return "cuda"
    except ImportError:
        pass
    return "cpu"


class SentenceTransformerEncoder:
    """Optional local inference for an explicit immutable profile.

    Uses sentence-transformers itself; no alternative model implementation or
    remote-code execution. Own one encoder per configured worker/model lifetime.

    Auto-detects the best available compute device unless ``device`` is
    explicitly provided. Priority: NVIDIA CUDA > AMD ROCm/HIP > Apple MPS >
    CPU. Both NVIDIA and AMD GPUs surface through ``torch.cuda`` (ROCm
    builds alias to the CUDA API); NVIDIA is preferred when both exist.
    """

    semantic = True

    def __init__(
        self,
        profile: EmbeddingProfile = BGE_SMALL_PROFILE,
        *,
        device: str | None = None,
    ) -> None:
        import threading

        self.profile = profile
        self.device = device if device is not None else detect_best_device()
        self._model: Any = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode one bounded document batch; propagate model/installation errors."""
        if len(texts) > 512:
            raise ValueError("embedding batch exceeds 512 inputs")
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self.profile.model_id,
                    revision=self.profile.revision,
                    device=self.device,
                    trust_remote_code=False,
                )
                if model.get_sentence_embedding_dimension() != self.profile.dimension:
                    raise ValueError("model/profile embedding dimension mismatch")
                model.max_seq_length = self.profile.max_tokens
                self._model = model
            values = self._model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=self.profile.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return validate_vectors(values, count=len(texts), dimension=self.profile.dimension)

    def encode_query(self, query: str) -> list[float]:
        return self.encode([self.profile.query_prefix + query])[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(self.encode, texts)

    async def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        # Compatibility with Victor's injected embedding-service seam. Cache
        # ownership remains outside this encoder; the flag never changes semantics.
        return (await self.embed_batch([text]))[0]
