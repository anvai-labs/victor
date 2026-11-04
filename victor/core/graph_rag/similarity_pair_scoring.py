"""Read reviewed source snapshots and score explicitly labelled function pairs.

No source is executed. SCIP and graph relationships are outside this initial
pair-evaluation profile. Model loading uses the existing pinned offline encoder.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sqlite3
from typing import Any, Callable, TYPE_CHECKING

from victor.core.graph_rag.similarity_evaluation import CHANNELS, Corpus, Observation, digest
from victor.core.graph_rag.structural_clones import FunctionUnit, StructuralReport, extract_units
from victor.core.graph_rag.body_provenance import ProducerProfile

if TYPE_CHECKING:
    from victor.core.graph_rag.body_clones import BodyOptions, LocalBodyEncoder

MAX_SOURCE_BYTES = 128 * 1024 * 1024
ENGINE_FILES = (
    "ast_fingerprints.py",
    "body_clones.py",
    "body_provenance.py",
    "clone_bindings.py",
    "language_parsers.py",
    "portable_clones.py",
    "numeric_literals.py",
    "similarity_pair_scoring.py",
    "similarity_evaluation.py",
    "source_reader.py",
    "structural_clones.py",
    "treesitter_clones.py",
)


@dataclass(frozen=True)
class ScoreRun:
    channel: str
    profile: str
    descriptor: dict[str, Any]
    observations: tuple[Observation, ...]
    sources: tuple[dict[str, Any], ...]
    details: dict[str, Any]


def detector_profile(
    channel: str, options: BodyOptions | None = None, *, producer: ProducerProfile | None = None
) -> dict[str, Any]:
    """Include normalization, implementation and installed frontend/model versions."""
    from victor.core.graph_rag.language_parsers import LANGUAGE_MODULES
    from victor.core.graph_rag.portable_clones import PORTABLE_SCHEMA

    if channel == "body" and producer is None:
        from victor.core.graph_rag.body_clones import BodyOptions, LocalBodyEncoder

        producer = LocalBodyEncoder(options or BodyOptions()).cache_producer

    versions: dict[str, str | None] = {}
    packages = {"tree-sitter", *(value[0].replace("_", "-") for value in LANGUAGE_MODULES.values())}
    if channel == "body":
        packages.update(
            {
                "numpy",
                "sentence-transformers",
                "torch",
                "transformers",
                "tokenizers",
                "victor-codegraph",
            }
        )
    for package in sorted(packages):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "schema": "victor-pair-score-profile-v1",
        "channel": channel,
        "python": platform.python_version(),
        "dependencies": versions,
        "normalization": "normalized-body-v4-field-roles",
        "portable": PORTABLE_SCHEMA,
        "compiler_evidence": "not_evaluated",
        "score_basis": "cosine" if channel == "body" else "set_jaccard",
        "body": asdict(options) if options is not None else None,
        "body_producer": producer.manifest if producer is not None else None,
        "implementation": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ENGINE_FILES
        },
    }


def _extract(file: str, source: str, language: str, channel: str) -> list[FunctionUnit]:
    if language == "python":
        return extract_units(
            file,
            source,
            None,
            min_nodes=1,
            strict_parse=True,
            with_subtrees=channel == "syntax",
            with_portable=channel == "portable",
            retain_source=True,
        )
    from victor.core.graph_rag.language_parsers import create_parser
    from victor.core.graph_rag.treesitter_clones import extract_tree_units

    parser = create_parser(language)
    if parser is None:
        raise ValueError(f"Frontend unavailable: {language}")
    diagnostics: dict[str, int] = {}
    units = extract_tree_units(
        file,
        source,
        parser,
        1,
        True,
        language,
        with_subtrees=channel == "syntax",
        with_portable=channel == "portable",
        diagnostics=diagnostics,
        retain_source=True,
    )
    if diagnostics.get("parse_error_files"):
        raise ValueError("Source has parse errors; pair profile requires an intact snapshot")
    return units


def score_corpus(
    corpus: Corpus,
    roots: dict[str, Path],
    *,
    channel: str = "syntax",
    body_options: BodyOptions | None = None,
    encoder: LocalBodyEncoder | None = None,
    cache_path: Path | None = None,
    max_file_bytes: int = 1_000_000,
) -> ScoreRun:
    """Score known pairs directly, retaining missing evidence as abstentions."""
    from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES
    from victor.core.graph_rag.portable_clones import MIN_PORTABLE_NODES
    from victor.core.graph_rag.source_reader import project_source_reader

    if (
        channel not in CHANNELS
        or type(max_file_bytes) is not int
        or not 1 <= max_file_bytes <= 16_000_000
    ):
        raise ValueError("Invalid pair channel or file byte budget")
    if set(roots) != set(corpus.projects):
        raise ValueError("Project roots must match every declared corpus project exactly")
    if channel != "body" and (
        body_options is not None or encoder is not None or cache_path is not None
    ):
        raise ValueError("Body configuration requires the body channel")
    if channel == "body" and body_options is None:
        from victor.core.graph_rag.body_clones import BodyOptions

        body_options = BodyOptions()
    producer = None
    if channel == "body":
        from victor.core.graph_rag.body_clones import LocalBodyEncoder

        assert body_options is not None
        encoder = encoder or LocalBodyEncoder(body_options)
        producer = getattr(encoder, "cache_producer", None)
        if not isinstance(producer, ProducerProfile):
            raise ValueError("Body pair evaluation requires explicit encoder producer provenance")
        if producer.manifest["input_profile"] != body_options.identity:
            raise ValueError("Body producer input profile does not match pair evaluation options")
    descriptor = detector_profile(channel, body_options, producer=producer)
    found: dict[str, FunctionUnit] = {}
    errors: dict[str, str] = {}
    files: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for unit in corpus.units.values():
        files[(unit.project, unit.file)].append(unit)
    sources: list[dict[str, Any]] = []
    total_bytes = 0
    retained_bytes = 0
    with ExitStack() as stack:
        readers: dict[str, Callable[[str, int], tuple[bytes, Any]]] = {}
        root_errors = {}
        for project, root in sorted(roots.items()):
            try:
                readers[project] = stack.enter_context(project_source_reader(root))
            except OSError as exc:
                root_errors[project] = str(exc)[:500]
        for (project, file), units in sorted(files.items()):
            try:
                if project in root_errors:
                    raise OSError(root_errors[project])
                data, info = readers[project](file, max_file_bytes)
                if info.st_size > max_file_bytes or len(data) > max_file_bytes:
                    raise ValueError("Source exceeds the evaluation file budget")
                total_bytes += len(data)
                if total_bytes > MAX_SOURCE_BYTES:
                    raise RuntimeError("Evaluation exceeds its total source byte budget")
                actual = hashlib.sha256(data).hexdigest()
                if any(unit.source_hash != actual for unit in units):
                    raise ValueError("Source hash mismatch; refresh and rereview the corpus label")
                languages = {unit.language for unit in units}
                if len(languages) != 1 or next(iter(languages)) != SOURCE_LANGUAGES.get(
                    Path(file).suffix.lower()
                ):
                    raise ValueError("Label language does not match the source extension")
                extracted = _extract(
                    f"{project}/{file}",
                    data.decode("utf-8"),
                    units[0].language,
                    channel,
                )
                for reference in units:
                    matches = [
                        unit
                        for unit in extracted
                        if (unit.name, unit.line, unit.end_line)
                        == (reference.symbol, reference.line, reference.end_line)
                    ]
                    if len(matches) != 1:
                        errors[reference.id] = (
                            "Source span/symbol did not identify one intact function"
                        )
                        continue
                    found[reference.id] = replace(matches[0], source_hash=actual)
                    retained_bytes += len(matches[0].source_code.encode())
                    if retained_bytes > MAX_SOURCE_BYTES:
                        raise RuntimeError("Evaluation exceeds its retained body byte budget")
                    sources.append(
                        {
                            "unit": reference.id,
                            "project": project,
                            "file": file,
                            "line": reference.line,
                            "end_line": reference.end_line,
                            "source_hash": actual,
                            "input_hash": hashlib.sha256(
                                matches[0].source_code.encode()
                            ).hexdigest(),
                        }
                    )
            except (OSError, UnicodeError, SyntaxError, RecursionError, ValueError) as exc:
                for unit in units:
                    errors[unit.id] = str(exc)[:500]
    observations: list[Observation] = []
    available = []
    for pair in corpus.pairs:
        unavailable = [
            f"{uid}: {errors.get(uid, 'source unavailable')}"
            for uid in (pair.left, pair.right)
            if uid not in found
        ]
        if unavailable:
            observations.append(Observation(pair.id, None, "; ".join(unavailable)))
        else:
            available.append(pair)
    details: dict[str, Any] = {"source_bytes": total_bytes, "retained_bytes": retained_bytes}
    if channel == "body":
        from victor.core.graph_rag.body_clones import analyze_bodies

        # Use the production window/cache/pooling path once; skip global neighbor
        # retrieval and compute only the labelled pair dots in bounded batches.
        ids = list(found)
        index = {uid: i for i, uid in enumerate(ids)}
        report = StructuralReport(
            [],
            [],
            {"source_files": len(files), "selected_regions": len(files)},
            units=[found[uid] for uid in ids],
        )
        try:
            assert body_options is not None
            payload = analyze_bodies(
                report,
                options=body_options,
                encoder=encoder,
                cache_path=cache_path,
                requested_pairs=[(index[p.left], index[p.right]) for p in available],
            )
            assert producer is not None
            if payload["profile"]["producer"] != {
                "identity": producer.identity,
                **producer.manifest,
            }:
                raise ValueError("Body analysis producer does not match evaluation provenance")
            for pair, item in zip(available, payload["pair_scores"], strict=True):
                observations.append(Observation(pair.id, item["score"], item.get("error")))
            details.update(
                {
                    "body_profile": payload["profile"],
                    "body_stats": payload["stats"],
                    "body_errors": payload["errors"],
                }
            )
        except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            observations.extend(Observation(pair.id, None, str(exc)[:500]) for pair in available)
    else:
        fingerprints = {
            uid: set(unit.subtree_hashes if channel == "syntax" else unit.portable_hashes)
            for uid, unit in found.items()
        }
        for pair in available:
            left, right = found[pair.left], found[pair.right]
            reason = None
            if channel == "syntax" and left.language != right.language:
                reason = "Syntax fingerprints have separate language namespaces"
            elif channel == "portable" and (left.portable_error or right.portable_error):
                reason = f"Portable projection unavailable: {left.portable_error or right.portable_error}"
            elif (
                channel == "portable"
                and min(left.portable_size, right.portable_size) < MIN_PORTABLE_NODES
            ):
                reason = "Portable projection below the production minimum size"
            a, b = fingerprints[pair.left], fingerprints[pair.right]
            if not a or not b:
                reason = reason or "Empty fingerprint representation"
            observations.append(
                Observation(pair.id, None, reason)
                if reason
                else Observation(pair.id, len(a & b) / len(a | b))
            )
    current_producer = (
        getattr(encoder, "cache_producer", producer) if encoder is not None else producer
    )
    if (
        current_producer != producer
        or detector_profile(channel, body_options, producer=current_producer) != descriptor
    ):
        observations = [
            Observation(pair.id, None, "Detector implementation changed during evaluation")
            for pair in corpus.pairs
        ]
    return ScoreRun(
        channel,
        digest(descriptor),
        descriptor,
        tuple(sorted(observations, key=lambda item: item.pair_id)),
        tuple(sources),
        details,
    )
