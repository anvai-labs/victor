"""Independent source-evidence channels and consistent-snapshot checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.core.graph_rag.body_clones import BodyOptions
    from victor.core.graph_rag.token_clones import TokenOptions


def analyze_source(
    root: Path,
    *,
    mode: str,
    limit: int = 20,
    scope: str | None = None,
    include_tests: bool = False,
    include_docs: bool = False,
    min_nodes: int = 40,
    near_threshold: float = 0.8,
    near_metric: str = "shingle",
    threshold: float = 0.9,
    top_k: int | None = None,
    include_same_file: bool = False,
    cross_language: bool = False,
    languages: set[str] | None = None,
    body_options: BodyOptions | None = None,
    token_options: TokenOptions | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Union evidence channels without requiring syntax matches before vectors.

    Families remain transitive components of their own channel. They are not
    expanded into invented pairwise matches or assigned a mixed confidence.
    """
    from victor.core.graph_rag.structural_clones import scan, report_to_dict

    if mode not in {"body", "tokens", "hybrid"}:
        raise ValueError("Expected body, tokens or hybrid analysis")
    if mode in {"tokens", "hybrid"} and languages:
        raise ValueError("Token/hybrid language filtering is not supported; use --scope")
    channels: dict[str, Any] = {}
    report = None
    if mode != "tokens":
        report = scan(
            root,
            scope=scope,
            include_tests=include_tests,
            min_nodes=min_nodes,
            near_threshold=near_threshold,
            include_same_file=include_same_file,
            near_metric=near_metric,
            languages=languages,
            cross_language=cross_language,
            include_docs=include_docs,
            retain_source=True,
            collect_only=mode == "body",
            max_units=body_options.max_functions if body_options is not None else 5000,
        )
        if mode == "hybrid":
            channels["structural"] = report_to_dict(report, limit)
        try:
            from victor.core.graph_rag.body_clones import BodyOptions, analyze_bodies

            channels["body"] = analyze_bodies(
                report,
                options=body_options or BodyOptions(),
                threshold=threshold,
                top_k=top_k,
                include_same_file=include_same_file,
                cache_path=cache_path,
            )
            channels["body"]["total_families"] = len(channels["body"]["families"])
            channels["body"]["families"] = channels["body"]["families"][:limit]
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            channels["body"] = {
                "evidence": "function_body_embeddings",
                "coverage_complete": False,
                "equivalence_verified": False,
                "error": str(exc)[:1500],
            }
    if mode != "body":
        try:
            from victor.core.graph_rag.token_clones import TokenOptions, scan_tokens

            channels["tokens"] = scan_tokens(
                root,
                options=token_options or TokenOptions(),
                scope=scope,
                include_tests=include_tests,
                include_docs=include_docs,
                include_same_file=include_same_file,
            )
            channels["tokens"]["total_fragments"] = len(channels["tokens"]["fragments"])
            channels["tokens"]["fragments"] = channels["tokens"]["fragments"][:limit]
        except (OSError, RuntimeError, ValueError) as exc:
            channels["tokens"] = {
                "evidence": "token_sequence",
                "coverage_complete": False,
                "equivalence_verified": False,
                "error": str(exc)[:1500],
            }
    inconsistencies = []
    if report is not None and "source_manifest" in channels.get("tokens", {}):
        manifest = channels["tokens"]["source_manifest"]
        inconsistencies = sorted(
            path
            for path in set(manifest) | set(report.source_manifest)
            if manifest.get(path, {}).get("sha256") != report.source_manifest.get(path)
        )
    payload: dict[str, Any] = {
        "evidence": "independent_source_channels",
        "equivalence_verified": False,
        "coverage_complete": not inconsistencies
        and all(channel["coverage_complete"] for channel in channels.values()),
        "snapshot_conflicts": inconsistencies,
        "channels": channels,
    }
    if mode != "hybrid":
        return next(iter(channels.values()))
    return payload
