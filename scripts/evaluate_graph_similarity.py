"""Evaluate reviewed function pairs with project-held-out calibration.

No source is executed and no model is downloaded. Missing evidence returns exit
1. --require-decisions also rejects facets without qualified calibration support.
Controlled fixtures exercise this protocol; their results are not product accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from victor.core.graph_rag.similarity_evaluation import Corpus, calibrate, evaluate
from victor.core.graph_rag.similarity_pair_scoring import score_corpus
from victor.core.graph_rag.source_reader import read_project_source


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def load_corpus(path: Path) -> Corpus:
    data, info = read_project_source(path.absolute().parent, path.name, 16 * 1024 * 1024)
    if info.st_size > 16 * 1024 * 1024 or len(data) > 16 * 1024 * 1024:
        raise ValueError("Corpus manifest exceeds its byte budget")
    return Corpus.from_dict(json.loads(data, object_pairs_hook=_unique_object))


def _aliases(left: Path, right: Path) -> bool:
    # Conservatively protect case aliases even before a SQLite sidecar exists
    # on a filesystem whose case rules may differ from this process's platform.
    if str(left.resolve()).casefold() == str(right.resolve()).casefold():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def _write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Explicit source root for each corpus project; no project config is run",
    )
    parser.add_argument("--channel", choices=["syntax", "portable", "body"], default="syntax")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Optional separate body-vector SQLite cache")
    parser.add_argument(
        "--body-profile", type=Path, help="Pinned BodyOptions JSON; local weights only"
    )
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--minimum-support", type=int, default=2)
    parser.add_argument("--rank-k", type=int, default=10)
    parser.add_argument("--require-decisions", action="store_true")
    parser.add_argument("--minimum-test-precision", type=float)
    args = parser.parse_args(argv)
    try:
        roots = {}
        for value in args.project:
            key, separator, path = value.partition("=")
            if not separator or not key or not path or key in roots:
                raise ValueError("Project roots require unique ID=PATH arguments")
            roots[key] = Path(path).absolute()
        if not math.isfinite(args.minimum_precision) or not 0 < args.minimum_precision <= 1:
            raise ValueError("Minimum precision must be finite and in (0,1]")
        if args.minimum_test_precision is not None and (
            not math.isfinite(args.minimum_test_precision)
            or not 0 < args.minimum_test_precision <= 1
        ):
            raise ValueError("Test precision gate must be finite and in (0,1]")
        if not 1 <= args.minimum_support <= 50_000 or not 1 <= args.rank_k <= 50_000:
            raise ValueError("Support/rank budgets must be positive and bounded")
        body_options = None
        if args.body_profile is not None:
            if args.channel != "body":
                raise ValueError("A body profile requires --channel body")
            from victor.core.graph_rag.body_clones import BodyOptions

            content, info = read_project_source(
                args.body_profile.absolute().parent, args.body_profile.name, 16_384
            )
            if info.st_size > 16_384 or len(content) > 16_384:
                raise ValueError("Body profile exceeds its byte budget")
            raw = json.loads(content, object_pairs_hook=_unique_object)
            allowed = {field.name for field in fields(BodyOptions)}
            if (
                not isinstance(raw, dict)
                or not {"model_id", "revision", "dimension"} <= set(raw) <= allowed
            ):
                raise ValueError(
                    "Body profiles require an explicit model/revision/dimension and known fields"
                )
            defaults = asdict(BodyOptions())
            if any(type(value) is not type(defaults[key]) for key, value in raw.items()):
                raise ValueError("Invalid body profile field types")
            body_options = BodyOptions(**raw)
        corpus = load_corpus(args.corpus)
        protected = {
            args.corpus.resolve(),
            *(
                root / ".victor" / ("project.db" + suffix)
                for root in roots.values()
                for suffix in ("", "-wal", "-shm", "-journal")
            ),
            *(
                root / unit.file
                for unit in corpus.units.values()
                if (root := roots.get(unit.project)) is not None
            ),
        }
        if args.body_profile is not None:
            protected.add(args.body_profile.resolve())
        if any(_aliases(args.output, path) for path in protected):
            raise ValueError("Evaluation output cannot overwrite a manifest or source input")
        if args.cache is not None:
            if args.cache.is_symlink() or (
                args.cache.exists()
                and (not args.cache.is_file() or args.cache.stat().st_nlink != 1)
            ):
                raise ValueError("Body cache must be an exclusive regular file, without links")
            if any(_aliases(args.cache, path) for path in (*protected, args.output)):
                raise ValueError("Body cache must be separate from inputs and the report")
        cpu, wall = time.process_time(), time.perf_counter()
        run = score_corpus(
            corpus, roots, channel=args.channel, body_options=body_options, cache_path=args.cache
        )
        policy = calibrate(
            corpus,
            run.observations,
            channel=run.channel,
            profile=run.profile,
            minimum_precision=args.minimum_precision,
            minimum_support=args.minimum_support,
        )
        report = evaluate(corpus, run.observations, policy, profile=run.profile, k=args.rank_k)
        report.update(
            {
                "runner_implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "corpus_manifest": corpus.to_dict(),
                "detector_profile": run.descriptor,
                "profile_identity": run.profile,
                "observations": [asdict(item) for item in run.observations],
                "source_manifest": list(run.sources),
                "run_details": run.details,
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
                "timing_scope": "source_capture_scoring_calibration_report_assembly",
            }
        )
        try:
            import resource

            report["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024
            )
        except ImportError:
            report["process_max_rss_bytes"] = None
        report["rss_scope"] = "process_lifetime_peak"
        report["labelled_pairs_per_second"] = len(run.observations) / max(
            report["wall_seconds"], 1e-9
        )
        qualified = report["coverage_complete"]
        if args.require_decisions or args.minimum_test_precision is not None:
            qualified = qualified and report["decision_complete"]
        if args.minimum_test_precision is not None:
            precision = report["test"]["overall"]["precision"]
            qualified = (
                qualified and precision is not None and precision >= args.minimum_test_precision
            )
            for metrics in report["test"]["facets"].values():
                qualified = (
                    qualified
                    and metrics["positive_pairs"] >= args.minimum_support
                    and metrics["labelled_pairs"] - metrics["positive_pairs"]
                    >= args.minimum_support
                    and metrics["precision"] is not None
                    and metrics["precision"] >= args.minimum_test_precision
                )
        report["requested_gates"] = {
            "require_decisions": args.require_decisions,
            "minimum_test_precision": args.minimum_test_precision,
            "minimum_test_class_support": args.minimum_support,
            "precision_scope": "each_test_facet_and_overall",
        }
        report["requested_gates_passed"] = qualified
        _write(args.output, report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "coverage_complete",
                        "decision_complete",
                        "requested_gates_passed",
                        "production_quality_qualified",
                        "profile_identity",
                        "wall_seconds",
                    )
                },
                indent=2,
            )
        )
        return 0 if qualified else 1
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {str(exc)[:1500]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
