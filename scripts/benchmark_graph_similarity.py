"""Read-only structural audit with reproducible CPU, wall-time and RSS evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

from victor.core.graph_rag.structural_clones import report_to_dict, scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--language", action="append")
    parser.add_argument("--scope")
    parser.add_argument("--metric", choices=["subtree", "shingle"], default="subtree")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-nodes", type=int, default=40)
    parser.add_argument("--cross-language", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start, cpu = time.perf_counter(), time.process_time()
    report = scan(
        args.path.resolve(),
        scope=args.scope,
        near_metric=args.metric,
        near_threshold=args.threshold,
        min_nodes=args.min_nodes,
        languages=set(args.language) if args.language else None,
        cross_language=args.cross_language,
    )
    payload = {
        "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu,
        # macOS reports bytes; Linux reports KiB. Keep the unit explicit.
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        **report_to_dict(report, 200),
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k not in {"families", "suppressed"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
