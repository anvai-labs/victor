#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate the canonical feature-flag inventory doc from code.

Facts that get restated in prose drift silently (see F-016/TD-17: `USE_SMART_ROUTING`
was documented "default OFF" in three places while the code defaulted it ON). This
generator makes the flag inventory a *generated* artifact instead of hand-written
prose: the single source of truth is ``victor.core.feature_flags.get_flag_manifest()``,
and a CI guard (``tests/unit/runtime/test_feature_flag_manifest_guard.py``) fails if
the committed doc drifts from it.

Usage:
    python scripts/gen_feature_flag_doc.py           # rewrite the doc
    python scripts/gen_feature_flag_doc.py --check    # exit 1 if the doc is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "architecture" / "feature-flags.md"


def render_feature_flag_markdown(manifest: List[Dict[str, object]]) -> str:
    """Render the flag manifest to the canonical (byte-stable) markdown doc.

    ``manifest`` is the output of ``get_flag_manifest()`` — a name-sorted list of
    ``{"name", "opt_in", "default"}`` entries. The output is deterministic so the
    guard test can byte-compare it against the committed doc.
    """
    total = len(manifest)
    opt_in = sum(1 for entry in manifest if entry["opt_in"])
    default_on = total - opt_in

    lines = [
        "# Feature Flags (generated)",
        "",
        "> **Generated file — do not edit by hand.** Regenerate with "
        "`python scripts/gen_feature_flag_doc.py`. A CI guard "
        "(`tests/unit/runtime/test_feature_flag_manifest_guard.py`) fails if this "
        "drifts from `victor.core.feature_flags.FeatureFlag`.",
        ">",
        "> A flag's **code default** is OFF iff it is in `is_opt_in_by_default()` "
        "(assuming no YAML/env override). This table is the single source of truth "
        "for flag defaults — cite it instead of restating defaults in prose, which "
        "drifts (see F-016 / TD-17).",
        "",
        f"Total flags: {total} · Opt-in (default OFF): {opt_in} · Default ON: {default_on}",
        "",
        "| Flag | Code default | Opt-in |",
        "|------|--------------|--------|",
    ]
    for entry in manifest:
        default_label = "ON" if entry["default"] else "OFF"
        opt_in_label = "yes" if entry["opt_in"] else "no"
        lines.append(f"| `{entry['name']}` | {default_label} | {opt_in_label} |")
    lines.append("")
    return "\n".join(lines)


def _current_doc() -> str:
    from victor.core.feature_flags import get_flag_manifest

    return render_feature_flag_markdown(get_flag_manifest())


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed doc is stale (does not write).",
    )
    args = parser.parse_args(argv)

    rendered = _current_doc()

    if args.check:
        existing = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
        if existing != rendered:
            print(
                f"{DOC_PATH.relative_to(ROOT)} is stale — run "
                "`python scripts/gen_feature_flag_doc.py`",
                file=sys.stderr,
            )
            return 1
        print(f"{DOC_PATH.relative_to(ROOT)} is up to date")
        return 0

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {DOC_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
