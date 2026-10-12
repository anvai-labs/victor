#!/usr/bin/env python3
"""Fail when this docs site drifts from the Anvai family brand.

<docs_dir>/stylesheets/tokens.css is a VERBATIM copy of anvaiops
design/tokens.css (the family's single source of truth); brand.css (the
Material mapping), the self-hosted Inter font and its license are shared
verbatim by every Anvai docs site. This script is itself shared verbatim.
All shared files are pinned by digest so a local edit fails; to adopt a
change, re-copy the files and update the digests in every repo.

mkdocs.yml must load tokens.css then brand.css, keep Material's font
loading off (no third-party font request) and use the custom palette for
every scheme entry. Runs after the docs toolchain is installed (PyYAML ships
with mkdocs).
"""

import hashlib
import sys
from pathlib import Path

import yaml

CANONICAL = "anvaiops design/tokens.css (as of e8215fc) + shared brand.css v1.3"
PINNED = {
    "tokens.css": "2ea50eda33844d7c1ae58a264c8356b5f08841c665365f97aa61df53b175bca6",
    "brand.css": "926255104c120e55d5d3d39d1b0685554a30bf946a5ad7c86a060e5fc52ab335",
    "fonts/inter-latin-wght-normal.woff2": "313809049cc15a226aaca190f5aee5f18d8125a5219eafe1cd0b52098a9b9699",
    "fonts/Inter-OFL.txt": "3b0a5fca3d17942cde889069889dedbbbd075e9b599968c82a95f4d944e9b345",
}
REQUIRED_EXTRA_CSS = ["stylesheets/tokens.css", "stylesheets/brand.css"]


class _Loader(yaml.SafeLoader):
    """SafeLoader that tolerates mkdocs' custom tags (!ENV, !!python/name:...)."""


_Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def check(root: Path) -> list[str]:
    config = yaml.load((root / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)
    docs_dir = root / (config.get("docs_dir") or "docs")
    failures = []
    for name, digest in PINNED.items():
        path = docs_dir / "stylesheets" / name
        if not path.is_file():
            failures.append(f"{path.relative_to(root)} is missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"{path.relative_to(root)} drifted (sha256 {actual}); re-copy it")

    extra_css = [str(entry) for entry in config.get("extra_css") or []]
    listed = [entry for entry in extra_css if entry in REQUIRED_EXTRA_CSS]
    if listed != REQUIRED_EXTRA_CSS:
        wanted = " then ".join(REQUIRED_EXTRA_CSS)
        failures.append(f"mkdocs.yml extra_css must list {wanted} (found {extra_css})")

    theme = config.get("theme") or {}
    if theme.get("font") is not False:
        failures.append("mkdocs.yml theme.font must be false (no third-party font request)")
    palette = theme.get("palette") or []
    entries = [palette] if isinstance(palette, dict) else palette
    schemes = [entry for entry in entries if isinstance(entry, dict) and "scheme" in entry]
    if not schemes:
        failures.append("mkdocs.yml theme.palette has no scheme entries")
    for entry in schemes:
        for key in ("primary", "accent"):
            if entry.get(key) != "custom":
                scheme = entry.get("scheme")
                failures.append(f"mkdocs.yml palette '{scheme}' must set {key}: custom")
    return failures


def main(root: Path) -> int:
    failures = check(root)
    for failure in failures:
        print(f"brand: {failure}", file=sys.stderr)
    if not failures:
        print(f"brand tokens OK ({CANONICAL})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
