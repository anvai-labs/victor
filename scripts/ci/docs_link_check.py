#!/usr/bin/env python3
"""Check local links, assets and anchors in an already-built documentation site.

No network requests are made. Absolute links beneath --site-url resolve against
the artifact, so project Pages prefixes work without fetching the published site.
External links and script-generated anchors are outside this check's scope.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: set[str] = set()
        self.links: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.anchors.add(values["id"] or "")
        if tag == "a" and values.get("name"):
            self.anchors.add(values["name"] or "")
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(values[key] or "")


def check_site(site_dir: Path, site_url: str) -> list[str]:
    root = site_dir.resolve()
    base_url = site_url.rstrip("/") + "/"
    base = urlsplit(base_url)
    if base.scheme not in ("https", "http") or not base.netloc or base.query or base.fragment:
        raise ValueError("site-url must be an absolute HTTP(S) URL without query or fragment")
    prefix = unquote(base.path)
    issues: set[str] = set()
    pages: dict[Path, Page] = {}
    targets: dict[Path, Page] = {}
    for source in root.rglob("*.html"):
        relative = source.relative_to(root).as_posix()
        try:
            resolved = source.resolve()
            if not resolved.is_relative_to(root):
                issues.add(f"{relative}: HTML source path escapes site")
                continue
            pages[source] = Page(resolved.read_text(encoding="utf-8"))
            targets[resolved] = pages[source]
        except (OSError, RuntimeError, UnicodeError, ValueError):
            issues.add(f"{relative}: cannot read HTML source")
    if not pages:
        issues.add(f"No HTML pages found in {root}")
        return sorted(issues)
    for source, page in pages.items():
        relative = source.relative_to(root).as_posix()
        page_url = urljoin(base_url, relative)
        for link in page.links:
            try:
                destination = urlsplit(urljoin(page_url, link))
            except ValueError:
                issues.add(f"{relative}: {link} -> invalid URL")
                continue
            if destination.scheme not in ("http", "https") or destination.netloc != base.netloc:
                continue
            path = unquote(destination.path)
            # Other projects on the same Pages host are external to this artifact.
            if path == prefix.rstrip("/"):
                suffix = ""
            elif path.startswith(prefix):
                suffix = path[len(prefix) :]
            else:
                continue
            try:
                target = (root / suffix).resolve()
                if target.is_relative_to(root) and target.is_dir():
                    target = (target / "index.html").resolve()
            except (OSError, RuntimeError, ValueError):
                issues.add(f"{relative}: {link} -> invalid target path")
                continue
            if not target.is_relative_to(root):
                issues.add(f"{relative}: {link} -> path escapes site")
                continue
            if not target.is_file():
                issues.add(f"{relative}: {link} -> missing target")
                continue
            fragment = unquote(destination.fragment)
            # Text fragments are handled by the browser, not by HTML element IDs.
            fragment = fragment.split(":~:text=", 1)[0]
            if fragment and target in targets and fragment not in targets[target].anchors:
                issues.add(f"{relative}: {link} -> missing anchor #{fragment}")
    return sorted(issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", type=Path)
    parser.add_argument("--site-url", default="https://anvai-labs.github.io/victor/")
    args = parser.parse_args()
    issues = check_site(args.site_dir, args.site_url)
    for issue in issues:
        print(issue)
    print(f"Built-site link check: {len(issues)} issue(s); external URLs were not fetched")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
