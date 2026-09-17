"""Artifact checks catch broken navigation without relying on external websites."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "docs_link_check", Path(__file__).resolve().parents[3] / "scripts/ci/docs_link_check.py"
)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def write(root, path, body):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)


def test_relative_assets_directory_links_and_encoded_anchors(tmp_path):
    write(tmp_path, "index.html", '<a href="guide/?q=1#caf%C3%A9">Guide</a>')
    write(
        tmp_path,
        "guide/index.html",
        '<h1 id="café">Guide</h1><img src="../image.svg"><a href="../">Home</a>',
    )
    write(tmp_path, "image.svg", "<svg/>")
    assert checker.check_site(tmp_path, "https://example.org/victor/") == []


def test_missing_anchors_and_assets_are_reported(tmp_path):
    write(tmp_path, "index.html", '<a href="guide/#absent">Guide</a><img src="missing.svg">')
    write(tmp_path, "guide/index.html", '<h1 id="present">Guide</h1>')
    issues = checker.check_site(tmp_path, "https://example.org/victor/")
    assert len(issues) == 2
    assert any("missing anchor #absent" in issue for issue in issues)
    assert any("missing.svg -> missing target" in issue for issue in issues)


def test_absolute_site_links_checked_but_other_sites_and_projects_skipped(tmp_path):
    write(
        tmp_path,
        "index.html",
        '<a href="https://example.org/victor/missing/">Broken</a>'
        '<a href="/another-project/">Other project</a>'
        '<a href="https://external.invalid/page">External</a>'
        '<a href="mailto:docs@example.org">Mail</a>',
    )
    issues = checker.check_site(tmp_path, "https://example.org/victor/")
    assert len(issues) == 1
    assert "https://example.org/victor/missing/" in issues[0]


def test_anchor_only_legacy_names_and_browser_text_fragments(tmp_path):
    write(
        tmp_path,
        "index.html",
        '<a name="legacy"></a><a href="#legacy">Legacy</a>'
        '<a href="#:~:text=hello">Text</a><a href="#missing">Bad</a>',
    )
    assert checker.check_site(tmp_path, "https://example.org/") == [
        "index.html: #missing -> missing anchor #missing"
    ]


def test_empty_artifact_is_not_a_success(tmp_path):
    assert "No HTML pages" in checker.check_site(tmp_path, "https://example.org/")[0]


def test_project_root_without_trailing_slash_still_checks_anchors(tmp_path):
    write(
        tmp_path,
        "index.html",
        '<h1 id="home">Home</h1><a href="/victor#home">Home</a>'
        '<a href="https://example.org/victor#absent">Bad</a>'
        '<script src="/victor/assets/missing.js"></script>'
        '<a href="/victorious/">Different project</a>',
    )
    issues = checker.check_site(tmp_path, "https://example.org/victor/")
    assert len(issues) == 2
    assert any("missing anchor #absent" in issue for issue in issues)
    assert any("missing.js -> missing target" in issue for issue in issues)


def test_source_html_symlink_cannot_read_outside_artifact(tmp_path, monkeypatch):
    root = tmp_path / "site"
    root.mkdir()
    outside = tmp_path / "outside.html"
    outside.write_text("<html>private</html>")
    write(root, "index.html", "<html></html>")
    (root / "leak.html").symlink_to(outside)
    original_read = Path.read_text

    def checked_read(path, *args, **kwargs):
        assert path.resolve().is_relative_to(root), "read escaped the artifact"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked_read)
    issues = checker.check_site(root, "https://example.org/victor/")
    assert issues == ["leak.html: HTML source path escapes site"]


def test_link_symlink_and_encoded_traversal_cannot_escape_artifact(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (root / "leak.txt").symlink_to(outside)
    write(
        root,
        "index.html",
        '<a href="leak.txt">Symlink</a><a href="%2e%2e/outside.txt">Traversal</a>',
    )
    issues = checker.check_site(root, "https://example.org/victor/")
    assert len(issues) == 2
    assert all("path escapes site" in issue for issue in issues)


def test_malformed_links_report_issues_without_stopping_other_checks(tmp_path):
    write(
        tmp_path,
        "index.html",
        '<a href="http://[invalid">Malformed</a><a href="bad%00name">Invalid path</a>'
        '<img src="missing.svg">',
    )
    issues = checker.check_site(tmp_path, "https://example.org/victor/")
    assert len(issues) == 3
    assert any("http://[invalid -> invalid URL" in issue for issue in issues)
    assert any("bad%00name -> invalid target path" in issue for issue in issues)
    assert any("missing.svg -> missing target" in issue for issue in issues)


def test_internal_html_symlink_keeps_anchor_validation(tmp_path):
    write(tmp_path, "page.data", '<h1 id="found">Title</h1><a href="#missing">Bad</a>')
    (tmp_path / "alias.html").symlink_to(tmp_path / "page.data")
    assert checker.check_site(tmp_path, "https://example.org/victor/") == [
        "alias.html: #missing -> missing anchor #missing"
    ]
