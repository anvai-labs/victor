"""Host-coordinate mapping and explicit embedded-source coverage."""

import pytest

from victor.core.graph_rag.structural_clones import report_to_dict, scan

TS = "function compute(x:number,y:number) { const value=x-y; return value*3+4; }"
RS = "fn compute(x:i32,y:i32)->i32 { let value=x-y; value*3+4 }"


@pytest.mark.parametrize("suffix", ["vue", "svelte", "astro"])
def test_script_regions_are_discovered_and_mapped(tmp_path, suffix):
    text = "<template>ignore this</template>\n<script lang='ts'>\n" + TS + "\n</script>"
    for name in ("left", "right"):
        (tmp_path / f"{name}.{suffix}").write_text(text)
    report = scan(tmp_path, min_nodes=5, near_metric="subtree", languages={"typescript"})
    assert report.coverage_complete, report.stats
    family = next(f for f in report.families if f.tier == "exact")
    assert {m.line for m in family.members} == {3}
    assert report.stats["embedded_regions"] == 2
    members = report_to_dict(report, 10)["families"][0]["members"]
    assert all(m["source_kind"] == "script" for m in members)
    assert all(m["resolved_symbols"] == 0 for m in members)


def test_same_line_script_and_multiple_regions(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.vue").write_text(
            "<script>"
            + TS.replace(":number", "")
            + "</script>\n<script lang='ts'>"
            + TS
            + "</script>"
        )
    report = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert report.stats["functions"] == 4
    assert {m.line for f in report.families for m in f.members} == {1, 2}


def test_markdown_requires_explicit_docs_opt_in(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.md").write_text("example\n\n```rust\n" + RS + "\n```\n")
    assert scan(tmp_path, min_nodes=5).stats["source_files"] == 0
    report = scan(tmp_path, min_nodes=5, near_metric="subtree", include_docs=True)
    assert report.coverage_complete, report.stats
    assert any(f.tier == "exact" for f in report.families)
    assert {m.line for f in report.families for m in f.members} == {4}
    assert all(m.source_kind == "documentation" for f in report.families for m in f.members)


def test_blockquoted_python_fence_has_host_lines(tmp_path):
    text = "> ```python\n> def compute(x):\n>     return x*3+4\n> ```\n"
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(text)
    report = scan(tmp_path, include_docs=True, min_nodes=5, near_metric="subtree")
    assert report.coverage_complete, report.stats
    assert {m.line for f in report.families for m in f.members} == {2}


@pytest.mark.parametrize(
    "text",
    [
        "<script lang='coffee'>function f() {return 3}</script>",
        "<script src='./external.ts'></script>",
        "<script>" + TS,
    ],
)
def test_unresolved_or_malformed_script_cannot_claim_complete(tmp_path, text):
    (tmp_path / "a.vue").write_text(text)
    report = scan(tmp_path, min_nodes=5)
    assert not report.coverage_complete
    assert not report.families


def test_documentation_cannot_create_runtime_binding_links(tmp_path):
    java = "class Native { public native int run(int value); }"
    native = "JNIEXPORT jint JNICALL Java_Native_run(JNIEnv* env,jobject obj,jint value) { return value+1; }"
    (tmp_path / "a.md").write_text("```java\n" + java + "\n```\n```c\n" + native + "\n```\n")
    report = scan(tmp_path, include_docs=True, cross_language=True, min_nodes=5)
    assert not report.ffi_links
    assert report.stats["ffi_jni_declarations"] == 0


def test_astro_frontmatter_is_distinct_from_client_script(tmp_path):
    text = "---\n" + TS + "\n---\n<script>" + TS.replace(":number", "") + "</script>"
    for name in ("a.astro", "b.astro"):
        (tmp_path / name).write_text(text)
    report = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert report.stats["functions"] == 4
    assert {m.source_kind for f in report.families for m in f.members} == {"frontmatter", "script"}


def test_filter_without_matching_script_is_not_a_clean_audit(tmp_path):
    (tmp_path / "a.vue").write_text("<script>function f(x){return x+1;}</script>")
    report = scan(tmp_path, languages={"rust"}, min_nodes=5)
    assert not report.coverage_complete
    assert report.stats["selected_regions"] == 0


def test_python_valid_sibling_is_recovered_with_partial_status(tmp_path):
    clean = "def compute(x):\n value=x+1\n return value*3+4\n"
    (tmp_path / "a.py").write_text(clean)
    (tmp_path / "b.py").write_text(clean + "\ndef broken( { not valid")
    report = scan(tmp_path, min_nodes=5, near_metric="subtree", retain_source=True)
    assert not report.coverage_complete
    assert report.stats["parse_error_files"] == 1
    assert report.stats["recovered_units"] == 1
    assert any(f.tier == "exact" for f in report.families)
    assert all(m.source_code.startswith("def compute") for m in report.units)


def test_astro_nested_runtime_scripts_and_crlf_frontmatter(tmp_path):
    text = (
        "---\r\n" + TS + "\r\n---\r\n<div><script>" + TS.replace(":number", "") + "</script></div>"
    )
    for name in ("a.astro", "b.astro"):
        (tmp_path / name).write_bytes(text.encode())
    report = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert report.coverage_complete, report.stats
    assert report.stats["functions"] == 4
    assert {m.line for m in report.units} == {2, 4}


def test_ambiguous_script_attributes_are_partial(tmp_path):
    (tmp_path / "a.vue").write_text("<script lang='ts' lang='coffee'>" + TS + "</script>")
    report = scan(tmp_path, min_nodes=5)
    assert not report.coverage_complete
    assert report.stats["unsupported_regions"] == 1
