"""scip-java toolchain preflight, scratch builds and full dump transcripts.

Observed on a real monorepo index: the provider ran an unsupported JDK and a
Gradle newer than the project's wrapper pin, compiled inside the indexed tree,
and kept 400 characters of the failure. Each behavior below pins the fix.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import victor.core.graph_rag.precise_providers as mod
from victor.core.graph_rag.precise_providers import DumpSkipped, ScipJavaProvider

JV = ScipJavaProvider()


def _executable(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _gradle_project(root: Path, *, pinned: str = "8.14.6", wrapper_jar: bool = False) -> Path:
    (root / "gradle" / "wrapper").mkdir(parents=True)
    (root / "gradle" / "wrapper" / "gradle-wrapper.properties").write_text(
        f"distributionUrl=https\\://services.gradle.org/distributions/gradle-{pinned}-bin.zip\n",
        encoding="utf-8",
    )
    (root / "settings.gradle").write_text("rootProject.name = 'demo'\n", encoding="utf-8")
    _executable(root, "gradlew")
    if wrapper_jar:
        (root / "gradle" / "wrapper" / "gradle-wrapper.jar").write_bytes(b"PK")
    return root


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    """A JDK home with a fake java; PATH holds nothing else."""
    jdk = tmp_path / "jdk"
    _executable(jdk / "bin", "java")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.delenv("JAVA_HOME", raising=False)  # hermetic: hosts and CI often set it
    monkeypatch.setattr(mod, "_java_major", lambda java, env: 21)
    return jdk


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_run_dump(command, *, cwd, output_path, timeout_seconds, node_heap, env=None):
        calls.append(SimpleNamespace(command=command, cwd=cwd, env=env, output=output_path))
        return output_path

    monkeypatch.setattr(mod, "_run_dump", fake_run_dump)
    return calls


def test_missing_jdk_skips_before_any_build(tmp_path, monkeypatch, captured):
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.delenv("JAVA_HOME", raising=False)
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    result = JV.dump("/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1)
    assert result == DumpSkipped("no-jdk") and captured == []


@pytest.mark.parametrize("major", [8, 11, 23])
def test_unsupported_jdk_majors_skip_with_the_major_named(
    tmp_path, monkeypatch, toolchain, captured, major
):
    monkeypatch.setattr(mod, "_java_major", lambda java, env: major)
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    result = JV.dump(
        "/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1, java_home=str(toolchain)
    )
    assert result == DumpSkipped(f"unsupported-jdk:{major}") and captured == []


def test_java_home_leads_path_and_is_exported_to_the_build(tmp_path, toolchain, captured):
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    JV.dump(
        "/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1, java_home=str(toolchain)
    )
    env = captured[0].env
    assert env["JAVA_HOME"] == str(toolchain)
    assert env["PATH"].split(":")[0] == str(toolchain / "bin")


def test_a_gradle_newer_than_the_wrapper_pin_is_never_run(
    tmp_path, monkeypatch, toolchain, captured
):
    gradle_bin = tmp_path / "gradle-9" / "bin"
    _executable(gradle_bin, "gradle")
    monkeypatch.setenv("PATH", str(gradle_bin))
    monkeypatch.setattr(mod, "_gradle_version", lambda gradle, env: "9.3.1")
    root = _gradle_project(tmp_path / "project", pinned="8.14.6", wrapper_jar=False)
    result = JV.dump(
        "/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1, java_home=str(toolchain)
    )
    assert result == DumpSkipped("gradle-version-mismatch:wants-8.14.6-found-9.3.1")
    assert captured == []


def test_a_missing_gradle_is_reported_against_the_pin(tmp_path, toolchain, captured):
    root = _gradle_project(tmp_path / "project", pinned="8.14.6", wrapper_jar=False)
    result = JV.dump(
        "/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1, java_home=str(toolchain)
    )
    assert result == DumpSkipped("gradle-version-mismatch:wants-8.14.6-found-none")


def test_the_pinned_gradle_on_path_proceeds(tmp_path, monkeypatch, toolchain, captured):
    gradle_bin = tmp_path / "gradle-8" / "bin"
    _executable(gradle_bin, "gradle")
    monkeypatch.setenv("PATH", str(gradle_bin))
    monkeypatch.setattr(mod, "_gradle_version", lambda gradle, env: "8.14.6")
    root = _gradle_project(tmp_path / "project", pinned="8.14.6", wrapper_jar=False)
    JV.dump(
        "/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1, java_home=str(toolchain)
    )
    assert len(captured) == 1


def test_a_usable_wrapper_owns_the_gradle_version(tmp_path, monkeypatch, toolchain, captured):
    monkeypatch.setattr(
        mod, "_gradle_version", lambda *a: pytest.fail("the wrapper pins Gradle itself")
    )
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    JV.dump(
        "/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1, java_home=str(toolchain)
    )
    assert len(captured) == 1


def test_build_args_scope_the_compile_after_a_separator(tmp_path, toolchain, captured):
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    args = [":iceberg-core:compileJava", "-DsparkVersions=", "-x", "test"]
    JV.dump(
        "/opt/scip-java",
        root,
        tmp_path / "out.scip",
        timeout_seconds=1,
        java_home=str(toolchain),
        build_args=args,
    )
    command = captured[0].command
    assert command[:3] == ["/opt/scip-java", "index", f"--output={tmp_path / 'out.scip'}"]
    assert command[command.index("--") + 1 :] == args


def test_the_build_runs_in_a_scratch_copy_and_never_writes_the_indexed_tree(
    tmp_path, monkeypatch, toolchain
):
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    (root / ".victor").mkdir()
    (root / ".victor" / "project.db").write_bytes(b"index state")
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    seen = {}

    def building_dump(command, *, cwd, output_path, timeout_seconds, node_heap, env=None):
        seen["cwd"] = cwd
        seen["copied_sources"] = (cwd / "settings.gradle").is_file()
        seen["index_state_copied"] = (cwd / ".victor").exists()
        (cwd / ".gradle" / "9").mkdir(parents=True)
        (cwd / "build" / "reports").mkdir(parents=True)
        output_path.write_bytes(b"scip")
        return output_path

    monkeypatch.setattr(mod, "_run_dump", building_dump)
    out = tmp_path / "precise" / "out.scip"
    assert JV.dump("/opt/scip-java", root, out, timeout_seconds=1, java_home=str(toolchain)) == out
    assert seen["cwd"] != root and seen["copied_sources"] and not seen["index_state_copied"]
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before
    assert not seen["cwd"].exists()  # scratch is cleaned up


def test_dump_transcripts_are_kept_whole(tmp_path):
    message = "x" * 3000
    script = f"import sys; sys.stderr.write({message!r}); sys.exit(1)"
    out = tmp_path / "precise" / "out.scip"
    result = mod._run_dump(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        output_path=out,
        timeout_seconds=60,
        node_heap=False,
    )
    assert result is None
    log = (out.parent / f"{Path(sys.executable).name}.dump.log").read_text(encoding="utf-8")
    assert message in log and "returncode: 1" in log


@pytest.mark.parametrize(
    ("banner", "major"),
    [
        ('openjdk version "21.0.13" 2026-10-20 LTS', 21),
        ('java version "1.8.0_292"', 8),
        ('openjdk version "23.0.1-internal" 2024-10-15', 23),
        ('openjdk version "25" 2025-09-16', 25),
        ("not a jdk", None),
    ],
)
def test_java_major_parsing(monkeypatch, banner, major):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=banner),
    )
    assert mod._java_major("java", {}) == major


def test_wrapper_pin_parsing(tmp_path):
    assert mod._wrapper_gradle_version(_gradle_project(tmp_path / "a", pinned="8.14.6")) == "8.14.6"
    assert mod._wrapper_gradle_version(tmp_path / "missing") is None


def test_pipeline_passes_and_cache_keys_java_toolchain_settings():
    from victor.core.graph_rag.indexing import GraphIndexingPipeline

    config = SimpleNamespace(
        precise_index_java_home="/opt/jdk-21",
        precise_index_build_args={"scip-java": [":core:compileJava"], "scip-python": ["x"]},
    )
    holder = SimpleNamespace(config=config)
    kwargs = GraphIndexingPipeline._precise_toolchain_kwargs(holder, "scip-java")
    assert kwargs == {"java_home": "/opt/jdk-21", "build_args": [":core:compileJava"]}
    assert GraphIndexingPipeline._precise_toolchain_kwargs(holder, "scip-python") == {}


# --- Verified build root: compile a real checkout of the indexed sources ----


def _index_and_build_roots(tmp_path):
    index_root = _gradle_project(tmp_path / "index", wrapper_jar=True)
    (index_root / "src").mkdir()
    (index_root / "src" / "A.java").write_text("class A {}\n", encoding="utf-8")
    (index_root / ".victor").mkdir()
    (index_root / ".victor" / "project.db").write_bytes(b"index state")
    build_root = tmp_path / "checkout"
    import shutil as _shutil

    _shutil.copytree(index_root, build_root, ignore=_shutil.ignore_patterns(".victor"))
    (build_root / ".git").mkdir()
    (build_root / ".git" / "HEAD").write_text("2ebe838e\n", encoding="utf-8")
    (build_root / "excluded-from-snapshot.bin").write_bytes(b"\x00binary")
    return index_root, build_root


def test_a_matching_build_root_builds_in_place(tmp_path, monkeypatch, toolchain):
    index_root, build_root = _index_and_build_roots(tmp_path)
    seen = {}

    def building_dump(command, *, cwd, output_path, timeout_seconds, node_heap, env=None):
        seen["cwd"] = cwd
        seen["has_git"] = (cwd / ".git" / "HEAD").is_file()
        return output_path

    monkeypatch.setattr(mod, "_run_dump", building_dump)
    out = tmp_path / "out.scip"
    result = JV.dump(
        "/opt/scip-java",
        index_root,
        out,
        timeout_seconds=1,
        java_home=str(toolchain),
        build_root=str(build_root),
    )
    assert result == out
    assert seen == {"cwd": build_root, "has_git": True}


def test_a_build_root_module_resolves_files_outside_the_indexed_tree(
    tmp_path, monkeypatch, toolchain
):
    """A Maven module's parent pom (../pom.xml) or a Gradle includeBuild sits
    outside the indexed tree; building the build root in place keeps it."""
    checkout = tmp_path / "checkout"
    module = checkout / "module-a"
    module.mkdir(parents=True)
    (checkout / "pom.xml").write_text("<project><modules/></project>", encoding="utf-8")
    (module / "pom.xml").write_text("<project><parent/></project>", encoding="utf-8")
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "pom.xml").write_text("<project><parent/></project>", encoding="utf-8")
    seen = {}

    def building_dump(command, *, cwd, output_path, timeout_seconds, node_heap, env=None):
        seen["parent_pom"] = (cwd.parent / "pom.xml").is_file()
        return output_path

    monkeypatch.setattr(mod, "_run_dump", building_dump)
    JV.dump(
        "/opt/scip-java",
        index_root,
        tmp_path / "out.scip",
        timeout_seconds=1,
        java_home=str(toolchain),
        build_root=str(module),
    )
    assert seen == {"parent_pom": True}


@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_a_build_root_that_differs_from_the_index_never_builds(
    tmp_path, monkeypatch, toolchain, captured, damage
):
    index_root, build_root = _index_and_build_roots(tmp_path)
    target = build_root / "src" / "A.java"
    if damage == "changed":
        target.write_text("class A { int drift; }\n", encoding="utf-8")
    else:
        target.unlink()
    result = JV.dump(
        "/opt/scip-java",
        index_root,
        tmp_path / "out.scip",
        timeout_seconds=1,
        java_home=str(toolchain),
        build_root=str(build_root),
    )
    assert result == DumpSkipped("build-root-mismatch:1") and captured == []


def test_build_root_verification_ignores_index_state_and_allows_extras(tmp_path):
    index_root, build_root = _index_and_build_roots(tmp_path)
    assert mod._build_root_mismatches(index_root, build_root) == []
    assert mod._build_root_mismatches(index_root, tmp_path / "absent") == ["<build root missing>"]


def test_pipeline_passes_the_build_root_to_scip_java_only():
    from victor.core.graph_rag.indexing import GraphIndexingPipeline

    holder = SimpleNamespace(
        config=SimpleNamespace(precise_index_build_root="/srv/checkout@2ebe838e")
    )
    assert GraphIndexingPipeline._precise_toolchain_kwargs(holder, "scip-java") == {
        "build_root": "/srv/checkout@2ebe838e"
    }
    assert GraphIndexingPipeline._precise_toolchain_kwargs(holder, "scip-python") == {}


def test_a_symlinked_indexed_file_is_verified_by_its_content(tmp_path):
    """The indexer follows file symlinks; verification must compare the bytes
    it read, never skip the path."""
    index_root, build_root = _index_and_build_roots(tmp_path)
    (index_root / "ext").mkdir()
    (index_root / "ext" / "Foo.java").write_text("class Foo { int a; }\n", encoding="utf-8")
    (index_root / "src" / "Foo.java").symlink_to(index_root / "ext" / "Foo.java")
    (build_root / "ext").mkdir()
    (build_root / "ext" / "Foo.java").write_text("class Foo { int a; }\n", encoding="utf-8")
    (build_root / "src" / "Foo.java").write_text("class Foo { EVIL }\n", encoding="utf-8")
    assert mod._build_root_mismatches(index_root, build_root) == ["src/Foo.java"]


def test_a_same_size_content_change_is_a_mismatch(tmp_path):
    index_root, build_root = _index_and_build_roots(tmp_path)
    (build_root / "src" / "A.java").write_text("class B {}\n", encoding="utf-8")  # same size
    assert mod._build_root_mismatches(index_root, build_root) == ["src/A.java"]


def test_identical_content_through_a_build_root_symlink_is_accepted(tmp_path):
    index_root, build_root = _index_and_build_roots(tmp_path)
    target = build_root / "src" / "A.java"
    real = build_root / "real-A.java"
    real.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(real)  # repositories legitimately carry symlinks
    assert mod._build_root_mismatches(index_root, build_root) == []


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX non-root")
def test_an_unreadable_indexed_file_is_a_named_mismatch_not_an_error(tmp_path):
    index_root, build_root = _index_and_build_roots(tmp_path)
    locked = build_root / "src" / "A.java"
    locked.chmod(0)
    try:
        assert mod._build_root_mismatches(index_root, build_root) == ["src/A.java"]
    finally:
        locked.chmod(0o644)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fifo")
def test_a_scratch_copy_failure_is_a_named_skip(tmp_path, toolchain, captured):
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    os.mkfifo(root / "pipe")
    result = JV.dump(
        "/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1, java_home=str(toolchain)
    )
    assert isinstance(result, DumpSkipped) and result.reason.startswith("scratch-copy-failed:")
    assert captured == []


def test_the_jdk_checked_is_the_java_home_the_build_uses(tmp_path, monkeypatch, captured):
    """JAVA_HOME wins over PATH for scip-java's launcher, Gradle and Maven."""
    jdk17 = tmp_path / "jdk17"
    _executable(jdk17 / "bin", "java")
    path_bin = tmp_path / "path-bin"
    _executable(path_bin, "java")
    monkeypatch.setenv("PATH", str(path_bin))
    monkeypatch.setenv("JAVA_HOME", str(jdk17))
    monkeypatch.setattr(mod, "_java_major", lambda java, env: 17 if "jdk17" in java else 11)
    root = _gradle_project(tmp_path / "project", wrapper_jar=True)
    JV.dump("/opt/scip-java", root, tmp_path / "out.scip", timeout_seconds=1)
    assert len(captured) == 1


def test_cache_key_keeps_the_tool_stat_and_keys_the_toolchain(tmp_path):
    from victor.core.graph_rag.precise_cache import dump_cache_key

    tool = tmp_path / "scip-java"
    tool.write_bytes(b"v1")
    toolchain = {"java_home": "/opt/jdk-21", "build_args": [":core:compileJava"]}
    before = dump_cache_key("scip-java", str(tool), {"A.java": "h"}, toolchain=toolchain)
    tool.write_bytes(b"upgraded indexer v2")
    after_upgrade = dump_cache_key("scip-java", str(tool), {"A.java": "h"}, toolchain=toolchain)
    assert before != after_upgrade
    other_scope = dump_cache_key(
        "scip-java", str(tool), {"A.java": "h"}, toolchain={**toolchain, "build_args": []}
    )
    assert other_scope != after_upgrade
    assert dump_cache_key("scip-java", str(tool), {"A.java": "h"}) != after_upgrade
