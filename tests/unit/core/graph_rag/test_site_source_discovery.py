"""Real file discovery and registry dispatch for ambiguous site source directories."""

from pathlib import Path

from victor.core.graph_rag.config import GraphIndexConfig
from victor.core.graph_rag.indexing import GraphIndexingPipeline
from victor.core.graph_rag.precise_providers import providers_for_files


def _pipeline(root: Path) -> GraphIndexingPipeline:
    pipeline = object.__new__(GraphIndexingPipeline)
    pipeline.config = GraphIndexConfig(root_path=root, enable_ccg=False, enable_embeddings=False)
    return pipeline


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


async def test_site_python_and_javascript_are_eligible_for_automatic_providers(tmp_path):
    for relative, source in {
        "site/hooks/plugin.py": "def render(): return 1\n",
        "site/assets/app.js": "export function render() { return 1; }\n",
        "src/Main.java": "class Main {}\n",
        "_site/generated.js": "export const generated = true;\n",
        "docs/_build/generated.py": "generated = True\n",
        "node_modules/dep/index.js": "export const dependency = true;\n",
    }.items():
        _write(tmp_path, relative, source)
    files = await _pipeline(tmp_path)._discover_files(tmp_path)
    relative = {path.relative_to(tmp_path).as_posix() for path in files}
    assert relative == {"site/hooks/plugin.py", "site/assets/app.js", "src/Main.java"}
    assert set(providers_for_files(sorted(relative))) == {
        "scip-python",
        "scip-typescript",
        "scip-java",
    }


async def test_project_gitignore_can_exclude_generated_site_without_hiding_nested_source(tmp_path):
    (tmp_path / ".gitignore").write_text("/site/\n")
    _write(tmp_path, "site/generated.js", "export const generated = true;\n")
    _write(tmp_path, "web/site/source.js", "export function render() { return 1; }\n")
    _write(tmp_path, "src/main.py", "def main(): return 1\n")
    files = await _pipeline(tmp_path)._discover_files(tmp_path)
    relative = {path.relative_to(tmp_path).as_posix() for path in files}
    assert relative == {"web/site/source.js", "src/main.py"}
