# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The graph tool and index shim must honor `.victor/graph_backend`.

Previously: the `graph` tool probed project.db's sqlite tables (dropping the
tool from the LLM schema on marker'd repos whose graph lives elsewhere) and
hardcoded SqliteGraphStore; the indexer accepted `graph_store_name` and
silently dropped it; the shim defaulted to sqlite. A repo flipped to another
backend got sqlite everywhere.
"""

from pathlib import Path

from victor_coding.tools.graph_tool import (
    _project_graph_has_data,
    _resolved_graph_backend,
)


def _mark(tmp_path: Path, backend: str) -> Path:
    marker_dir = tmp_path / ".victor"
    marker_dir.mkdir(exist_ok=True)
    (marker_dir / "graph_backend").write_text(backend)
    return tmp_path


def test_resolved_backend_reads_marker(tmp_path: Path):
    assert _resolved_graph_backend(tmp_path) == "sqlite"
    _mark(tmp_path, "proxima")
    assert _resolved_graph_backend(tmp_path) == "proxima"


def test_has_data_advertises_tool_on_marked_repos(tmp_path: Path):
    # A proxima-marked repo must not be probed via project.db's sqlite
    # tables (which are legitimately empty there) — the tool stays
    # advertised and the load path constructs the real store.
    _mark(tmp_path, "proxima")
    assert _project_graph_has_data(tmp_path) is True


def test_shim_default_is_auto():
    import inspect

    from victor_coding.codebase.graph.registry import create_graph_store

    sig = inspect.signature(create_graph_store)
    assert sig.parameters["name"].default == "auto"


def test_indexer_threads_graph_store_name():
    import inspect

    from victor_coding.codebase.indexer import CodebaseIndex

    src = inspect.getsource(CodebaseIndex.__init__)
    assert (
        'name=graph_store_name or "auto"' in src
    ), "graph_store_name must reach create_graph_store, not be dropped"
