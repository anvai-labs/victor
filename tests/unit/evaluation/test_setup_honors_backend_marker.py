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

"""The benchmark setup indexing path must honor `.victor/graph_backend`.

`setup_repo_with_indexes` previously read only the global
`codebase_graph_store` setting, so the per-repo marker — whose entire
purpose is flipping one repo's backend without changing global settings —
was silently ignored by benchmark/eval indexing.
"""

from pathlib import Path

from victor.storage.graph.registry import resolve_graph_backend


def test_marker_overrides_global_default(tmp_path: Path):
    marker_dir = tmp_path / ".victor"
    marker_dir.mkdir(exist_ok=True)
    (marker_dir / "graph_backend").write_text("proxima\n")
    assert resolve_graph_backend(tmp_path, default="sqlite") == "proxima"


def test_missing_marker_falls_back_to_supplied_default(tmp_path: Path):
    assert resolve_graph_backend(tmp_path, default="sqlite") == "sqlite"
    assert resolve_graph_backend(tmp_path, default="proxima") == "proxima"


def test_setup_path_resolves_via_marker(tmp_path: Path, monkeypatch):
    # The loader body computes graph_store_name via resolve_graph_backend on
    # the cache path; simulate exactly that call pattern.
    cache_path = tmp_path
    (cache_path / ".victor").mkdir(exist_ok=True)
    (cache_path / ".victor" / "graph_backend").write_text("proxima")
    global_default = "sqlite"
    assert (
        resolve_graph_backend(cache_path, default=global_default) == "proxima"
    ), "benchmark setup must index with the repo's chosen backend"
