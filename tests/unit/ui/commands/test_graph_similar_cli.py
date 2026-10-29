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

"""CLI surface tests for `victor graph similar` (no store required)."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from victor.ui.commands.graph import graph_app

runner = CliRunner()


def _invoke(*args):
    return runner.invoke(graph_app, ["similar", *args])


def test_missing_store_hint(tmp_path: Path):
    result = _invoke("--path", str(tmp_path))
    assert result.exit_code == 1
    assert "victor graph index --embeddings" in result.output


def test_empty_scope_vs_missing_store_distinction(tmp_path: Path):
    store = tmp_path / ".victor" / "embeddings"
    store.mkdir(parents=True)
    with patch(
        "victor.core.graph_rag.semantic_clones.load_embedding_symbols",
        side_effect=FileNotFoundError(str(store)),
    ):
        result = _invoke("--path", str(tmp_path), "--scope", "victor/core/")
    assert result.exit_code == 1
    # scope-aware message, NOT the generic no-store one
    assert "not have reached" in " ".join(result.output.split())
    assert "index --embeddings" not in result.output.split("\n")[0]


def test_renders_families_from_the_loader(tmp_path: Path):
    import numpy as np

    from victor.core.graph_rag.semantic_clones import SymbolRecord

    records = [
        SymbolRecord("1", "pkg/a.py", "run_check", "semantic", 10),
        SymbolRecord("2", "pkg/b.py", "run_check", "semantic", 20),
    ]
    mat = np.array([[1.0, 0.0], [1.0, 0.001]], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    with patch(
        "victor.core.graph_rag.semantic_clones.load_embedding_symbols",
        return_value=(records, mat),
    ):
        result = _invoke("--path", str(tmp_path), "--threshold", "0.9")
    assert result.exit_code == 0
    assert "run_check" in result.output
    assert "pkg/a.py" in result.output and "pkg/b.py" in result.output


def test_json_output_shape(tmp_path: Path):
    import numpy as np

    from victor.core.graph_rag.semantic_clones import SymbolRecord

    records = [
        SymbolRecord("1", "pkg/a.py", "run_check", "semantic", 10),
        SymbolRecord("2", "pkg/b.py", "run_check", "semantic", 20),
    ]
    mat = np.array([[1.0, 0.0], [1.0, 0.001]], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    with patch(
        "victor.core.graph_rag.semantic_clones.load_embedding_symbols",
        return_value=(records, mat),
    ):
        result = _invoke("--path", str(tmp_path), "--json")
    assert result.exit_code == 0
    assert '"symbols_indexed": 2' in result.output
    assert '"similarity"' in result.output


def test_no_families_message(tmp_path: Path):
    import numpy as np

    from victor.core.graph_rag.semantic_clones import SymbolRecord

    records = [
        SymbolRecord("1", "pkg/a.py", "one", "semantic", 1),
        SymbolRecord("2", "pkg/b.py", "two", "semantic", 2),
    ]
    mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # orthogonal
    with patch(
        "victor.core.graph_rag.semantic_clones.load_embedding_symbols",
        return_value=(records, mat),
    ):
        result = _invoke("--path", str(tmp_path), "--threshold", "0.99")
    assert result.exit_code == 0
    assert "No semantic near-duplicates" in result.output


def test_too_many_pairs_is_a_clean_error(tmp_path: Path):
    from victor.core.graph_rag.semantic_clones import TooManyPairsError

    with (
        patch(
            "victor.core.graph_rag.semantic_clones.load_embedding_symbols",
            return_value=([], None),
        ),
        patch(
            "victor.core.graph_rag.semantic_clones.find_clone_families",
            side_effect=TooManyPairsError("threshold 0.5 matched over 5,000,000 pairs"),
        ),
    ):
        result = _invoke("--path", str(tmp_path), "--threshold", "0.5")
    assert result.exit_code == 1
    assert "--threshold 0.85" in " ".join(result.output.split())
