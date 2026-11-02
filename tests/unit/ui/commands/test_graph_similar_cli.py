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

from typer.testing import CliRunner

from victor.ui.commands.graph import graph_app

runner = CliRunner()


def _invoke(*args):
    return runner.invoke(graph_app, ["similar", *args])


def test_token_mode_missing_backend_keeps_json_failure(tmp_path):
    import json

    result = _invoke(
        "--path",
        str(tmp_path),
        "--mode",
        "tokens",
        "--jscpd-binary",
        "victor-missing-jscpd",
        "--json",
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert not payload["coverage_complete"]
    assert "Install jscpd" in payload["error"]


def test_token_sarif_missing_backend_is_unsuccessful_execution(tmp_path):
    import json

    result = _invoke(
        "--path",
        str(tmp_path),
        "--mode",
        "tokens",
        "--sarif",
        "--jscpd-binary",
        "victor-missing-jscpd",
    )
    assert result.exit_code == 1
    run = json.loads(result.output)["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert not run["results"]
    assert (
        "Install jscpd" in run["invocations"][0]["toolExecutionNotifications"][0]["message"]["text"]
    )


def test_structural_sarif_exports_real_source_families(tmp_path):
    import json

    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("def f(x):\n value=x+1\n return value*3+4\n")
    result = _invoke("--path", str(tmp_path), "--mode", "structural", "--min-nodes", "5", "--sarif")
    assert result.exit_code == 0, result.output
    run = json.loads(result.output)["runs"][0]
    assert run["invocations"][0]["executionSuccessful"]
    assert run["results"]
    assert run["artifacts"][0]["hashes"]["sha-256"]
    assert all(row["properties"]["channel"] == "structural" for row in run["results"])


def test_body_sarif_preserves_failure_as_machine_readable_execution(tmp_path):
    import json

    with patch(
        "victor.core.graph_rag.similarity_analysis.analyze_source",
        side_effect=OSError("missing weights"),
    ):
        result = _invoke("--path", str(tmp_path), "--mode", "body", "--sarif")
    assert result.exit_code == 1
    run = json.loads(result.output)["runs"][0]
    assert not run["invocations"][0]["executionSuccessful"]
    assert "missing weights" in str(run["invocations"])


def test_sarif_stdout_uses_the_bounded_encoding_without_pretty_expansion(tmp_path, monkeypatch):
    import json
    from rich.console import Console
    from victor.core.graph_rag import similarity_sarif, structural_clones
    from victor.ui.commands import graph

    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("def f(x):\n value=x+1\n return value*3+4\n")
    original = structural_clones.report_to_dict

    def diagnostic(*args, **kwargs):
        raw = original(*args, **kwargs)
        nested = list(range(300))
        for _ in range(40):
            nested = [nested]
        raw["stats"]["diagnostic"] = nested
        return raw

    monkeypatch.setattr(structural_clones, "report_to_dict", diagnostic)
    monkeypatch.setattr(similarity_sarif, "_MAX_BYTES", 12000)
    monkeypatch.setattr(graph, "console", Console(force_terminal=True, width=20))
    result = _invoke("--path", str(tmp_path), "--mode", "structural", "--min-nodes", "5", "--sarif")
    assert result.exit_code == 0, result.output
    assert len(result.stdout.encode()) <= 12000
    assert json.loads(result.stdout)["runs"][0]["invocations"][0]["executionSuccessful"]


def test_summary_sarif_is_index_evidence_and_retains_component_members(tmp_path):
    import json
    import numpy as np
    from victor.core.graph_rag.semantic_clones import SymbolRecord

    records = [SymbolRecord(str(i), f"{i}.py", "f", "semantic", i) for i in range(3)]
    theta = np.array([0, 0.2, 0.45])
    matrix = np.column_stack((np.cos(theta), np.sin(theta)))
    with patch(
        "victor.core.graph_rag.semantic_clones.load_embedding_symbols",
        return_value=(records, matrix),
    ):
        result = _invoke("--path", str(tmp_path), "--threshold", "0.95", "--sarif")
    assert result.exit_code == 0, result.output
    run = json.loads(result.output)["runs"][0]
    assert run["properties"]["source_freshness"] == "unverified_index_snapshot"
    assert len(run["results"]) == 1
    assert run["results"][0]["properties"]["member_count"] == 3
    assert run["results"][0]["properties"]["matching_pairs"] == 2


def test_missing_summary_index_and_structural_failure_emit_sarif(tmp_path):
    import json

    result = _invoke("--path", str(tmp_path), "--sarif")
    assert result.exit_code == 1
    assert not json.loads(result.output)["runs"][0]["invocations"][0]["executionSuccessful"]
    with patch(
        "victor.core.graph_rag.structural_clones.scan", side_effect=OSError("source inaccessible")
    ):
        result = _invoke("--path", str(tmp_path), "--mode", "structural", "--sarif")
    assert result.exit_code == 1
    assert not json.loads(result.output)["runs"][0]["invocations"][0]["executionSuccessful"]


def test_copy_workflow_rejects_incompatible_cli_options_before_scanning(tmp_path):
    with patch(
        "victor.core.graph_rag.similarity_analysis.analyze_source",
        side_effect=AssertionError("must not scan"),
    ):
        cases = [
            ["--mode", "semantic", "--token-memory-mib", "192"],
            ["--mode", "hybrid", "--token-exclude", "alias.py"],
            ["--mode", "semantic", "--baseline", str(tmp_path / "copies.json")],
            ["--mode", "tokens", "--fail-on-new"],
            ["--mode", "tokens", "--changed-file", "a.py"],
            ["--mode", "tokens", "--sarif", "--json"],
            [
                "--mode",
                "tokens",
                "--write-baseline",
                str(tmp_path / "out.json"),
                "--baseline",
                str(tmp_path / "copies.json"),
            ],
        ]
        for args in cases:
            result = _invoke("--path", str(tmp_path), *args)
            assert result.exit_code == 2, result.output


def test_embedded_docs_flag_reaches_structural_scan(tmp_path):
    import json

    source = "```rust\nfn compute(x:i32)->i32 { let value=x+1; value*3+4 }\n```\n"
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(source)
    result = _invoke(
        "--path",
        str(tmp_path),
        "--mode",
        "structural",
        "--include-docs",
        "--min-nodes",
        "5",
        "--json",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["families"]


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
    import json
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
    assert '"equivalence_verified": false' in result.output
    assert '"similarity_basis": "maximum_matching_edge"' in result.output
    family = json.loads(result.output)["families"][0]
    assert family["cohesion"]["representative_id"] in {m["id"] for m in family["members"]}
    assert family["cohesion"]["scope"] == "all_component_member_pairs"
    assert not family["cohesion"]["pairwise_minimum_verified"]


def test_summary_human_output_exposes_a_direct_weak_pair(tmp_path):
    import numpy as np
    from victor.core.graph_rag.semantic_clones import SymbolRecord

    records = [SymbolRecord(str(i), f"{i}.py", "f", "semantic", 1) for i in range(3)]
    theta = np.array([0, 0.2, 0.45])
    matrix = np.column_stack((np.cos(theta), np.sin(theta)))
    with patch(
        "victor.core.graph_rag.semantic_clones.load_embedding_symbols",
        return_value=(records, matrix),
    ):
        result = _invoke("--path", str(tmp_path), "--threshold", "0.95")
    assert result.exit_code == 0, result.output
    assert "representative: 1.py:1" in result.output
    assert "pair witness: 0.py:1" in result.output
    assert "0.900" in result.output
    assert "not a verified pairwise minimum" in result.output


def test_body_human_output_resolves_cohesion_ids_to_source(tmp_path):
    from victor.core.graph_rag.body_clones import BodyOptions, LocalBodyEncoder, analyze_bodies
    from victor.core.graph_rag.structural_clones import scan

    class Encoder(LocalBodyEncoder):
        def windows(self, source):
            return [source], [(0, len(source))]

        def encode(self, texts):
            return [[1, 0, 0] for _ in texts]

    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("def f(x):\n value=x+1\n return value*3\n")
    options = BodyOptions(model_id="test/model", revision="0" * 40, dimension=3)

    channel = analyze_bodies(
        scan(tmp_path, min_nodes=5, retain_source=True),
        options=options,
        encoder=Encoder(options),
    )
    payload = {"coverage_complete": True, "channels": {"body": channel}}
    with patch("victor.core.graph_rag.similarity_analysis.analyze_source", return_value=payload):
        result = _invoke("--path", str(tmp_path), "--mode", "body")
    assert result.exit_code == 0, result.output
    assert "representative:" in result.output
    assert "pair witness:" in result.output
    assert ".py:1" in result.output


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
    assert "No summary-similarity candidates" in result.output


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
    assert "Raise --threshold" in " ".join(result.output.split())


# --- --mode structural ------------------------------------------------------

_COPY = """
def total(items, scale):
    acc = 0
    for item in items:
        if item.enabled and item.weight > 0:
            acc += item.weight * scale
        else:
            acc -= 1
    return {"total": round(acc, 2), "count": len(items)}
"""


def _structural_repo(tmp_path: Path) -> Path:
    import subprocess

    (tmp_path / "a.py").write_text(_COPY)
    (tmp_path / "b.py").write_text(_COPY.replace("items", "rows").replace("acc", "running"))
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_structural_mode_needs_no_embedding_store(tmp_path: Path):
    result = _invoke(
        "--path", str(_structural_repo(tmp_path)), "--mode", "structural", "--min-nodes", "10"
    )
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "exact 1" in out and "a.py" in out and "b.py" in out
    assert "No SCIP dump" in out  # the name-only fallback is announced, not silent


def test_structural_json_shape(tmp_path: Path):
    import json

    result = _invoke(
        "--path",
        str(_structural_repo(tmp_path)),
        "--mode",
        "structural",
        "--min-nodes",
        "10",
        "--json",
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["stats"]["exact_families"] == 1
    (family,) = data["families"]
    assert family["tier"] == "exact" and family["files"] == ["a.py", "b.py"]
    assert {"file", "symbol", "line", "size"} <= set(family["members"][0])


def test_subtree_metric_is_available_through_cli(tmp_path):
    result = _invoke(
        "--path",
        str(_structural_repo(tmp_path)),
        "--mode",
        "structural",
        "--min-nodes",
        "10",
        "--ast-metric",
        "subtree",
        "--json",
    )
    assert result.exit_code == 0, result.output
    import json

    assert json.loads(result.output)["stats"]["near_metric"] == "subtree"


def test_invalid_subtree_metric_is_clean_error(tmp_path):
    result = _invoke("--path", str(tmp_path), "--mode", "structural", "--ast-metric", "typo")
    assert result.exit_code == 1
    assert "Unknown AST metric" in result.output


def test_cross_language_cli_detects_typescript_and_rust_automatically(tmp_path):
    import json

    (tmp_path / "reference.ts").write_text(
        "function f(x:number,y:number) { let sum=x+y; return sum*2; }"
    )
    (tmp_path / "native.rs").write_text("fn f(a:i32,b:i32)->i32 { let value=a+b; value*2 }")
    result = _invoke(
        "--path",
        str(tmp_path),
        "--mode",
        "structural",
        "--cross-language",
        "--min-nodes",
        "5",
        "--json",
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["stats"]["cross_language_families"] == 1
    assert data["families"][0]["evidence"] == "portable_operation_shape"
    assert data["equivalence_verified"] is False


def test_jni_relationship_prints_even_without_duplicate_bodies(tmp_path):
    (tmp_path / "Native.java").write_text("package pkg; class Native { native int sum(int x); }")
    (tmp_path / "native.cpp").write_text(
        "JNIEXPORT jint JNICALL Java_pkg_Native_sum(JNIEnv *e, jobject o, jint x) { return x+1; }"
    )
    result = _invoke(
        "--path", str(tmp_path), "--mode", "structural", "--cross-language", "--min-nodes", "5"
    )
    assert result.exit_code == 0, result.output
    assert "Java_pkg_Native_sum" in result.output
    assert "jni_declaration_export" in result.output
    assert "matching_declaration" in result.output
    assert "ABI unverified" in result.output


def test_language_filter_and_incomplete_coverage_are_machine_readable(tmp_path):
    import json

    (tmp_path / "broken.rs").write_text("fn broken( { invalid")
    result = _invoke(
        "--path", str(tmp_path), "--mode", "structural", "--language", "rust", "--json"
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["coverage_complete"] is False
    result = _invoke("--path", str(tmp_path), "--language", "rust")
    assert result.exit_code == 2
    assert "require --mode structural" in result.output


def test_structural_no_families_message(tmp_path: Path):
    import subprocess

    (tmp_path / "a.py").write_text(_COPY)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    result = _invoke("--path", str(tmp_path), "--mode", "structural", "--min-nodes", "10")
    assert result.exit_code == 0
    assert "No structural near-duplicates" in result.output


def test_unsupported_scope_cannot_exit_as_a_clean_scan(tmp_path):
    import json

    (tmp_path / "only.sql").write_text("SELECT 1")
    result = _invoke("--path", str(tmp_path), "--mode", "structural", "--json")
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["stats"]["source_files"] == 0
    assert payload["coverage_complete"] is False


def test_structural_budget_is_a_clean_error(tmp_path: Path):
    from victor.core.graph_rag.semantic_clones import TooManyPairsError

    with patch(
        "victor.core.graph_rag.structural_clones.scan", side_effect=TooManyPairsError("too many")
    ):
        result = _invoke("--path", str(tmp_path), "--mode", "structural")
    assert result.exit_code == 1 and "too many" in result.output


def test_unknown_mode_is_rejected(tmp_path: Path):
    result = _invoke("--path", str(tmp_path), "--mode", "fuzzy")
    assert result.exit_code == 2 and "Unknown --mode" in result.output
