"""Reproducible capability probes; these controlled examples are not an accuracy benchmark.

Runs an installed jscpd binary against generated scratch sources, compares Victor
evidence, and records a JS/Python numeric counterexample. No model, network,
repository reindex, or production source execution is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any

from victor.core.graph_rag.structural_clones import report_to_dict, scan
from victor.core.graph_rag.language_parsers import LANGUAGE_MODULES, SOURCE_LANGUAGES
from victor_codegraph.embeddings import embedding_text


@dataclass(frozen=True)
class Probe:
    name: str
    files: dict[str, str]
    question: str
    portable: bool = False
    scip: bool = False
    include_docs: bool = False


PYTHON = """def compute(x, y):
    total = x - y
    adjusted = total * 3
    bounded = adjusted + 4
    if bounded > 10:
        return bounded + 2
    return bounded - 2
"""
RUST = "fn compute(x:i32,y:i32)->i32 { let total=x-y; let adjusted=total*3; let bounded=adjusted+4; if bounded>10 { return bounded+2; } bounded-2 }\n"
TYPESCRIPT = "function compute(x:number,y:number) { let total=x-y; let adjusted=total*3; let bounded=adjusted+4; if(bounded>10) { return bounded+2; } return bounded-2; }\n"


def probes() -> list[Probe]:
    renamed = re.sub(
        r"\b(x|y|total|adjusted|bounded|compute)\b",
        lambda m: {
            "x": "first",
            "y": "second",
            "total": "sum_value",
            "adjusted": "scaled",
            "bounded": "result",
            "compute": "calculate",
        }[m[0]],
        PYTHON,
    )
    shared = "\n".join(f"    value = value * {i} + {i+1}" for i in range(2, 12)) + "\n"
    left = "def first(x):\n    value = x\n" + shared + "    return value\n"
    right = (
        "def second(config):\n    values = []\n    for key in config:\n        values.append(config[key])\n    value = len(values)\n"
        + shared
        + "    while values:\n        values.pop()\n    return str(value)\n"
    )
    closure = "fn compute(xs: Vec<i32>) -> i32 { let values=xs.iter().map(|item| item+1).collect::<Vec<_>>(); let result=values.iter().sum(); result }\n"
    kotlin = "fun compute(x:Int,y:Int):Int { val total=x-y; val adjusted=total*3; val bounded=adjusted+4; if(bounded>10) { return bounded+2 }; return bounded-2 }\n"
    call = "def compute(key):\n    value = store.get(key)\n    adjusted = value * 3\n    return adjusted + 2\n"
    return [
        Probe("python_copy", {"a.py": PYTHON, "b.py": PYTHON}, "Verbatim source copy"),
        Probe(
            "python_alpha_rename",
            {"a.py": PYTHON, "b.py": renamed},
            "Consistent local/parameter renaming",
        ),
        Probe(
            "python_parameter_reversal",
            {"a.py": PYTHON, "b.py": PYTHON.replace("x - y", "y - x")},
            "Same parameter order, different subtraction behavior",
        ),
        Probe(
            "python_operator_change",
            {"a.py": PYTHON, "b.py": PYTHON.replace("x - y", "x + y")},
            "Preserve changed operators",
        ),
        Probe(
            "python_literal_change",
            {"a.py": PYTHON, "b.py": PYTHON.replace("> 10", "> 30")},
            "Normalized syntax can match changed behavior",
        ),
        Probe(
            "python_callee_change",
            {"a.py": call, "b.py": call.replace(".get(", ".delete(")},
            "Resolved/global callee distinction versus template pattern",
        ),
        Probe(
            "python_field_change",
            {
                "a.py": call.replace("store.get(key)", "key.left"),
                "b.py": call.replace("store.get(key)", "key.right"),
            },
            "Attribute identity versus normalized fields",
        ),
        Probe(
            "partial_copied_block",
            {"a.py": left, "b.py": right},
            "Token fragments inside structurally different whole functions",
        ),
        Probe(
            "same_file_copy",
            {"a.py": PYTHON + "\n" + renamed},
            "Default cross-file scope versus opt-in same-file matching",
        ),
        Probe(
            "rust_closure_rename",
            {"a.rs": closure, "b.rs": re.sub(r"\bitem\b", "entry", closure)},
            "Closure binder scope and generic calls",
            portable=True,
        ),
        Probe(
            "rust_parameter_reversal",
            {"a.rs": RUST, "b.rs": RUST.replace("x-y", "y-x")},
            "Parameter-order preservation in Rust",
        ),
        Probe(
            "rust_match_rename",
            {
                "a.rs": "fn compute(x:i32)->i32 { let value=match x { 0=>1, n=>n*3+2 }; value+4 }",
                "b.rs": "fn compute(arg:i32)->i32 { let result=match arg { 0=>1, item=>item*3+2 }; result+4 }",
            },
            "Match-arm binding and portable coverage",
            portable=True,
        ),
        Probe(
            "kotlin_copy",
            {"a.kt": kotlin, "b.kt": kotlin},
            "Grammar breadth without a SCIP provider",
        ),
        Probe(
            "vue_script_copy",
            {
                "a.vue": '<script lang="ts">\n'
                + TYPESCRIPT
                + "</script>\n<template><div /></template>",
                "b.vue": '<script lang="ts">\n'
                + TYPESCRIPT
                + "</script>\n<template><div /></template>",
            },
            "Embedded-language region extraction",
        ),
        Probe(
            "markdown_rust_copy",
            {"a.md": "```rust\n" + RUST + "```\n", "b.md": "```rust\n" + RUST + "```\n"},
            "Code in documentation blocks",
            include_docs=True,
        ),
        Probe(
            "malformed_rust",
            {"a.rs": "fn broken( { not valid", "b.rs": "fn broken( { not valid"},
            "An incomplete analysis must not claim a clean audit",
        ),
        Probe(
            "partial_parse_rust",
            {"a.rs": RUST + "\nfn broken( { not valid", "b.rs": RUST + "\nfn broken( { not valid"},
            "Recover valid functions while retaining a partial-file warning",
        ),
        Probe(
            "typescript_rust_port",
            {"a.ts": TYPESCRIPT, "b.rs": RUST},
            "Common portable operations across languages",
            portable=True,
        ),
        Probe(
            "python_js_numeric_domain",
            {
                "a.py": "def compute(x):\n    adjusted = x + 1\n    return adjusted * 2\n",
                "b.js": "function compute(x) { let adjusted=x+1; return adjusted*2; }\n",
            },
            "Identical portable operations can have different numeric domains",
            portable=True,
        ),
        Probe(
            "scip_different_targets",
            {"a.py": call, "b.py": call},
            "Same spelling, different compiler receiver targets",
            scip=True,
        ),
        Probe(
            "jni_abi_mismatch",
            {
                "Native.java": "package pkg; class Native { native int compute(int x); }",
                "native.cpp": "JNIEXPORT jlong JNICALL Java_pkg_Native_compute(JNIEnv *e, jobject o, jlong x) { return x+1; }",
            },
            "Name binding does not verify ABI compatibility",
            portable=True,
        ),
    ]


def _scip(root: Path, files: dict[str, str]) -> None:
    scip_pb2: Any = import_module("victor.core.graph_rag._scip.scip_pb2")

    index = scip_pb2.Index()
    for (path, source), target in zip(files.items(), ("Registry", "Cache")):
        doc = index.documents.add(relative_path=path, text=source, position_encoding=1)
        prefix = source[: source.index("get")]
        column = len(prefix.rsplit("\n", 1)[-1].encode())
        doc.occurrences.add(
            range=[prefix.count("\n"), column, column + 3],
            symbol=f"scip-python python pkg 1 `store`/{target}#get().",
        )
    directory = root / ".victor" / "precise"
    directory.mkdir(parents=True)
    (directory / "scip-python.scip").write_bytes(index.SerializeToString())


def _jscpd(binary: str, source: Path, output: Path, ast_mode: bool) -> dict[str, Any]:
    command = [
        binary,
        "--mode",
        "weak",
        "--min-tokens",
        "1" if ast_mode else "20",
        "--min-lines",
        "0" if ast_mode else "1",
        "--reporters",
        "json,edn",
        "--output",
        str(output),
        "--silent",
    ]
    if ast_mode:
        command += ["--similarity", "0.3", "--min-nodes", "5", "--kind", "ast"]
    command.append(str(source))
    completed = subprocess.run(
        command, cwd=source, capture_output=True, text=True, timeout=60, check=False
    )
    report = output / "jscpd-report.json"
    data = json.loads(report.read_text()) if report.exists() else {}
    edn = output / "jscpd-report.edn"
    scores = (
        [float(value) for value in re.findall(r":score\s+([0-9.eE+-]+)", edn.read_text())]
        if edn.exists()
        else []
    )
    return {
        "exit_code": completed.returncode,
        "analyzed_files": data.get("statistics", {}).get("total", {}).get("sources", 0),
        "reported_pairs": len(data.get("duplicates", [])),
        "ast_candidate_scores": scores,
        "ast_candidates_edn": edn.read_text() if ast_mode and edn.exists() else None,
        "stderr": completed.stderr.strip()[:500],
        "arguments": command[1:-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jscpd", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = str(Path(args.jscpd).resolve())
    version = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=True, timeout=15
    ).stdout.strip()
    implementation = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(inspect.getfile(scan)).parent.glob("*.py"))
        if path.name
        in {
            "structural_clones.py",
            "semantic_clones.py",
            "treesitter_clones.py",
            "portable_clones.py",
            "polyglot_clones.py",
            "ffi_bindings.py",
            "ast_fingerprints.py",
            "language_parsers.py",
            "clone_bindings.py",
            "source_regions.py",
            "source_reader.py",
            "numeric_literals.py",
            "body_clones.py",
            "token_clones.py",
            "similarity_analysis.py",
        }
    }
    results: list[dict[str, Any]] = []
    for case in probes():
        with tempfile.TemporaryDirectory(prefix="victor-parity-") as directory:
            work = Path(directory)
            root = work / "source"
            root.mkdir()
            for name, source in case.files.items():
                (root / name).write_text(source, encoding="utf-8")
            if case.scip:
                _scip(root, case.files)
            report = scan(
                root,
                min_nodes=5,
                near_metric="subtree",
                near_threshold=0.3,
                cross_language=case.portable,
                include_docs=case.include_docs,
            )
            payload = report_to_dict(report, 100)
            row: dict[str, Any] = {
                "name": case.name,
                "question": case.question,
                "corpus_hash": hashlib.sha256(
                    json.dumps(case.files, sort_keys=True).encode()
                ).hexdigest(),
                "victor": {
                    "families": [
                        {"tier": f["tier"], "score": f["score"]} for f in payload["families"]
                    ],
                    "coverage_complete": report.coverage_complete,
                    "stats": report.stats,
                    "ffi_links": payload["ffi_links"],
                },
                "jscpd_ast": _jscpd(binary, root, work / "ast", True),
                "jscpd_tokens": _jscpd(binary, root, work / "tokens", False),
            }
            if case.name == "same_file_copy":
                row["victor_same_file_families"] = len(
                    scan(root, min_nodes=5, near_metric="subtree", include_same_file=True).families
                )
            results.append(row)
    numeric: dict[str, Any] = {"input": 2**53, "python": (2**53 + 1) * 2}
    node = shutil.which("node")
    if node:
        numeric["javascript"] = json.loads(
            subprocess.run(
                [node, "-e", "console.log(JSON.stringify((9007199254740992+1)*2))"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
        )
    dependencies: dict[str, str | None] = {}
    for name in sorted(
        {
            "tree-sitter",
            "markdown-it-py",
            "victor-codegraph",
            *(module.replace("_", "-") for module, _ in LANGUAGE_MODULES.values()),
        }
    ):
        try:
            dependencies[name] = package_version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    result = {
        "schema": "victor-jscpd-capability-probes-v1",
        "accuracy_benchmark": False,
        "python": platform.python_version(),
        "dependency_versions": dependencies,
        "source_inventory": dict(sorted(SOURCE_LANGUAGES.items())),
        "jscpd": version,
        "victor_module": inspect.getfile(scan),
        "threshold_note": "AST candidate search 0.3; inspect scores at 0.8/1.0. Different normalizations and units are not calibrated equivalents.",
        "semantic_models_run": False,
        "summary_collision": {
            "input": embedding_text("compute", "(x: int) -> int", None),
            "body_a": "return x + 1",
            "body_b": "return x - 1",
            "outputs_at_3": [4, 2],
            "identical_summary_input": True,
            "inference": "A deterministic encoder receives identical inputs and cannot distinguish the different bodies.",
        },
        "implementation_hashes": implementation,
        "implementation_changed_during_run": any(
            hashlib.sha256((Path(inspect.getfile(scan)).parent / name).read_bytes()).hexdigest()
            != digest
            for name, digest in implementation.items()
        ),
        "numeric_counterexample": numeric,
        "probes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(
            row["name"],
            "Victor",
            row["victor"]["families"],
            "jscpd AST",
            row["jscpd_ast"]["ast_candidate_scores"],
            "token pairs",
            row["jscpd_tokens"]["reported_pairs"],
        )
    print("numeric_counterexample", numeric)


if __name__ == "__main__":
    main()
