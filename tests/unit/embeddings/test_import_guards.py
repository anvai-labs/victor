"""Regression tests for optional embedding dependency import guards."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[3]


def _run_blocked_import_script(script: str) -> str:
    """Run an isolated Python process with numpy-like imports blocked."""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", dedent(script)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_embeddings_package_lazy_exports_do_not_eagerly_import_optional_dependencies():
    output = _run_blocked_import_script("""
        import builtins
        import sys

        real_import = builtins.__import__
        blocked = {"numpy", "sentence_transformers"}

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded

        import victor.storage.embeddings as embeddings

        assert "victor.storage.embeddings.service" not in sys.modules
        assert "victor.storage.embeddings.collections" not in sys.modules

        _ = embeddings.EmbeddingService
        _ = embeddings.StaticEmbeddingCollection

        assert "victor.storage.embeddings.service" in sys.modules
        assert "victor.storage.embeddings.collections" in sys.modules
        print("ok-embeddings")
        """)

    assert "ok-embeddings" in output


def test_semantic_selector_import_does_not_eagerly_import_embedding_service():
    output = _run_blocked_import_script("""
        import builtins
        import sys

        real_import = builtins.__import__
        blocked = {"numpy", "sentence_transformers"}

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded

        import victor.tools.semantic_selector as selector

        assert "victor.storage.embeddings.service" not in sys.modules
        assert selector.EmbeddingService is None
        print("ok-selector")
        """)

    assert "ok-selector" in output


def test_orchestrator_import_succeeds_without_optional_embedding_dependencies():
    output = _run_blocked_import_script("""
        import builtins

        real_import = builtins.__import__
        blocked = {"numpy", "sentence_transformers"}

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] in blocked:
                raise ImportError(f"blocked {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded

        import victor.agent.orchestrator

        print("ok-orchestrator")
        """)

    assert "ok-orchestrator" in output
