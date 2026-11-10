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
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Surface the subprocess's own traceback — a CalledProcessError alone
        # hides the import error this guard exists to diagnose.
        raise AssertionError(
            f"blocked-import subprocess failed (rc={result.returncode}):\n"
            f"{result.stderr[-2000:]}"
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


_BLOCK_NUMPY = """
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in {"numpy", "sentence_transformers"}:
            raise ImportError(f"No module named {name!r} (blocked)")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
"""


def test_bare_install_constructs_collections_and_intent_classifier_without_numpy(tmp_path):
    """Agent creation builds the IntentClassifier singleton unconditionally, so
    constructing its collections must not import numpy (bare-install regression,
    victor-ai 0.27.0: every `victor chat` failed with "No module named 'numpy'")."""
    output = _run_blocked_import_script(_BLOCK_NUMPY + f"""
    from pathlib import Path

    class StubEmbeddings:
        dimension = 4
        model_name = "stub"

    from victor.storage.embeddings.collections import StaticEmbeddingCollection
    from victor.storage.embeddings.intent_classifier import IntentClassifier

    collection = StaticEmbeddingCollection(
        "bare", cache_dir=Path({str(tmp_path)!r}), embedding_service=StubEmbeddings()
    )
    assert not collection.is_initialized
    IntentClassifier(cache_dir=Path({str(tmp_path)!r}), embedding_service=StubEmbeddings())
    print("BARE_CONSTRUCT_OK")
    """)

    assert "BARE_CONSTRUCT_OK" in output


def test_bare_install_still_fails_loudly_on_vector_math_without_numpy(tmp_path):
    """Deferring the import must not hide it: the first vector operation still
    raises the numpy ImportError instead of silently producing empty state."""
    output = _run_blocked_import_script(_BLOCK_NUMPY + f"""
    from pathlib import Path

    class StubEmbeddings:
        dimension = 4
        model_name = "stub"

    from victor.storage.embeddings.collections import StaticEmbeddingCollection

    collection = StaticEmbeddingCollection(
        "bare", cache_dir=Path({str(tmp_path)!r}), embedding_service=StubEmbeddings()
    )
    try:
        collection.initialize_sync([])
    except ImportError as exc:
        assert "numpy" in str(exc), exc
        print("BARE_MATH_RAISES")
    else:
        raise AssertionError("initialize_sync succeeded without numpy")
    assert collection._embeddings is None
    """)

    assert "BARE_MATH_RAISES" in output
