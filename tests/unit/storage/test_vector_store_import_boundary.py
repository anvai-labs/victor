"""Optional vector backends must not execute during registry/search imports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "victor.storage.vector_stores",
        "victor.storage.vector_stores.lancedb_provider",
        "victor.framework.search.codebase_embedding_bridge",
    ],
)
def test_import_does_not_execute_installed_lancedb(tmp_path: Path, module: str) -> None:
    # The package is discoverable but fails if executed, as an incompatible native
    # wheel would. Run in a fresh process so cached imports cannot hide the defect.
    (tmp_path / "lancedb.py").write_text(
        'raise RuntimeError("optional native backend executed during import")\n'
    )
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(root), str(root / "victor-contracts"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; "
            f"importlib.import_module({module!r}); "
            "assert 'lancedb' not in sys.modules",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
