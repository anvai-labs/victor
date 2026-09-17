from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_dependency_locks.py"
    spec = importlib.util.spec_from_file_location("check_dependency_locks", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_dependency_locks = _load_module()


def _write_repository(root: Path, *, core_pin: str = "2.5.0", fsspec_pin: str = "2.0") -> None:
    (root / "requirements/api").mkdir(parents=True)
    (root / "requirements/embeddings-cpu").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[project]
name = "fixture"
version = "1.0.0"
dependencies = ["core>=2,<3"]

[project.optional-dependencies]
api = ["api-lib>=1"]
embeddings = ["fsspec>=1"]
""".strip() + "\n",
        encoding="utf-8",
    )
    (root / "constraints.txt").write_text("fsspec<3\n", encoding="utf-8")
    (root / "requirements.txt").write_text(f"core=={core_pin}\n", encoding="utf-8")
    (root / "requirements/api/requirements.txt").write_text(
        f"core=={core_pin}\napi-lib==1.2\n", encoding="utf-8"
    )
    (root / "requirements/embeddings-cpu/requirements.txt").write_text(
        f"core=={core_pin}\nfsspec=={fsspec_pin}\n", encoding="utf-8"
    )


def test_consistent_locks_pass(tmp_path: Path) -> None:
    _write_repository(tmp_path)

    assert check_dependency_locks.check_locks(tmp_path) == []


def test_constraint_violation_fails(tmp_path: Path) -> None:
    _write_repository(tmp_path, fsspec_pin="3.0")

    failures = check_dependency_locks.check_locks(tmp_path)

    assert any("fsspec==3.0 violates constraint <3" in failure for failure in failures)


def test_project_requirement_violation_fails(tmp_path: Path) -> None:
    _write_repository(tmp_path, core_pin="1.9")

    failures = check_dependency_locks.check_locks(tmp_path)

    assert any("core==1.9 violates project requirement <3,>=2" in failure for failure in failures)
