"""Qualify an installed SDK wheel, including downstream static rejection.

Run after installing the wheel and MyPy. Temporary consumers are outside the
source checkout so editable/source resolution cannot stand in for wheel typing.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import victor_contracts


def main() -> None:
    distribution = importlib.metadata.distribution("victor-contracts")
    installed_init = Path(str(distribution.locate_file("victor_contracts/__init__.py"))).resolve()
    if Path(victor_contracts.__file__).resolve() != installed_init:
        raise RuntimeError("SDK import must resolve to the installed wheel, not editable source")
    for package in ("victor_contracts", "victor_sdk"):
        if not Path(str(distribution.locate_file(f"{package}/py.typed"))).is_file():
            raise RuntimeError(f"Installed wheel is missing {package}/py.typed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_schema = importlib.import_module("victor_sdk.team_schema")
    schema = importlib.import_module("victor_contracts.team_schema")
    if legacy_schema is not schema:
        raise RuntimeError("Legacy submodule must preserve the canonical SDK module identity")

    valid = """from victor_contracts.team_schema import MemoryConfig, TeamMemberSpec
from victor_contracts.handler_runtime import handler_decorator
from victor_sdk import TeamMemberSpec as LegacyMemberSpec

@handler_decorator("sample")
class Handler:
    pass

handler: Handler = Handler()
member = TeamMemberSpec(
    role="researcher", goal="Inspect", memory_config=MemoryConfig(enabled=False)
)
legacy: TeamMemberSpec = LegacyMemberSpec(role="reviewer", goal="Review")
"""
    with tempfile.TemporaryDirectory(prefix="victor-contracts-typing-") as directory:
        root = Path(directory)
        config = root / "mypy.ini"
        config.write_text("[mypy]\nstrict = true\nfollow_imports = silent\n")
        consumer = root / "consumer.py"

        def check(source: str) -> subprocess.CompletedProcess[str]:
            consumer.write_text(source)
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--config-file",
                    str(config),
                    "--python-executable",
                    sys.executable,
                    str(consumer),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=120,
            )

        positive = check(valid)
        if positive.returncode != 0:
            raise RuntimeError(positive.stdout + positive.stderr)
        negative = check(valid + '\nmember.goal = 42\nMemoryConfig(enabled="yes")\n')
        if negative.returncode != 1 or not all(
            diagnostic in negative.stdout for diagnostic in ("[assignment]", "[arg-type]")
        ):
            raise RuntimeError(
                "Invalid consumer was not rejected: " + negative.stdout + negative.stderr
            )
    print(
        json.dumps(
            {
                "version": distribution.version,
                "wheel_markers": 2,
                "valid": True,
                "invalid_rejected": True,
            }
        )
    )


if __name__ == "__main__":
    main()
