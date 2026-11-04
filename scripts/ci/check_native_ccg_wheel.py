"""Exercise installed native CCG schema, byte identities and UTF-8 without core extras."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tomllib
from typing import Any


def check(engine: Any, expected_version: str) -> dict[str, Any]:
    if engine.__version__ != expected_version or engine.CCG_STATEMENT_SCHEMA != 2:
        raise RuntimeError("Native wheel version/CCG schema mismatch")
    source = b"def f():\n    return outer(inner(), sibling())\n"
    records = engine.extract_statements_batch([("python", "calls.py", source)])[0]
    calls = [record for record in records if record.statement_type == "call"]
    if len(calls) != 3 or len({record.node_id for record in calls}) != 3:
        raise RuntimeError("Same-line native call identities collide")
    expected = {
        hashlib.sha256(f"ccg-statement-v2:calls.py:{start}:{end}:call".encode()).hexdigest()[:16]
        for start, end in ((20, 45), (26, 33), (35, 44))
    }
    if {record.node_id for record in calls} != expected:
        raise RuntimeError("Native byte-span identity does not match the reference")
    text = "def f():\n    return '" + "🔥" * 110 + "'\n"
    unicode_records = engine.extract_statements_batch([("python", "unicode.py", text.encode())])[0]
    returns = [record for record in unicode_records if record.statement_type == "return"]
    if len(returns) != 1 or returns[0].content_preview != text.splitlines()[1].strip()[:100]:
        raise RuntimeError("Native UTF-8 preview differs from Python character slicing")
    result = engine.build_ccg_batch([("python", "calls.py", source)])[0]
    if result.cdg_edges or not result.statements or not result.cfg_edges:
        raise RuntimeError("Native CCG inputs contain legacy CDG or missing graph records")
    return {
        "schema": "victor-native-ccg-wheel-check-v1",
        "native_version": engine.__version__,
        "statement_schema": engine.CCG_STATEMENT_SCHEMA,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "same_line_calls": len(calls),
        "unicode_preview_verified": True,
        "program_semantics_verified": False,
    }


def main() -> None:
    import victor_native

    metadata = Path(__file__).resolve().parents[2] / "rust/pyproject.toml"
    expected = tomllib.loads(metadata.read_text())["project"]["version"]
    if importlib.metadata.version("victor-native") != expected:
        raise RuntimeError("Installed wheel distribution metadata does not match source")
    print(json.dumps(check(victor_native, expected), indent=2))


if __name__ == "__main__":
    main()
