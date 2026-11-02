"""Installed-wheel smoke checks must reject stale headers and false identities."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as N

import pytest


def load():
    path = Path(__file__).resolve().parents[3] / "scripts/ci/check_native_ccg_wheel.py"
    spec = importlib.util.spec_from_file_location("native_ccg_wheel_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version,schema", [("old", 2), ("candidate", 1), ("candidate", None)])
def test_invalid_native_header_is_rejected_before_calling_parser(version, schema):
    def forbidden(*args):
        raise AssertionError("Rejected wheel must not reach parser calls")

    engine = N(__version__=version, CCG_STATEMENT_SCHEMA=schema, extract_statements_batch=forbidden)
    with pytest.raises(RuntimeError, match="schema mismatch"):
        load().check(engine, "candidate")


@pytest.mark.parametrize(
    "identities,reason",
    [
        (("same", "same", "same"), "collide"),
        (("one", "two", "three"), "reference"),
    ],
)
def test_source_identity_failures_cannot_pass_the_wheel_probe(identities, reason):
    records = [N(statement_type="call", node_id=value) for value in identities]
    engine = N(
        __version__="candidate",
        CCG_STATEMENT_SCHEMA=2,
        extract_statements_batch=lambda *args: [records],
    )
    with pytest.raises(RuntimeError, match=reason):
        load().check(engine, "candidate")
