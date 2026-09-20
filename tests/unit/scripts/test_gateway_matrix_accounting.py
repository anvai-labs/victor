"""Accounting fault injection; never represented as actual-member evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "scripts/validation/gateway_matrix_accounting.py"
_SPEC = importlib.util.spec_from_file_location("gateway_matrix_accounting", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
accounting = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(accounting)


@pytest.fixture
def joined():
    wire = {
        "ordinal": 0,
        "session_id": "member-1",
        "run_id": "member-1",
        "step_id": None,
        "model": "qwen3-coder-30b",
        "before_rowid": 10,
        "after_rowid": 11,
        "http_status": 200,
        "elapsed_seconds": 2,
        "origin_request_id": "req-1",
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 2,
            "total_tokens": 17,
            "prompt_tokens_details": {"cached_tokens": 5},
        },
    }
    row = {
        "rowid": 11,
        "request_id": "req-1",
        "session_id": "member-1",
        "run_id": "member-1",
        "step_id": None,
        "model": "qwen3-coder-30b",
        "provider": "inferflux",
        "tokens_in": 10,
        "tokens_out": 2,
        "cache_read_tokens": 5,
        "cache_creation_tokens": 0,
        "cache_read_status": "reported",
        "cache_read_source": "origin_usage",
    }
    page = {
        "truncated": False,
        "rows": [
            {
                **row,
                "warnings": [],
                "cache_read_observation": {"status": "reported", "source": "origin_usage"},
            }
        ],
    }
    pages = {
        key: copy.deepcopy(page) for key in ("request:req-1", "session:member-1", "run:member-1")
    }
    return wire, row, pages


def test_wire_sqlite_c4_conservation_with_explicit_cache(joined):
    wire, row, pages = joined
    result = accounting.reconcile([wire], [row], pages)
    assert result["failures"] == []
    assert result["joins"][0]["request_id"] == "req-1"
    assert result["totals"]["cache_read_tokens"] == 5


@pytest.mark.parametrize(
    "fault",
    [
        "cache_missing",
        "cache_wrong",
        "input",
        "output",
        "total",
        "timeout",
        "http",
        "correlation",
        "c4_truncated",
        "extra_row",
    ],
)
def test_failed_attempts_or_accounting_are_never_hidden(joined, fault):
    wire, row, pages = copy.deepcopy(joined)
    rows = [row]
    if fault == "cache_missing":
        wire["usage"].pop("prompt_tokens_details")
    elif fault == "cache_wrong":
        wire["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    elif fault in {"input", "output", "total"}:
        field = {"input": "prompt_tokens", "output": "completion_tokens", "total": "total_tokens"}[
            fault
        ]
        wire["usage"][field] += 1
    elif fault == "timeout":
        wire["elapsed_seconds"] = 120
    elif fault == "http":
        wire["http_status"] = 504
    elif fault == "correlation":
        wire["origin_request_id"] = "wrong"
    elif fault == "c4_truncated":
        pages["request:req-1"]["truncated"] = True
    else:
        rows.append({**row, "rowid": 12, "request_id": "req-2"})
    result = accounting.reconcile([wire], rows, pages)
    assert result["failures"]


@pytest.mark.parametrize("fault", [None, "timeout", "counter_secret"])
async def test_forward_records_only_usage_metadata_and_keeps_failed_attempts(tmp_path, fault):
    import asyncio
    import json
    from unittest.mock import AsyncMock, Mock

    observer = accounting.GatewayObserver(
        "http://gateway", "fixture-admin", tmp_path / "unused.db", tmp_path, 18084, "inferflux"
    )
    observer.last_row = Mock(side_effect=[0, 1])
    secret = "fixture-secret-do-not-retain"
    response = AsyncMock()
    response.status = 200
    response.headers = {"x-inferflux-client-request-id": "req-1"}
    response.read.return_value = json.dumps(
        {
            "choices": [{"message": {"content": secret}}],
            "usage": {
                "prompt_tokens": secret if fault == "counter_secret" else 15,
                "completion_tokens": 2,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 5, "untrusted_extra": secret},
            },
        }
    ).encode()
    context = AsyncMock()
    context.__aenter__.return_value = response
    if fault == "timeout":
        context.__aenter__.side_effect = asyncio.TimeoutError()
    observer.client = Mock()
    observer.client.request.return_value = context
    request = Mock()
    request.path = request.path_qs = "/v1/chat/completions"
    request.method = "POST"
    request.headers = {
        "Authorization": "Bearer " + secret,
        "x-sandhi-session": "session",
        "x-sandhi-run-id": "session",
    }
    request.read = AsyncMock(
        return_value=json.dumps({"model": "reference", "messages": [{"content": secret}]}).encode()
    )
    if fault == "timeout":
        with pytest.raises(asyncio.TimeoutError):
            await observer.forward(request)
    else:
        assert (await observer.forward(request)).status == 200
    observer.client.request.assert_called_once()
    assert (
        observer.client.request.call_args.kwargs["headers"]["Authorization"] == "Bearer " + secret
    )
    wire = (tmp_path / "wire.json").read_text()
    assert secret not in wire
    assert len(json.loads(wire)) == 1
    if fault == "timeout":
        assert observer.records[0]["observer_error_type"] == "TimeoutError"
    else:
        assert observer.records[0]["usage"]["prompt_tokens_details"] == {"cached_tokens": 5}


@pytest.mark.parametrize(
    "kind,field",
    [
        ("request", "step_id"),
        ("request", "provider"),
        ("session", "tokens_out"),
        ("run", "tokens_in"),
        ("session", "cache_read_observation"),
    ],
)
def test_every_c4_selector_must_agree_with_the_ledger(joined, kind, field):
    wire, row, pages = joined
    key = kind + (":req-1" if kind == "request" else ":member-1")
    pages[key]["rows"][0][field] = "contradiction"
    assert accounting.reconcile([wire], [row], pages)["failures"]


def test_requested_provider_is_part_of_acceptance(joined):
    wire, row, pages = joined
    assert accounting.reconcile([wire], [row], pages, expected_provider="zai")["failures"]


@pytest.mark.parametrize("fault", ["negative", "cache_creation"])
def test_compensating_or_misreported_counters_are_rejected(joined, fault):
    wire, row, pages = joined
    if fault == "negative":
        row.update(tokens_in=-1, cache_read_tokens=16)
        wire["usage"]["prompt_tokens_details"]["cached_tokens"] = 16
    else:
        row.update(tokens_in=9, cache_creation_tokens=1)
        wire["usage"]["cache_creation_input_tokens"] = 99
    for page in pages.values():
        page["rows"][0].update({field: row[field] for field in accounting.COUNTS})
    assert accounting.reconcile([wire], [row], pages)["failures"]
