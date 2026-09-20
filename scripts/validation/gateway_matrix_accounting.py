"""Strict buffered wire/SQLite/C4 joins for real gateway validation runs.

Only usage and correlation metadata are retained. This does not establish backend
executed cache reuse, tokenizer equivalence, streaming or origin cancellation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

COUNTS = ("tokens_in", "tokens_out", "cache_read_tokens", "cache_creation_tokens")
FIELDS = (
    "request_id",
    "session_id",
    "run_id",
    "step_id",
    "parent_id",
    "provider",
    "model",
    *COUNTS,
    "cache_read_status",
    "cache_read_source",
    "duration_ms",
    "duration_source",
)
BUFFERED_DEADLINE_SECONDS = 120


def diagnostic_errors(item: dict[str, Any], row: dict[str, Any]) -> list[str]:
    """One comparison for every C4 selector; all describe the same ledger row."""
    errors = []
    for field in ("request_id", "session_id", "run_id", "step_id", "provider", "model", *COUNTS):
        if item.get(field) != row[field] or (field in COUNTS and type(item.get(field)) is not int):
            errors.append("c4_" + field)
    observation = item.get("cache_read_observation")
    if not isinstance(observation, dict) or any(
        observation.get(key) != row["cache_read_" + key] for key in ("status", "source")
    ):
        errors.append("c4_cache_observation")
    if item.get("warnings"):
        errors.append("c4_warnings")
    return errors


def reconcile(
    wire: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    pages: dict[str, Any],
    *,
    expected_provider: str | None = None,
) -> dict[str, Any]:
    """Conserve inclusive prompt units and retain every failed physical HTTP attempt."""
    failures: list[dict[str, Any]] = []
    joins: list[dict[str, Any]] = []
    joined_ids: list[str] = []
    for request in wire:
        errors: list[str] = []
        entry = {"ordinal": request["ordinal"], "failures": errors}
        joins.append(entry)
        if request.get("http_status") != 200:
            errors.append("http_failure")
        if request.get("elapsed_seconds", BUFFERED_DEADLINE_SECONDS) >= BUFFERED_DEADLINE_SECONDS:
            errors.append("deadline_reached")
        matches = [
            row
            for row in rows
            if request["before_rowid"] < row["rowid"] <= request.get("after_rowid", -1)
            and all(
                row[field] == request[field]
                for field in ("session_id", "run_id", "step_id", "model")
            )
        ]
        if len(matches) != 1:
            errors.append("nonunique_request_join")
        else:
            row = matches[0]
            joined_ids.append(row["request_id"])
            entry.update(request_id=row["request_id"], sqlite=row, wire=request.get("usage"))
            if expected_provider is not None and row["provider"] != expected_provider:
                errors.append("provider_route")
            if any(type(row[field]) is not int or row[field] < 0 for field in COUNTS):
                errors.append("invalid_ledger_counter")
            if request.get("invalid_usage_fields"):
                errors.append("invalid_usage_fields")
            if (
                row["provider"] == "inferflux"
                and request.get("origin_request_id") != row["request_id"]
            ):
                errors.append("origin_correlation_mismatch")
            try:
                usage = request["usage"]
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if type(usage[field]) is not int or usage[field] < 0:
                        raise ValueError("Invalid token units")
                if (
                    usage["prompt_tokens"]
                    != row["tokens_in"] + row["cache_read_tokens"] + row["cache_creation_tokens"]
                ):
                    errors.append("input_conservation")
                if usage["completion_tokens"] != row["tokens_out"]:
                    errors.append("output_conservation")
                if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                    errors.append("total_conservation")
                if "cache_creation_input_tokens" in usage and (
                    type(usage["cache_creation_input_tokens"]) is not int
                    or usage["cache_creation_input_tokens"] != row["cache_creation_tokens"]
                ):
                    errors.append("cache_creation_conservation")
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
                if type(cached) is not int or cached != row["cache_read_tokens"]:
                    errors.append("explicit_cache_conservation")
            except (KeyError, TypeError, ValueError, AttributeError):
                errors.append("invalid_or_missing_usage")
            if (row["cache_read_status"], row["cache_read_source"]) != ("reported", "origin_usage"):
                errors.append("availability_coverage")
            page = pages.get("request:" + row["request_id"], {})
            diagnostics = page.get("rows", [])
            if page.get("truncated") is not False or len(diagnostics) != 1:
                errors.append("c4_request_join")
            else:
                errors.extend(diagnostic_errors(diagnostics[0], row))
        failures.extend({"ordinal": request["ordinal"], "check": error} for error in errors)
    if not wire or len(rows) != len(wire) or len(set(joined_ids)) != len(rows):
        failures.append({"check": "wire_ledger_call_count"})
    for session in {row["session_id"] for row in rows}:
        wanted = {row["request_id"] for row in rows if row["session_id"] == session}
        for kind in ("session", "run"):
            page = pages.get(kind + ":" + session, {})
            if (
                page.get("truncated") is not False
                or len(page.get("rows", [])) != len(wanted)
                or {row["request_id"] for row in page.get("rows", [])} != wanted
            ):
                failures.append({"check": "c4_set", "kind": kind, "session_id": session})
            else:
                by_id = {row["request_id"]: row for row in rows}
                for item in page["rows"]:
                    failures.extend(
                        {"check": error, "kind": kind, "session_id": session}
                        for error in diagnostic_errors(item, by_id[item["request_id"]])
                    )
    return {
        "failures": failures,
        "joins": joins,
        "totals": (
            {field: sum(row[field] for row in rows) for field in COUNTS}
            if all(type(row[field]) is int and row[field] >= 0 for row in rows for field in COUNTS)
            else None
        ),
    }


class GatewayObserver:
    """Read-only ledger observer; authentication is never written to evidence."""

    def __init__(
        self,
        gateway_url: str,
        admin: str,
        database: Path,
        output: Path,
        port: int,
        expected_provider: str,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self._admin = admin
        self.database = database
        self.output = output
        self.port = port
        self.expected_provider = expected_provider
        self.records: list[dict[str, Any]] = []
        self.client: ClientSession | None = None
        self.runner: web.AppRunner | None = None
        self.before: dict[str, Any] = {}
        self.baseline = 0

    def save(self, name: str, value: Any) -> None:
        temporary = self.output / (name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / name)

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        connection = sqlite3.connect("file:" + str(self.database) + "?mode=ro", uri=True)
        try:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(sql, parameters)]
        finally:
            connection.close()

    def last_row(self) -> int:
        return int(self.query("SELECT COALESCE(MAX(rowid),0) AS last FROM usage_events")[0]["last"])

    async def admin_call(self, path: str, payload: Any = None) -> Any:
        if self.client is None:
            raise RuntimeError("Observer is not started")
        async with self.client.request(
            "POST" if payload is not None else "GET",
            self.gateway_url + path,
            json=payload,
            headers={"Authorization": "Bearer " + self._admin},
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Admin API returned HTTP {response.status}")
            return await response.json()

    async def snapshot(self, name: str) -> dict[str, Any]:
        dashboard = await self.admin_call("/dashboard/api/usage")
        totals = self.query(
            "SELECT COUNT(*) AS calls,"
            + ",".join("COALESCE(SUM(" + field + "),0) AS " + field for field in COUNTS)
            + " FROM usage_events"
        )[0]
        result = {
            "dashboard": dashboard,
            "sqlite": totals,
            "passed": all(
                dashboard["total"][field] == totals[field] for field in ("calls", *COUNTS)
            ),
        }
        self.save("dashboard-" + name + ".json", result)
        return result

    async def start(self) -> None:
        self.client = ClientSession(timeout=ClientTimeout(total=300), auto_decompress=False)
        try:
            self.before = await self.snapshot("before")
            self.baseline = self.last_row()
            app = web.Application(client_max_size=16 * 1024 * 1024)
            app.router.add_route("*", "/{tail:.*}", self.forward)
            self.runner = web.AppRunner(app)
            await self.runner.setup()
            await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        except BaseException:
            await self.close()
            raise

    async def forward(self, request: web.Request) -> web.Response:
        if self.client is None:
            raise RuntimeError("Observer is not started")
        body = await request.read()
        record: dict[str, Any] | None = None
        started = time.monotonic()
        if request.path.endswith("/chat/completions"):
            payload = json.loads(body)
            record = {
                "ordinal": len(self.records),
                "model": payload.get("model"),
                "session_id": request.headers.get("x-sandhi-session"),
                "run_id": request.headers.get("x-sandhi-run-id"),
                "step_id": request.headers.get("x-sandhi-step-id"),
                "request_sha256": hashlib.sha256(body).hexdigest(),
                "before_rowid": self.last_row(),
                "stream": bool(payload.get("stream", False)),
            }
            self.records.append(record)
            self.save("wire.json", self.records)
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "transfer-encoding", "connection"}
        }
        headers["Accept-Encoding"] = "identity"
        try:
            async with self.client.request(
                request.method,
                self.gateway_url + request.path_qs,
                data=body,
                headers=headers,
            ) as response:
                output = await response.read()
                if record is not None:
                    record.update(
                        http_status=response.status,
                        after_rowid=self.last_row(),
                        origin_request_id=response.headers.get("x-inferflux-client-request-id"),
                        response_sha256=hashlib.sha256(output).hexdigest(),
                    )
                    try:
                        value = json.loads(output)
                        usage = value.get("usage") if isinstance(value, dict) else None
                        if isinstance(usage, dict):
                            projected: dict[str, Any] = {}
                            invalid = []
                            for field in (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                                "cache_creation_input_tokens",
                                "cache_read_input_tokens",
                            ):
                                if field in usage:
                                    if type(usage[field]) is int and usage[field] >= 0:
                                        projected[field] = usage[field]
                                    else:
                                        invalid.append(
                                            {
                                                "field": field,
                                                "value_type": type(usage[field]).__name__,
                                            }
                                        )
                            details = usage.get("prompt_tokens_details")
                            if isinstance(details, dict) and "cached_tokens" in details:
                                cached = details["cached_tokens"]
                                if type(cached) is int and cached >= 0:
                                    projected["prompt_tokens_details"] = {"cached_tokens": cached}
                                else:
                                    invalid.append(
                                        {
                                            "field": "cached_tokens",
                                            "value_type": type(cached).__name__,
                                        }
                                    )
                            record["usage"] = projected
                            if invalid:
                                record["invalid_usage_fields"] = invalid
                    except (ValueError, TypeError):
                        record["invalid_json"] = True
                return web.Response(
                    status=response.status,
                    body=output,
                    headers={
                        key: value
                        for key, value in response.headers.items()
                        if key.lower()
                        not in {
                            "content-length",
                            "transfer-encoding",
                            "connection",
                            "content-encoding",
                        }
                    },
                )
        except BaseException as exc:
            if record is not None:
                record["observer_error_type"] = type(exc).__name__
            raise
        finally:
            if record is not None:
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                self.save("wire.json", self.records)

    async def collect(self) -> dict[str, Any]:
        after = await self.snapshot("after")
        all_rows = self.query(
            "SELECT rowid," + ",".join(FIELDS) + " FROM usage_events WHERE rowid>? ORDER BY rowid",
            (self.baseline,),
        )
        sessions = {request["session_id"] for request in self.records}
        rows = [row for row in all_rows if row["session_id"] in sessions]
        self.save("sqlite-rows.json", rows)
        pages: dict[str, Any] = {}
        selectors = [("request", row["request_id"]) for row in rows]
        selectors += [
            (kind, session)
            for session in sessions
            if isinstance(session, str)
            for kind in ("session", "run")
        ]
        for kind, value in selectors:
            try:
                page = await self.admin_call(
                    "/admin/usage/diagnostics",
                    {
                        "selector": {"kind": kind, "value": value},
                        "limit": 500,
                    },
                )
            except Exception as exc:
                page = {"error_type": type(exc).__name__}
            pages[kind + ":" + value] = page
        self.save("c4.json", pages)
        result = reconcile(self.records, rows, pages, expected_provider=self.expected_provider)
        if not self.before["passed"] or not after["passed"]:
            result["failures"].append({"check": "dashboard_sqlite_snapshot"})
        for field in COUNTS:
            if after["sqlite"][field] - self.before["sqlite"][field] != sum(
                row[field] for row in all_rows
            ):
                result["failures"].append({"check": "dashboard_delta", "field": field})
        if after["sqlite"]["calls"] - self.before["sqlite"]["calls"] != len(all_rows):
            result["failures"].append({"check": "dashboard_call_delta"})
        result.update(unrelated_new_rows=len(all_rows) - len(rows), requests=len(self.records))
        self.save("accounting.json", result)
        return result

    async def close(self) -> None:
        try:
            if self.runner is not None:
                await self.runner.cleanup()
        finally:
            if self.client is not None:
                await self.client.close()
