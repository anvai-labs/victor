"""Shared fixture server for the sandhi transport parity battery (FEP-0020 Phase 4b, T5).

respx cannot intercept the Rust reqwest client inside the sandhi binding, so parity runs
both transports against a REAL localhost HTTP server (threaded stdlib ``HTTPServer``) that
serves configured fixture responses and records every request for request-side parity.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import pytest


@dataclass
class Recorded:
    path: str
    headers: Dict[str, str]
    body: bytes


@dataclass
class FixtureServer:
    """One configured response per server instance; records all requests."""

    status: int = 200
    body: bytes = b"{}"
    content_type: str = "application/json"
    delay_secs: float = 0.0
    requests: List[Recorded] = field(default_factory=list)
    url: str = ""
    _httpd: Optional[HTTPServer] = None
    _thread: Optional[threading.Thread] = None

    def start(self) -> "FixtureServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — stdlib naming
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                server.requests.append(
                    Recorded(
                        path=self.path,
                        headers={k.lower(): v for k, v in self.headers.items()},
                        body=raw,
                    )
                )
                if server.delay_secs:
                    time.sleep(server.delay_secs)
                self.send_response(server.status)
                self.send_header("Content-Type", server.content_type)
                self.send_header("Content-Length", str(len(server.body)))
                self.end_headers()
                self.wfile.write(server.body)

            def log_message(self, *args: Any) -> None:  # silence
                pass

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


@pytest.fixture
def fixture_server():
    servers: List[FixtureServer] = []

    def _make(**kwargs: Any) -> FixtureServer:
        server = FixtureServer(**kwargs).start()
        servers.append(server)
        return server

    yield _make
    for server in servers:
        server.stop()


@pytest.fixture(params=["deepseek", "inferflux", "zai"])
def make_pair(request):
    """Factory fixture: two independent typed handles for determinism checks."""

    def _make(server_url: str, timeout: int = 30) -> Tuple[Any, Any]:
        from victor.providers.deepseek_provider import DeepSeekProvider
        from victor.providers.inferflux_provider import InferfluxProvider
        from victor.providers.zai_provider import ZAIProvider

        provider_cls = {
            "deepseek": DeepSeekProvider,
            "inferflux": InferfluxProvider,
            "zai": ZAIProvider,
        }[request.param]

        kwargs = {
            "api_key": "parity-key",
            "base_url": f"{server_url}/v1",
            "timeout": timeout,
            "max_retries": 0,
        }
        return provider_cls(**kwargs), provider_cls(**kwargs)

    return _make
