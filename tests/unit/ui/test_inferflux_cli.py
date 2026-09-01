# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InferFlux CLI surfaces: models-list discovery + provider-aware check default."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from victor.providers.inferflux_provider import INFERFLUX_MODELS, InferfluxProvider
from victor.ui.commands.models import SUPPORTED_PROVIDERS
from victor.ui.commands.providers import _default_check_model


def test_inferflux_is_a_supported_models_list_provider():
    assert "inferflux" in SUPPORTED_PROVIDERS


def test_default_check_model_reads_the_policy_tier():
    assert _default_check_model("inferflux") in INFERFLUX_MODELS
    assert _default_check_model("deepseek") != "deepseek-chat"  # policy, not hardcode


def test_default_check_model_falls_back_for_policyless_providers():
    assert _default_check_model("ollama") == "deepseek-chat"


@pytest.mark.asyncio
async def test_list_inferflux_models_live_discovery_with_fallback(capsys):
    """Live /v1/models when the server answers; static policy tier when it doesn't."""
    from victor.ui.commands.models import _list_inferflux_models

    served = json.dumps(
        {"object": "list", "data": [{"id": "tinyllama", "object": "model"}]}
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = served
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()

    class _Settings:
        def get_provider_settings(self, name):
            return {}

    try:
        await _list_inferflux_models(_Settings(), f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
    out = capsys.readouterr().out
    assert "tinyllama" in out

    # Offline: static tier.
    class _Dead:
        def get_provider_settings(self, name):
            return {"base_url": "http://127.0.0.1:1"}

    await _list_inferflux_models(_Dead())
    out2 = capsys.readouterr().out
    assert next(iter(INFERFLUX_MODELS)) in out2
    assert "unreachable" in out2


def test_inferflux_provider_row_exists():
    from victor.ui.commands.providers import _list_providers_impl  # noqa: F401
    from victor.ui.commands import providers as providers_mod

    import inspect

    src = inspect.getsource(providers_mod)
    assert '"inferflux"' in src and "Batched decode" in src
