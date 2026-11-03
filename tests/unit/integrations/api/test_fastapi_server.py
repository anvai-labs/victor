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

"""Unit tests for victor/integrations/api/fastapi_server.py.

This module provides basic import and class-existence tests for the FastAPI server.
Comprehensive integration tests are in tests/integration/api/ and other test modules
in this directory (test_api_auth_boundary.py, test_fastapi_chat_wire_stream.py, etc.).
"""

from __future__ import annotations

import pytest


def test_fastapi_server_import() -> None:
    """Verify fastapi_server module imports successfully."""
    from victor.integrations.api import fastapi_server

    assert fastapi_server is not None


def test_victor_fastapi_server_class_exists() -> None:
    """Verify VictorFastAPIServer class is defined."""
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

    assert VictorFastAPIServer is not None


def test_victor_fastapi_server_requires_auth_or_explicit_opt_in() -> None:
    """VictorFastAPIServer should raise without api_keys or allow_unauthenticated."""
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

    with pytest.raises(ValueError, match="api_keys.*allow_unauthenticated"):
        VictorFastAPIServer()


def test_victor_fastapi_server_signature_has_allow_unauthenticated() -> None:
    """VictorFastAPIServer.__init__ should accept allow_unauthenticated parameter."""
    import inspect

    from victor.integrations.api.fastapi_server import VictorFastAPIServer

    sig = inspect.signature(VictorFastAPIServer.__init__)
    params = sig.parameters
    assert "allow_unauthenticated" in params
    assert params["allow_unauthenticated"].default is None
