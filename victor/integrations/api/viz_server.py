# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Standalone wrapper for the code-graph explorer (ADR-033 phase 1).

THIN: the explorer implementation lives in
``victor/integrations/api/routes/viz_routes.py`` and is mounted by the
main ``victor serve`` server. This wrapper exists for the zero-config
local case (``victor viz serve`` - no auth, no providers, read-only),
the same way ``victor serve hitl`` stands alone. One implementation,
two mounts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import FastAPI

from victor.integrations.api.routes.viz_routes import create_router


def create_viz_app(workspace_root: Optional[Path] = None) -> FastAPI:
    """Build the standalone explorer app for a workspace root."""
    root = (workspace_root or Path.cwd()).resolve()
    server_stub = SimpleNamespace(workspace_root=str(root))
    app = FastAPI(title="Victor Code Explorer", docs_url=None, redoc_url=None)
    app.include_router(create_router(server_stub))
    return app
