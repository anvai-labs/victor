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

"""Victor presentation defaults without modifying workspace or deployment files."""

from pathlib import Path
from typing import TYPE_CHECKING

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

if TYPE_CHECKING:
    from chainlit.config import UISettings
    from fastapi import FastAPI

_ASSETS = Path(__file__).with_name("assets")
_MOUNT = "/victor-assets"


def configure_presentation(app: "FastAPI", ui: "UISettings", *, root_path: str = "") -> None:
    """Mount only packaged assets and retain explicit deployment customizations."""
    # Chainlit's SPA catch-all already exists when its app module is imported.
    # Register this narrow mount ahead of it, once (including watch reloads).
    if not any(isinstance(route, Mount) and route.path == _MOUNT for route in app.routes):
        app.router.routes.insert(
            0, Mount(_MOUNT, app=StaticFiles(directory=_ASSETS), name="victor-assets")
        )
    url = root_path.rstrip("/") + _MOUNT
    if ui.name == "Assistant":
        ui.name = "Victor"
    if not ui.description:
        ui.description = "Your workspace for code, tools and agent workflows."
    if not ui.custom_css:
        ui.custom_css = url + "/workspace.css"
    if not ui.logo_file_url:
        ui.logo_file_url = url + "/victor.svg"
    if not ui.default_avatar_file_url:
        ui.default_avatar_file_url = url + "/victor.svg"
    if ui.layout == "default":
        ui.layout = "wide"
