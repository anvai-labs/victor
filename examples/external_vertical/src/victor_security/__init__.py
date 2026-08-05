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

"""Contract-only external security vertical example."""

from victor_contracts import PluginContext, VictorPlugin

from victor_security.assistant import SecurityAssistant
from victor_security.mode_config import SecurityModeConfigProvider
from victor_security.tools import SecretPatternScanTool

__version__ = "0.3.0"


class SecurityPlugin(VictorPlugin):
    """VictorPlugin wrapper for the contract-only security example."""

    @property
    def name(self) -> str:
        return "security"

    def register(self, context: PluginContext) -> None:
        context.register_vertical(SecurityAssistant)
        context.register_tool(SecretPatternScanTool())

    def get_cli_app(self):
        return None

    def on_activate(self) -> None:
        return None

    def on_deactivate(self) -> None:
        return None

    async def on_activate_async(self) -> None:
        return None

    async def on_deactivate_async(self) -> None:
        return None

    def health_check(self) -> dict[str, object]:
        return {"healthy": True, "vertical": "security"}


plugin = SecurityPlugin()

__all__ = [
    "SecretPatternScanTool",
    "SecurityAssistant",
    "SecurityModeConfigProvider",
    "SecurityPlugin",
    "plugin",
]
