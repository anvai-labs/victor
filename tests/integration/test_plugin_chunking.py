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

import unittest
from unittest.mock import MagicMock, patch

from victor.core.chunking.registry import ChunkingRegistry
from victor.core.chunking.base import Chunk, ChunkingStrategy
from victor.core.plugins.protocol import VictorPlugin


class MockChunkingStrategy(ChunkingStrategy):
    @property
    def name(self) -> str:
        return "mock_strategy"

    @property
    def supported_types(self) -> list[str]:
        return ["mocktype"]

    def chunk(self, content: str) -> list[Chunk]:
        return [Chunk(content="mocked", start_char=0, end_char=6, chunk_type="mock")]


class MockChunkingPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "chunkplugin"

    def register(self, container) -> None:
        from victor.core.chunking.registry import get_chunking_registry

        registry = get_chunking_registry()
        registry.register(MockChunkingStrategy())

    def get_cli_app(self):
        return None


class TestPluginChunkingIntegration(unittest.TestCase):
    @patch("victor.core.plugins.registry.PluginRegistry.discover")
    def test_chunking_registry_uses_plugin_strategy(self, mock_discover):
        # Reset registry for test
        from victor.core.chunking.registry import _default_registry
        import victor.core.chunking.registry as registry_mod

        registry_mod._default_registry = None

        # Setup mock plugin
        plugin = MockChunkingPlugin()
        mock_discover.return_value = [plugin]

        # In a real app, bootstrap calls register_all()
        from victor.core.plugins.registry import PluginRegistry

        PluginRegistry.get_instance().register_all(MagicMock())

        # Get registry and verify it has the mock strategy
        registry = registry_mod.get_chunking_registry()
        self.assertIn("mocktype", registry.supported_types)

        # Use it
        chunks = registry.chunk("some content", doc_type="mocktype")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "mocked")
