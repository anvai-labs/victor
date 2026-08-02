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

import pytest
from unittest.mock import MagicMock, patch
from victor.storage.memory.extractors.tree_sitter_extractor import (
    TreeSitterEntityExtractor,
)
from victor.storage.memory.entity_types import RelationType


@pytest.mark.asyncio
async def test_extract_graphrag_edges():
    # Mock symbols and edges returned by the underlying extractor
    mock_symbol = MagicMock()
    mock_symbol.name = "my_func"
    mock_symbol.type = "function"
    mock_symbol.line_number = 1
    mock_symbol.end_line = 10
    mock_symbol.parent_symbol = None

    mock_edge_call = MagicMock()
    mock_edge_call.source = "caller_func"
    mock_edge_call.target = "callee_func"
    mock_edge_call.edge_type = "CALLS"

    mock_edge_data = MagicMock()
    mock_edge_data.source = "var_a"
    mock_edge_data.target = "var_b"
    mock_edge_data.edge_type = "DATA_DEP"

    mock_edge_control = MagicMock()
    mock_edge_control.source = "flag"
    mock_edge_control.target = "action"
    mock_edge_control.edge_type = "CONTROL_DEP"

    # Setup the extractor
    extractor = TreeSitterEntityExtractor()
    mock_provider = MagicMock()
    mock_provider.detect_language.return_value = "python"
    mock_provider.extract_all.return_value = (
        [mock_symbol],
        [mock_edge_call, mock_edge_data, mock_edge_control],
    )

    with patch.object(extractor, "_get_extractor", return_value=mock_provider):
        result = await extractor.extract("dummy code", source="test.py")

        # Check relations
        relations = {r.relation_type: r for r in result.relations}

        assert RelationType.CALLS in relations
        assert RelationType.DATA_DEP in relations
        assert RelationType.CONTROL_DEP in relations

        assert relations[RelationType.CALLS].relation_type == RelationType.CALLS
        assert relations[RelationType.DATA_DEP].relation_type == RelationType.DATA_DEP
        assert relations[RelationType.CONTROL_DEP].relation_type == RelationType.CONTROL_DEP


@pytest.mark.asyncio
async def test_graceful_degradation_when_capability_absent():
    """When the Tree-sitter capability is not registered, extraction degrades to an empty
    result on the normal control-flow path (no exception), and is_available() reports
    False. This is the optional-dependency graceful-degradation contract."""
    extractor = TreeSitterEntityExtractor()
    with patch.object(extractor, "_get_extractor", return_value=None):
        assert extractor.is_available() is False
        result = await extractor.extract("def f():\n    pass\n", source="x.py")
        assert result.entities == []
        assert result.relations == []
        # Safe to rerun (idempotent for the unavailable case).
        again = await extractor.extract("def f():\n    pass\n", source="x.py")
        assert again.entities == []
        assert again.relations == []


@pytest.mark.asyncio
async def test_inline_extraction_degrades_gracefully_when_capability_absent():
    """The inline (no source path) path must also degrade gracefully, with temp-file
    cleanup still happening via the finally block."""
    extractor = TreeSitterEntityExtractor()
    with patch.object(extractor, "_get_extractor", return_value=None):
        result = await extractor.extract("def f():\n    pass\n", context={"language": "python"})
        assert result.entities == []
        assert result.relations == []
