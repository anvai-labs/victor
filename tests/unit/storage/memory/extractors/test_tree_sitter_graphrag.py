# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

import inspect

import pytest

from victor.core.codegraph_adapter import parse_code
from victor.storage.memory.extractors import tree_sitter_extractor as extractor_module
from victor.storage.memory.extractors.tree_sitter_extractor import (
    TreeSitterEntityExtractor,
)
from victor.storage.memory.entity_types import RelationType


@pytest.mark.asyncio
async def test_extract_graphrag_edges_preserves_codegraph_identity():
    pytest.importorskip("victor_codegraph")
    content = "def callee():\n    return 1\n\ndef caller():\n    return callee()\n"
    extractor = TreeSitterEntityExtractor()
    result = await extractor.extract(content, source="test.py")
    canonical = parse_code(content, file_path="test.py", language="python")

    assert canonical is not None
    canonical_ids = {symbol.id for symbol in canonical.symbols}
    memory_ids = {entity.id for entity in result.entities if entity.entity_type.value == "function"}
    assert memory_ids == canonical_ids
    assert any(relation.relation_type is RelationType.CALLS for relation in result.relations)


@pytest.mark.asyncio
async def test_graceful_degradation_when_codegraph_absent(monkeypatch):
    extractor = TreeSitterEntityExtractor()
    monkeypatch.setattr(extractor_module, "parse_code", lambda *args, **kwargs: None)

    result = await extractor.extract("def f():\n    pass\n", source="x.py")

    assert result.entities == []
    assert result.relations == []


@pytest.mark.asyncio
async def test_inline_extraction_uses_codegraph_without_temp_files():
    pytest.importorskip("victor_codegraph")
    extractor = TreeSitterEntityExtractor()

    result = await extractor.extract("def f():\n    pass\n", context={"language": "python"})

    assert {entity.name for entity in result.entities} == {"f"}


def test_legacy_tree_sitter_plumbing_is_deleted() -> None:
    source = inspect.getsource(extractor_module)
    assert "NamedTemporaryFile" not in source
    assert "CapabilityRegistry" not in source
    assert "extract_all" not in source
