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

"""Guard: memory RelationType code-relations must match the graph EdgeType vocabulary.

The CALLS / DATA_DEP / CONTROL_DEP relation values are intentionally UPPERCASE because
they mirror the canonical program-analysis edge types the graph index emits and that
graph/code_search edge filters match on. If they drift (e.g. someone "normalizes" them to
lowercase for consistency with the semantic relations), tree-sitter graphrag relations
would silently stop matching edge-type-filtered graph queries. This test locks the
alignment so that regression is caught at test time.
"""

from victor.storage.graph.edge_types import EdgeType
from victor.storage.memory.entity_types import RelationType


def test_code_relation_values_match_graph_edge_vocabulary():
    assert RelationType.CALLS.value == EdgeType.CALLS.value == "CALLS"
    assert RelationType.DATA_DEP.value == EdgeType.DDG_DEF_USE.value == "DDG_DEF_USE"
    assert RelationType.CONTROL_DEP.value == EdgeType.CDG.value == "CDG"
