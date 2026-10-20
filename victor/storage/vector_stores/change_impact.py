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

"""Graph-guided change-impact analysis helpers."""

from __future__ import annotations

from victor.storage.vector_stores._graph_accumulator import GraphSeedAccumulator


class ChangeImpactAccumulator(GraphSeedAccumulator):
    """Accumulate semantic seeds and graph evidence into blast-radius candidates."""

    id_prefix = "impact"
    metadata_key = "impact"
    seed_reason = "semantic_seed"
    relation_word = "impact from"

    def __init__(self, change_description: str, *, context_limit: int = 3) -> None:
        super().__init__(change_description, context_limit=context_limit)
        self.change_description = change_description


__all__ = ["ChangeImpactAccumulator"]
