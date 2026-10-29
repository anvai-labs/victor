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

"""Graph-guided issue localization helpers.

This module implements a lightweight file-level ranking layer for issue/code
localization. Providers can feed semantic seed hits plus graph neighbors into
the accumulator and receive a stable result shape suitable for code-search
tools and benchmark workflows.
"""

from __future__ import annotations


from victor.storage.vector_stores._graph_accumulator import (  # noqa: F401
    GraphSeedAccumulator,
    _dedupe_graph_rows,
    _match_hints,
    extract_issue_hints,
)


class IssueLocalizationAccumulator(GraphSeedAccumulator):
    """Accumulate seed hits and graph evidence into file-level localization ranks."""

    id_prefix = "localization"
    metadata_key = "localization"
    seed_reason = "semantic_match"
    relation_word = "of"
    context_caller_weight = 0.05
    context_callee_weight = 0.05
    relation_caller_weight = 0.18
    relation_other_weight = 0.14
    support_boost_unit = 0.03
    support_boost_cap = 0.12

    def __init__(self, issue_description: str, *, context_limit: int = 3) -> None:
        super().__init__(issue_description, context_limit=context_limit)
        self.issue_description = issue_description


__all__ = [
    "IssueLocalizationAccumulator",
    "extract_issue_hints",
]
