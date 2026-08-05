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

"""RAG Tools Package.

Provides tools for RAG operations:
- RAGIngestTool: Ingest documents into the store
- RAGSearchTool: Search for relevant chunks
- RAGQueryTool: Query with automatic context retrieval
- RAGListTool: List indexed documents
- RAGDeleteTool: Delete documents
- RAGStatsTool: Get store statistics
"""

from victor_rag.tools.ingest import RAGIngestTool
from victor_rag.tools.search import RAGSearchTool
from victor_rag.tools.query import RAGQueryTool
from victor_rag.tools.management import RAGListTool, RAGDeleteTool, RAGStatsTool

__all__ = [
    "RAGIngestTool",
    "RAGSearchTool",
    "RAGQueryTool",
    "RAGListTool",
    "RAGDeleteTool",
    "RAGStatsTool",
]
