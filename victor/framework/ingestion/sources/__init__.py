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

"""Source content handlers for document ingestion.

Provides handlers for extracting content from various sources:
- FileHandler: Local files (text, code, etc.)
- PDFHandler: PDF documents
- URLHandler: Web pages and remote content
- HTMLHandler: HTML content extraction

Usage:
    from victor.framework.ingestion.sources import FileHandler, URLHandler

    file_handler = FileHandler()
    content = await file_handler.extract("/path/to/document.md")

    url_handler = URLHandler()
    content = await url_handler.extract("https://example.com/page.html")
"""

from victor.framework.ingestion.sources.handlers import (
    FileHandler,
    PDFHandler,
    URLHandler,
    extract_text_from_html,
)

__all__ = [
    "FileHandler",
    "PDFHandler",
    "URLHandler",
    "extract_text_from_html",
]
