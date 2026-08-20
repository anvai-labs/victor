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

"""Document type detection for chunking strategy selection.

Detects document type from:
1. File extension (highest priority)
2. URL patterns
3. Content analysis (fallback)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# File extension to document type mapping
EXTENSION_TO_DOCTYPE = {
    # HTML/Web
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    # Code (common languages)
    ".py": "code",
    ".pyw": "code",
    ".pyi": "code",
    ".js": "code",
    ".jsx": "code",
    ".ts": "code",
    ".tsx": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".c": "code",
    ".cpp": "code",
    ".cc": "code",
    ".h": "code",
    ".hpp": "code",
    ".cs": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    ".scala": "code",
    ".r": "code",
    ".R": "code",
    ".jl": "code",
    ".lua": "code",
    ".sh": "code",
    ".bash": "code",
    ".zsh": "code",
    ".sql": "code",
    ".css": "code",
    ".scss": "code",
    ".less": "code",
    # Data formats
    ".json": "json",
    ".jsonl": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".csv": "csv",
    ".tsv": "csv",
    # Documents
    ".txt": "text",
    ".log": "text",
    ".pdf": "pdf",
    ".doc": "document",
    ".docx": "document",
    ".rtf": "document",
}

# Content patterns for detection
HTML_PATTERN = re.compile(
    r"<(!DOCTYPE|html|head|body|div|p|table|section|article)\b",
    re.IGNORECASE,
)
XML_PATTERN = re.compile(r"<\?xml\s|^<[a-zA-Z][\w-]*[^>]*>", re.MULTILINE)
MARKDOWN_PATTERN = re.compile(r"^#{1,6}\s+\w+|```|^\*\*\w+\*\*|^\[\w+\]\(", re.MULTILINE)
CODE_PATTERN = re.compile(
    r"^(def |class |function |fn |func |import |from |package |#include |using )",
    re.MULTILINE,
)


def detect_from_extension(source: str) -> Optional[str]:
    """Detect document type from file extension.

    Args:
        source: File path or URL

    Returns:
        Document type or None if not detected
    """
    # Handle URLs - extract path
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        source = parsed.path

    # Get extension
    ext = Path(source).suffix.lower()
    if ext in EXTENSION_TO_DOCTYPE:
        return EXTENSION_TO_DOCTYPE[ext]

    return None


def detect_from_content(content: str, sample_size: int = 2000) -> str:
    """Detect document type from content analysis.

    Args:
        content: Document content
        sample_size: Number of characters to analyze

    Returns:
        Detected document type (defaults to "text")
    """
    sample = content[:sample_size].strip()

    # HTML detection
    if HTML_PATTERN.search(sample):
        # Make sure it's not XML
        if not sample.lstrip().startswith("<?xml"):
            return "html"

    # JSON detection
    if sample.startswith(("{", "[")):
        try:
            json.loads(content[:10000])
            return "json"
        except json.JSONDecodeError:
            pass

    # XML detection (after HTML to avoid false positives)
    if sample.lstrip().startswith("<?xml") or (
        XML_PATTERN.match(sample) and not HTML_PATTERN.search(sample)
    ):
        return "xml"

    # YAML detection (simple heuristic)
    if re.match(r"^[\w-]+:\s*\S+", sample, re.MULTILINE) and not sample.startswith("{"):
        lines = sample.split("\n")[:10]
        yaml_like = sum(1 for line in lines if re.match(r"^\s*[\w-]+:\s*", line))
        if yaml_like >= 3:
            return "yaml"

    # Markdown detection
    if MARKDOWN_PATTERN.search(sample):
        return "markdown"

    # Code detection
    if CODE_PATTERN.search(sample):
        return "code"

    return "text"


def detect_document_type(
    source: Optional[str] = None,
    content: Optional[str] = None,
) -> str:
    """Detect document type from source and/or content.

    Priority:
    1. File extension from source (if available)
    2. Content-based detection
    3. Default to "text"

    Args:
        source: Source URL or file path (optional)
        content: Document content (optional)

    Returns:
        Document type string (e.g., "html", "json", "markdown", "code", "text")
    """
    # Try extension-based detection first
    if source:
        doc_type = detect_from_extension(source)
        if doc_type:
            logger.debug(f"Detected doc_type={doc_type} from source: {source}")
            return doc_type

    # Fall back to content-based detection
    if content:
        doc_type = detect_from_content(content)
        logger.debug(f"Detected doc_type={doc_type} from content analysis")
        return doc_type

    logger.debug("No source or content provided, defaulting to text")
    return "text"
