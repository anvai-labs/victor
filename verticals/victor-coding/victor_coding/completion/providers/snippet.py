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

"""Snippet-based completion provider.

Provides template-based completions for common code patterns.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from victor_coding.completion.protocol import (
    CompletionCapabilities,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionParams,
    InsertTextFormat,
    InlineCompletionList,
    InlineCompletionParams,
)
from victor_coding.completion.provider import CachingCompletionProvider

logger = logging.getLogger(__name__)


@dataclass
class Snippet:
    """A code snippet template."""

    name: str
    prefix: str  # Trigger prefix
    body: str  # Snippet body (may contain $1, $2, etc.)
    description: str = ""
    scope: list[str] = field(default_factory=list)  # Languages

    def matches_prefix(self, text: str) -> bool:
        """Check if this snippet matches the given prefix."""
        return self.prefix.startswith(text) or text.startswith(self.prefix)

    def get_insert_text(self) -> str:
        """Get the snippet body formatted for insertion."""
        return self.body


# Built-in snippets organized by language
BUILTIN_SNIPPETS: dict[str, list[dict]] = {
    "python": [
        {
            "name": "def",
            "prefix": "def",
            "body": "def ${1:name}(${2:args}):\n    ${3:pass}",
            "description": "Define a function",
        },
        {
            "name": "class",
            "prefix": "class",
            "body": "class ${1:Name}:\n    def __init__(self${2:, args}):\n        ${3:pass}",
            "description": "Define a class",
        },
        {
            "name": "if",
            "prefix": "if",
            "body": "if ${1:condition}:\n    ${2:pass}",
            "description": "If statement",
        },
        {
            "name": "for",
            "prefix": "for",
            "body": "for ${1:item} in ${2:iterable}:\n    ${3:pass}",
            "description": "For loop",
        },
        {
            "name": "with",
            "prefix": "with",
            "body": "with ${1:expression} as ${2:var}:\n    ${3:pass}",
            "description": "With statement",
        },
        {
            "name": "try",
            "prefix": "try",
            "body": "try:\n    ${1:pass}\nexcept ${2:Exception} as e:\n    ${3:pass}",
            "description": "Try/except block",
        },
        {
            "name": "async def",
            "prefix": "async",
            "body": "async def ${1:name}(${2:args}):\n    ${3:pass}",
            "description": "Async function",
        },
        {
            "name": "dataclass",
            "prefix": "@dataclass",
            "body": "@dataclass\nclass ${1:Name}:\n    ${2:field}: ${3:type}",
            "description": "Dataclass",
        },
        {
            "name": "main",
            "prefix": "main",
            "body": 'if __name__ == "__main__":\n    ${1:main()}',
            "description": "Main block",
        },
        {
            "name": "logger",
            "prefix": "logger",
            "body": "logger = logging.getLogger(__name__)",
            "description": "Logger instance",
        },
    ],
    "javascript": [
        {
            "name": "function",
            "prefix": "function",
            "body": "function ${1:name}(${2:params}) {\n    ${3}\n}",
            "description": "Function declaration",
        },
        {
            "name": "arrow",
            "prefix": "=>",
            "body": "(${1:params}) => {\n    ${2}\n}",
            "description": "Arrow function",
        },
        {
            "name": "const",
            "prefix": "const",
            "body": "const ${1:name} = ${2:value};",
            "description": "Const declaration",
        },
        {
            "name": "async function",
            "prefix": "async",
            "body": "async function ${1:name}(${2:params}) {\n    ${3}\n}",
            "description": "Async function",
        },
        {
            "name": "try catch",
            "prefix": "try",
            "body": "try {\n    ${1}\n} catch (${2:error}) {\n    ${3}\n}",
            "description": "Try/catch block",
        },
        {
            "name": "import",
            "prefix": "import",
            "body": "import { ${2:module} } from '${1:package}';",
            "description": "ES6 import",
        },
        {
            "name": "export",
            "prefix": "export",
            "body": "export ${1:default} ${2:name};",
            "description": "Export statement",
        },
        {
            "name": "class",
            "prefix": "class",
            "body": "class ${1:Name} {\n    constructor(${2:params}) {\n        ${3}\n    }\n}",
            "description": "Class declaration",
        },
    ],
    "typescript": [
        {
            "name": "interface",
            "prefix": "interface",
            "body": "interface ${1:Name} {\n    ${2:property}: ${3:type};\n}",
            "description": "Interface declaration",
        },
        {
            "name": "type",
            "prefix": "type",
            "body": "type ${1:Name} = ${2:type};",
            "description": "Type alias",
        },
        {
            "name": "function typed",
            "prefix": "function",
            "body": "function ${1:name}(${2:params}: ${3:type}): ${4:ReturnType} {\n    ${5}\n}",
            "description": "Typed function",
        },
        {
            "name": "async function typed",
            "prefix": "async",
            "body": "async function ${1:name}(${2:params}: ${3:type}): Promise<${4:ReturnType}> {\n    ${5}\n}",
            "description": "Typed async function",
        },
        {
            "name": "enum",
            "prefix": "enum",
            "body": "enum ${1:Name} {\n    ${2:Value},\n}",
            "description": "Enum declaration",
        },
    ],
    "rust": [
        {
            "name": "fn",
            "prefix": "fn",
            "body": "fn ${1:name}(${2:args}) -> ${3:()} {\n    ${4}\n}",
            "description": "Function",
        },
        {
            "name": "struct",
            "prefix": "struct",
            "body": "struct ${1:Name} {\n    ${2:field}: ${3:Type},\n}",
            "description": "Struct",
        },
        {
            "name": "impl",
            "prefix": "impl",
            "body": "impl ${1:Type} {\n    ${2}\n}",
            "description": "Implementation block",
        },
        {
            "name": "match",
            "prefix": "match",
            "body": "match ${1:expr} {\n    ${2:pattern} => ${3:result},\n    _ => ${4:default},\n}",
            "description": "Match expression",
        },
        {
            "name": "test",
            "prefix": "#[test]",
            "body": "#[test]\nfn ${1:test_name}() {\n    ${2}\n}",
            "description": "Test function",
        },
    ],
    "go": [
        {
            "name": "func",
            "prefix": "func",
            "body": "func ${1:name}(${2:args}) ${3:returnType} {\n    ${4}\n}",
            "description": "Function",
        },
        {
            "name": "struct",
            "prefix": "type struct",
            "body": "type ${1:Name} struct {\n    ${2:Field} ${3:Type}\n}",
            "description": "Struct type",
        },
        {
            "name": "interface",
            "prefix": "type interface",
            "body": "type ${1:Name} interface {\n    ${2:Method}(${3:args}) ${4:returnType}\n}",
            "description": "Interface type",
        },
        {
            "name": "if err",
            "prefix": "iferr",
            "body": "if err != nil {\n    return ${1:err}\n}",
            "description": "Error check",
        },
        {
            "name": "for range",
            "prefix": "forr",
            "body": "for ${1:i}, ${2:v} := range ${3:collection} {\n    ${4}\n}",
            "description": "For range loop",
        },
    ],
    "java": [
        {
            "name": "class",
            "prefix": "class",
            "body": "public class ${1:Name} {\n    ${2}\n}",
            "description": "Class declaration",
        },
        {
            "name": "method",
            "prefix": "public",
            "body": "public ${1:void} ${2:name}(${3:args}) {\n    ${4}\n}",
            "description": "Public method",
        },
        {
            "name": "main",
            "prefix": "main",
            "body": "public static void main(String[] args) {\n    ${1}\n}",
            "description": "Main method",
        },
        {
            "name": "try catch",
            "prefix": "try",
            "body": "try {\n    ${1}\n} catch (${2:Exception} e) {\n    ${3}\n}",
            "description": "Try/catch block",
        },
        {
            "name": "for",
            "prefix": "for",
            "body": "for (int ${1:i} = 0; ${1:i} < ${2:length}; ${1:i}++) {\n    ${3}\n}",
            "description": "For loop",
        },
    ],
}


class SnippetCompletionProvider(CachingCompletionProvider):
    """Snippet-based completion provider.

    Provides completions from:
    - Built-in snippets for common patterns
    - User-defined snippets from config files
    - Project-specific snippets
    """

    def __init__(
        self,
        priority: int = 70,
        snippets_dir: Optional[Path] = None,
    ):
        """Initialize the snippet provider.

        Args:
            priority: Provider priority (default 70)
            snippets_dir: Directory containing custom snippets
        """
        super().__init__(priority=priority)
        self._snippets: dict[str, list[Snippet]] = {}
        self._snippets_dir = snippets_dir
        self._load_builtin_snippets()

    @property
    def name(self) -> str:
        return "snippet"

    def get_capabilities(self) -> CompletionCapabilities:
        """Return snippet completion capabilities."""
        return CompletionCapabilities(
            supports_completion=True,
            supports_inline_completion=False,
            supports_resolve=False,
            supports_snippets=True,
            supported_languages=list(self._snippets.keys()),
        )

    def _load_builtin_snippets(self) -> None:
        """Load built-in snippets."""
        for language, snippets_data in BUILTIN_SNIPPETS.items():
            self._snippets[language] = [
                Snippet(
                    name=s["name"],
                    prefix=s["prefix"],
                    body=s["body"],
                    description=s.get("description", ""),
                    scope=[language],
                )
                for s in snippets_data
            ]

    def load_snippets_from_file(self, file_path: Path) -> None:
        """Load snippets from a JSON file.

        File format (VS Code compatible):
        {
            "Snippet Name": {
                "prefix": "trigger",
                "body": ["line1", "line2"],
                "description": "Description",
                "scope": "python,javascript"
            }
        }

        Args:
            file_path: Path to snippets JSON file
        """
        try:
            with open(file_path) as f:
                data = json.load(f)

            for name, snippet_data in data.items():
                prefix = snippet_data.get("prefix", name)
                body = snippet_data.get("body", "")
                if isinstance(body, list):
                    body = "\n".join(body)

                description = snippet_data.get("description", "")
                scope_str = snippet_data.get("scope", "")
                scope = [s.strip() for s in scope_str.split(",")] if scope_str else []

                snippet = Snippet(
                    name=name,
                    prefix=prefix,
                    body=body,
                    description=description,
                    scope=scope,
                )

                # Add to all scoped languages
                if scope:
                    for lang in scope:
                        if lang not in self._snippets:
                            self._snippets[lang] = []
                        self._snippets[lang].append(snippet)
                else:
                    # Add to a special "all" category
                    if "all" not in self._snippets:
                        self._snippets["all"] = []
                    self._snippets["all"].append(snippet)

            logger.debug(f"Loaded snippets from {file_path}")

        except Exception as e:
            logger.warning(f"Failed to load snippets from {file_path}: {e}")

    def load_snippets_from_directory(self, directory: Path) -> None:
        """Load all snippet files from a directory.

        Args:
            directory: Directory containing snippet files
        """
        if not directory.exists():
            return

        for file_path in directory.glob("*.json"):
            self.load_snippets_from_file(file_path)

    def add_snippet(self, snippet: Snippet) -> None:
        """Add a snippet.

        Args:
            snippet: Snippet to add
        """
        for lang in snippet.scope or ["all"]:
            if lang not in self._snippets:
                self._snippets[lang] = []
            self._snippets[lang].append(snippet)

    def remove_snippet(self, name: str, language: Optional[str] = None) -> bool:
        """Remove a snippet by name.

        Args:
            name: Snippet name
            language: Optional language scope

        Returns:
            True if found and removed
        """
        languages = [language] if language else list(self._snippets.keys())
        found = False

        for lang in languages:
            if lang in self._snippets:
                original_len = len(self._snippets[lang])
                self._snippets[lang] = [s for s in self._snippets[lang] if s.name != name]
                if len(self._snippets[lang]) < original_len:
                    found = True

        return found

    async def provide_completions(self, params: CompletionParams) -> CompletionList:
        """Provide snippet completions.

        Args:
            params: Completion parameters

        Returns:
            CompletionList with matching snippets
        """
        # Check cache first
        cached = self._get_cached(params)
        if cached:
            return cached

        # Get snippets for this language
        language = params.language.lower()
        snippets = self._snippets.get(language, [])
        snippets.extend(self._snippets.get("all", []))

        # Filter by prefix
        prefix = params.prefix.strip()
        if not prefix:
            # Return all snippets when no prefix
            matching = snippets
        else:
            # Get last word as potential trigger
            words = prefix.split()
            trigger = words[-1] if words else ""

            matching = [s for s in snippets if s.matches_prefix(trigger)]

        # Convert to completion items
        items = []
        for snippet in matching:
            item = CompletionItem(
                label=snippet.prefix,
                kind=CompletionItemKind.SNIPPET,
                detail=snippet.name,
                documentation=snippet.description,
                insert_text=snippet.get_insert_text(),
                insert_text_format=InsertTextFormat.SNIPPET,
                provider=self.name,
                confidence=0.9,
            )
            items.append(item)

        # Sort by prefix
        items.sort(key=lambda i: i.label)

        result = CompletionList(
            is_incomplete=False,
            items=items[: params.max_results],
        )

        self._set_cached(params, result)
        return result

    async def provide_inline_completions(
        self, params: InlineCompletionParams
    ) -> InlineCompletionList:
        """Snippets don't provide inline completions."""
        return InlineCompletionList(items=[])

    def get_snippets_for_language(self, language: str) -> list[Snippet]:
        """Get all snippets for a language.

        Args:
            language: Language identifier

        Returns:
            List of snippets
        """
        snippets = self._snippets.get(language.lower(), [])
        snippets.extend(self._snippets.get("all", []))
        return snippets
