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

"""Language Server Protocol (LSP) standard types.

This module provides core LSP types as defined in the LSP specification.
These types are cross-vertical, usable beyond coding contexts for any
document-based operations.

Cross-Vertical Benefits:
- Coding: Language server integration
- DevOps: Config file diagnostics
- Data Analysis: Data file location references
- Research: Document cross-references

Based on LSP Specification 3.17:
https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/

Example usage:
    from victor.protocols.lsp_types import Position, Range, Location, Diagnostic

    # Create a position (0-indexed)
    pos = Position(line=10, character=5)

    # Create a range
    range_ = Range(
        start=Position(0, 0),
        end=Position(0, 10),
    )

    # Create a diagnostic
    diag = Diagnostic(
        range=range_,
        message="Variable not defined",
        severity=DiagnosticSeverity.ERROR,
    )
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

# =============================================================================
# Enumerations
# =============================================================================


class DiagnosticSeverity(IntEnum):
    """Diagnostic severity levels (LSP 3.17)."""

    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


class CompletionItemKind(IntEnum):
    """Completion item kinds (LSP 3.17)."""

    TEXT = 1
    METHOD = 2
    FUNCTION = 3
    CONSTRUCTOR = 4
    FIELD = 5
    VARIABLE = 6
    CLASS = 7
    INTERFACE = 8
    MODULE = 9
    PROPERTY = 10
    UNIT = 11
    VALUE = 12
    ENUM = 13
    KEYWORD = 14
    SNIPPET = 15
    COLOR = 16
    FILE = 17
    REFERENCE = 18
    FOLDER = 19
    ENUM_MEMBER = 20
    CONSTANT = 21
    STRUCT = 22
    EVENT = 23
    OPERATOR = 24
    TYPE_PARAMETER = 25


class SymbolKind(IntEnum):
    """Symbol kinds (LSP 3.17)."""

    FILE = 1
    MODULE = 2
    NAMESPACE = 3
    PACKAGE = 4
    CLASS = 5
    METHOD = 6
    PROPERTY = 7
    FIELD = 8
    CONSTRUCTOR = 9
    ENUM = 10
    INTERFACE = 11
    FUNCTION = 12
    VARIABLE = 13
    CONSTANT = 14
    STRING = 15
    NUMBER = 16
    BOOLEAN = 17
    ARRAY = 18
    OBJECT = 19
    KEY = 20
    NULL = 21
    ENUM_MEMBER = 22
    STRUCT = 23
    EVENT = 24
    OPERATOR = 25
    TYPE_PARAMETER = 26


class DiagnosticTag(IntEnum):
    """Diagnostic tags for additional metadata (LSP 3.17)."""

    UNNECESSARY = 1  # Unused or unnecessary code
    DEPRECATED = 2  # Deprecated or obsolete code


# =============================================================================
# Position and Range Types
# =============================================================================


@dataclass
class Position:
    """A position in a text document (0-indexed).

    Represents a cursor position with line and character offset.
    Both are 0-indexed.

    Attributes:
        line: Line position (0-indexed)
        character: Character offset in line (0-indexed)
    """

    line: int
    character: int

    def to_dict(self) -> Dict[str, int]:
        """Convert to LSP-compatible dictionary."""
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Position":
        """Create from LSP dictionary."""
        return cls(line=data["line"], character=data["character"])

    def __lt__(self, other: "Position") -> bool:
        """Compare positions."""
        if self.line != other.line:
            return self.line < other.line
        return self.character < other.character

    def __le__(self, other: "Position") -> bool:
        """Compare positions."""
        return self == other or self < other


@dataclass
class Range:
    """A range in a text document.

    Represents a text span from start to end position.
    End position is exclusive.

    Attributes:
        start: Start position (inclusive)
        end: End position (exclusive)
    """

    start: Position
    end: Position

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Range":
        """Create from LSP dictionary."""
        return cls(
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
        )

    def contains(self, position: Position) -> bool:
        """Check if position is within this range."""
        return self.start <= position and position < self.end

    def overlaps(self, other: "Range") -> bool:
        """Check if ranges overlap."""
        return self.start < other.end and other.start < self.end

    @property
    def is_empty(self) -> bool:
        """Check if range is empty (start == end)."""
        return self.start == self.end


@dataclass
class Location:
    """A location in a document.

    Represents a specific location (file + range) in a workspace.

    Attributes:
        uri: Document URI (file:// format)
        range: Range within the document
    """

    uri: str
    range: Range

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        return {"uri": self.uri, "range": self.range.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Location":
        """Create from LSP dictionary."""
        return cls(uri=data["uri"], range=Range.from_dict(data["range"]))


@dataclass
class LocationLink:
    """A link to a location in a document.

    Extended location with origin information for better navigation.

    Attributes:
        target_uri: Target document URI
        target_range: Full range of the target (e.g., function body)
        target_selection_range: Range to select/highlight (e.g., function name)
        origin_selection_range: Range in origin document that triggered this
    """

    target_uri: str
    target_range: Range
    target_selection_range: Range
    origin_selection_range: Optional[Range] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        result = {
            "targetUri": self.target_uri,
            "targetRange": self.target_range.to_dict(),
            "targetSelectionRange": self.target_selection_range.to_dict(),
        }
        if self.origin_selection_range:
            result["originSelectionRange"] = self.origin_selection_range.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationLink":
        """Create from LSP dictionary."""
        origin = data.get("originSelectionRange")
        return cls(
            target_uri=data["targetUri"],
            target_range=Range.from_dict(data["targetRange"]),
            target_selection_range=Range.from_dict(data["targetSelectionRange"]),
            origin_selection_range=Range.from_dict(origin) if origin else None,
        )


# =============================================================================
# Diagnostic Types
# =============================================================================


@dataclass
class DiagnosticRelatedInformation:
    """Related information for a diagnostic.

    Provides additional context about a diagnostic, such as
    related code locations.

    Attributes:
        location: Related location
        message: Description of the relationship
    """

    location: Location
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        return {"location": self.location.to_dict(), "message": self.message}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosticRelatedInformation":
        """Create from LSP dictionary."""
        return cls(
            location=Location.from_dict(data["location"]),
            message=data["message"],
        )


@dataclass
class Diagnostic:
    """A diagnostic (error, warning, etc.).

    Represents a problem or suggestion in a document.

    Attributes:
        range: Range where the diagnostic applies
        message: Human-readable message
        severity: Severity level (ERROR, WARNING, etc.)
        source: Source of the diagnostic (e.g., "pylint", "typescript")
        code: Diagnostic code (e.g., "E501", "TS2304")
        code_description: URI to documentation about the diagnostic code
        tags: Additional metadata tags
        related_information: Related locations and messages
        data: Additional data for code actions
    """

    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    source: Optional[str] = None
    code: Optional[Union[str, int]] = None
    code_description: Optional[str] = None
    tags: List[DiagnosticTag] = field(default_factory=list)
    related_information: List[DiagnosticRelatedInformation] = field(default_factory=list)
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        result: Dict[str, Any] = {
            "range": self.range.to_dict(),
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.source:
            result["source"] = self.source
        if self.code is not None:
            result["code"] = self.code
        if self.code_description:
            result["codeDescription"] = {"href": self.code_description}
        if self.tags:
            result["tags"] = [t.value for t in self.tags]
        if self.related_information:
            result["relatedInformation"] = [ri.to_dict() for ri in self.related_information]
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Diagnostic":
        """Create from LSP dictionary."""
        code_desc = data.get("codeDescription", {})
        tags = [DiagnosticTag(t) for t in data.get("tags", [])]
        related = [
            DiagnosticRelatedInformation.from_dict(ri) for ri in data.get("relatedInformation", [])
        ]
        return cls(
            range=Range.from_dict(data["range"]),
            message=data["message"],
            severity=DiagnosticSeverity(data.get("severity", 1)),
            source=data.get("source"),
            code=data.get("code"),
            code_description=code_desc.get("href") if code_desc else None,
            tags=tags,
            related_information=related,
            data=data.get("data"),
        )

    @property
    def is_error(self) -> bool:
        """Check if this is an error diagnostic."""
        return self.severity == DiagnosticSeverity.ERROR

    @property
    def is_warning(self) -> bool:
        """Check if this is a warning diagnostic."""
        return self.severity == DiagnosticSeverity.WARNING


# =============================================================================
# Completion Types
# =============================================================================


@dataclass
class CompletionItem:
    """A completion item.

    Represents a suggested completion with metadata.

    Attributes:
        label: Display label
        kind: Item kind (function, variable, etc.)
        detail: Short description (e.g., type signature)
        documentation: Full documentation
        insert_text: Text to insert (if different from label)
        filter_text: Text for filtering
        sort_text: Text for sorting
        preselect: Whether to preselect this item
        deprecated: Whether this item is deprecated
    """

    label: str
    kind: CompletionItemKind = CompletionItemKind.TEXT
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None
    filter_text: Optional[str] = None
    sort_text: Optional[str] = None
    preselect: bool = False
    deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        result: Dict[str, Any] = {
            "label": self.label,
            "kind": self.kind.value,
        }
        if self.detail:
            result["detail"] = self.detail
        if self.documentation:
            result["documentation"] = self.documentation
        if self.insert_text:
            result["insertText"] = self.insert_text
        if self.filter_text:
            result["filterText"] = self.filter_text
        if self.sort_text:
            result["sortText"] = self.sort_text
        if self.preselect:
            result["preselect"] = self.preselect
        if self.deprecated:
            result["deprecated"] = self.deprecated
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletionItem":
        """Create from LSP dictionary."""
        doc = data.get("documentation")
        if isinstance(doc, dict):
            doc = doc.get("value", "")
        return cls(
            label=data["label"],
            kind=CompletionItemKind(data.get("kind", 1)),
            detail=data.get("detail"),
            documentation=doc,
            insert_text=data.get("insertText"),
            filter_text=data.get("filterText"),
            sort_text=data.get("sortText"),
            preselect=data.get("preselect", False),
            deprecated=data.get("deprecated", False),
        )


# =============================================================================
# Hover Types
# =============================================================================


@dataclass
class Hover:
    """Hover information.

    Information shown when hovering over a symbol.

    Attributes:
        contents: Hover contents (markdown or plain text)
        range: Range this hover applies to
    """

    contents: str
    range: Optional[Range] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        result: Dict[str, Any] = {"contents": {"kind": "markdown", "value": self.contents}}
        if self.range:
            result["range"] = self.range.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hover":
        """Create from LSP dictionary."""
        contents = data.get("contents", "")
        if isinstance(contents, dict):
            contents = contents.get("value", "")
        elif isinstance(contents, list):
            contents = "\n".join(c.get("value", c) if isinstance(c, dict) else c for c in contents)

        range_data = data.get("range")
        return cls(
            contents=contents,
            range=Range.from_dict(range_data) if range_data else None,
        )


# =============================================================================
# Symbol Types
# =============================================================================


@dataclass
class DocumentSymbol:
    """A symbol in a document.

    Hierarchical representation of symbols (classes, functions, etc.).

    Attributes:
        name: Symbol name
        kind: Symbol kind
        range: Full range of the symbol
        selection_range: Range of the symbol name
        detail: Additional details (e.g., type signature)
        children: Child symbols
        deprecated: Whether the symbol is deprecated
    """

    name: str
    kind: SymbolKind
    range: Range
    selection_range: Range
    detail: Optional[str] = None
    children: List["DocumentSymbol"] = field(default_factory=list)
    deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        result: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "range": self.range.to_dict(),
            "selectionRange": self.selection_range.to_dict(),
        }
        if self.detail:
            result["detail"] = self.detail
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        if self.deprecated:
            result["deprecated"] = self.deprecated
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSymbol":
        """Create from LSP dictionary."""
        children = [DocumentSymbol.from_dict(c) for c in data.get("children", [])]
        return cls(
            name=data["name"],
            kind=SymbolKind(data["kind"]),
            range=Range.from_dict(data["range"]),
            selection_range=Range.from_dict(data["selectionRange"]),
            detail=data.get("detail"),
            children=children,
            deprecated=data.get("deprecated", False),
        )


@dataclass
class SymbolInformation:
    """Symbol information (flat representation).

    Non-hierarchical symbol representation with location.

    Attributes:
        name: Symbol name
        kind: Symbol kind
        location: Symbol location
        container_name: Name of the containing symbol
        deprecated: Whether the symbol is deprecated
    """

    name: str
    kind: SymbolKind
    location: Location
    container_name: Optional[str] = None
    deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        result: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "location": self.location.to_dict(),
        }
        if self.container_name:
            result["containerName"] = self.container_name
        if self.deprecated:
            result["deprecated"] = self.deprecated
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolInformation":
        """Create from LSP dictionary."""
        return cls(
            name=data["name"],
            kind=SymbolKind(data["kind"]),
            location=Location.from_dict(data["location"]),
            container_name=data.get("containerName"),
            deprecated=data.get("deprecated", False),
        )


# =============================================================================
# Text Edit Types
# =============================================================================


@dataclass
class TextEdit:
    """A text edit.

    Represents a modification to a document.

    Attributes:
        range: Range to replace
        new_text: Replacement text
    """

    range: Range
    new_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        return {"range": self.range.to_dict(), "newText": self.new_text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextEdit":
        """Create from LSP dictionary."""
        return cls(
            range=Range.from_dict(data["range"]),
            new_text=data["newText"],
        )


@dataclass
class TextDocumentIdentifier:
    """Identifies a text document.

    Attributes:
        uri: Document URI
    """

    uri: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to LSP-compatible dictionary."""
        return {"uri": self.uri}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "TextDocumentIdentifier":
        """Create from LSP dictionary."""
        return cls(uri=data["uri"])


@dataclass
class VersionedTextDocumentIdentifier(TextDocumentIdentifier):
    """Identifies a versioned text document.

    Attributes:
        uri: Document URI
        version: Document version
    """

    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        return {"uri": self.uri, "version": self.version}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionedTextDocumentIdentifier":
        """Create from LSP dictionary."""
        return cls(uri=data["uri"], version=data.get("version", 0))


@dataclass
class TextDocumentEdit:
    """Edit to a versioned text document.

    Attributes:
        text_document: Document to edit
        edits: List of edits
    """

    text_document: VersionedTextDocumentIdentifier
    edits: List[TextEdit]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LSP-compatible dictionary."""
        return {
            "textDocument": self.text_document.to_dict(),
            "edits": [e.to_dict() for e in self.edits],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextDocumentEdit":
        """Create from LSP dictionary."""
        return cls(
            text_document=VersionedTextDocumentIdentifier.from_dict(data["textDocument"]),
            edits=[TextEdit.from_dict(e) for e in data["edits"]],
        )


__all__ = [
    # Enumerations
    "DiagnosticSeverity",
    "CompletionItemKind",
    "SymbolKind",
    "DiagnosticTag",
    # Position and Range
    "Position",
    "Range",
    "Location",
    "LocationLink",
    # Diagnostics
    "DiagnosticRelatedInformation",
    "Diagnostic",
    # Completions
    "CompletionItem",
    # Hover
    "Hover",
    # Symbols
    "DocumentSymbol",
    "SymbolInformation",
    # Text Edits
    "TextEdit",
    "TextDocumentIdentifier",
    "VersionedTextDocumentIdentifier",
    "TextDocumentEdit",
]
