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

"""Conversation export functionality for sharing and archiving.

Provides export capabilities for conversations in various formats
including Markdown, JSON, and HTML.
"""

from victor.core.json_utils import json_dumps, json_loads
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from html import escape as html_escape

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Available export formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    TEXT = "text"


@dataclass
class ConversationMessage:
    """A message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationExport:
    """Exported conversation data."""

    messages: List[ConversationMessage]
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationExporter:
    """Exports conversations to various formats."""

    def __init__(self):
        """Initialize the exporter."""
        self._exporters = {
            ExportFormat.MARKDOWN: self._export_markdown,
            ExportFormat.JSON: self._export_json,
            ExportFormat.HTML: self._export_html,
            ExportFormat.TEXT: self._export_text,
        }

    def export(
        self,
        conversation: ConversationExport,
        format: ExportFormat = ExportFormat.MARKDOWN,
        include_metadata: bool = True,
        include_tool_calls: bool = True,
    ) -> str:
        """Export a conversation to the specified format.

        Args:
            conversation: The conversation to export
            format: Output format
            include_metadata: Include metadata in export
            include_tool_calls: Include tool call details

        Returns:
            Exported content as string
        """
        exporter = self._exporters.get(format)
        if not exporter:
            raise ValueError(f"Unsupported format: {format}")

        return exporter(conversation, include_metadata, include_tool_calls)

    def export_to_file(
        self,
        conversation: ConversationExport,
        file_path: str,
        format: Optional[ExportFormat] = None,
        include_metadata: bool = True,
        include_tool_calls: bool = True,
    ) -> str:
        """Export a conversation to a file.

        Args:
            conversation: The conversation to export
            file_path: Output file path
            format: Output format (auto-detected from extension if not specified)
            include_metadata: Include metadata
            include_tool_calls: Include tool calls

        Returns:
            Path to the exported file
        """
        path = Path(file_path)

        # Auto-detect format from extension
        if format is None:
            ext = path.suffix.lower()
            format_map = {
                ".md": ExportFormat.MARKDOWN,
                ".json": ExportFormat.JSON,
                ".html": ExportFormat.HTML,
                ".htm": ExportFormat.HTML,
                ".txt": ExportFormat.TEXT,
            }
            format = format_map.get(ext, ExportFormat.MARKDOWN)

        content = self.export(conversation, format, include_metadata, include_tool_calls)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        logger.info(f"Exported conversation to {path}")
        return str(path)

    def _export_markdown(
        self,
        conversation: ConversationExport,
        include_metadata: bool,
        include_tool_calls: bool,
    ) -> str:
        """Export to Markdown format."""
        lines = []

        # Title
        title = conversation.title or "Conversation Export"
        lines.append(f"# {title}")
        lines.append("")

        # Metadata
        if include_metadata:
            lines.append("## Session Info")
            if conversation.created_at:
                lines.append(f"- **Date:** {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if conversation.model:
                lines.append(f"- **Model:** {conversation.model}")
            if conversation.provider:
                lines.append(f"- **Provider:** {conversation.provider}")
            if conversation.session_id:
                lines.append(f"- **Session ID:** {conversation.session_id}")
            lines.append("")

        # Messages
        lines.append("## Conversation")
        lines.append("")

        for msg in conversation.messages:
            # Role header
            role_display = {
                "user": "**User**",
                "assistant": "**Assistant**",
                "system": "**System**",
            }.get(msg.role, f"**{msg.role.title()}**")

            if msg.timestamp:
                role_display += f" _{msg.timestamp.strftime('%H:%M:%S')}_"

            lines.append(f"### {role_display}")
            lines.append("")

            # Content
            lines.append(msg.content)
            lines.append("")

            # Tool calls
            if include_tool_calls and msg.tool_calls:
                lines.append("<details>")
                lines.append("<summary>Tool Calls</summary>")
                lines.append("")
                for tool_call in msg.tool_calls:
                    lines.append(f"**{tool_call.get('name', 'Unknown')}**")
                    lines.append("```json")
                    args = tool_call.get("arguments", {})
                    lines.append(json_dumps(args, indent=2))
                    lines.append("```")
                    if "result" in tool_call:
                        lines.append("Result:")
                        lines.append("```")
                        lines.append(str(tool_call["result"])[:500])
                        lines.append("```")
                    lines.append("")
                lines.append("</details>")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Footer
        lines.append(f"*Exported from Victor on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def _export_json(
        self,
        conversation: ConversationExport,
        include_metadata: bool,
        include_tool_calls: bool,
    ) -> str:
        """Export to JSON format."""
        data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "messages": [],
        }

        if include_metadata:
            data["metadata"] = {
                "title": conversation.title,
                "created_at": (
                    conversation.created_at.isoformat() if conversation.created_at else None
                ),
                "model": conversation.model,
                "provider": conversation.provider,
                "session_id": conversation.session_id,
            }
            if conversation.metadata:
                data["metadata"].update(conversation.metadata)

        for msg in conversation.messages:
            msg_data = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.timestamp:
                msg_data["timestamp"] = msg.timestamp.isoformat()
            if include_tool_calls and msg.tool_calls:
                msg_data["tool_calls"] = msg.tool_calls
            if include_metadata and msg.metadata:
                msg_data["metadata"] = msg.metadata

            data["messages"].append(msg_data)

        return json_dumps(data, indent=2, ensure_ascii=False)

    def _export_html(
        self,
        conversation: ConversationExport,
        include_metadata: bool,
        include_tool_calls: bool,
    ) -> str:
        """Export to HTML format."""
        title = html_escape(conversation.title or "Conversation Export")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .conversation {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        h1 {{ color: #333; }}
        .metadata {{
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        .message {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
        }}
        .user {{
            background: #e3f2fd;
            margin-left: 20px;
        }}
        .assistant {{
            background: #f3e5f5;
            margin-right: 20px;
        }}
        .system {{
            background: #fff3e0;
            font-style: italic;
        }}
        .role {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #666;
        }}
        .timestamp {{
            font-size: 0.8em;
            color: #999;
        }}
        .tool-calls {{
            background: #f5f5f5;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85em;
        }}
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 0.8em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="conversation">
        <h1>{title}</h1>
"""

        if include_metadata:
            html += '        <div class="metadata">\n'
            if conversation.created_at:
                html += f'            <div><strong>Date:</strong> {conversation.created_at.strftime("%Y-%m-%d %H:%M:%S")}</div>\n'
            if conversation.model:
                html += f"            <div><strong>Model:</strong> {html_escape(conversation.model)}</div>\n"
            if conversation.provider:
                html += f"            <div><strong>Provider:</strong> {html_escape(conversation.provider)}</div>\n"
            html += "        </div>\n"

        for msg in conversation.messages:
            role_class = msg.role.lower()
            role_display = msg.role.title()

            html += f'        <div class="message {role_class}">\n'
            html += f'            <div class="role">{role_display}'
            if msg.timestamp:
                html += f' <span class="timestamp">{msg.timestamp.strftime("%H:%M:%S")}</span>'
            html += "</div>\n"

            # Convert content to HTML (basic markdown-like formatting)
            content = html_escape(msg.content)
            content = content.replace("\n", "<br>\n")
            html += f"            <div>{content}</div>\n"

            if include_tool_calls and msg.tool_calls:
                html += '            <div class="tool-calls">\n'
                html += "                <strong>Tool Calls:</strong><br>\n"
                for tool_call in msg.tool_calls:
                    name = html_escape(tool_call.get("name", "Unknown"))
                    html += f"                <em>{name}</em><br>\n"
                html += "            </div>\n"

            html += "        </div>\n"

        html += f"""        <div class="footer">
            Exported from Victor on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>"""

        return html

    def _export_text(
        self,
        conversation: ConversationExport,
        include_metadata: bool,
        include_tool_calls: bool,
    ) -> str:
        """Export to plain text format."""
        lines = []

        title = conversation.title or "Conversation Export"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

        if include_metadata:
            if conversation.created_at:
                lines.append(f"Date: {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            if conversation.model:
                lines.append(f"Model: {conversation.model}")
            if conversation.provider:
                lines.append(f"Provider: {conversation.provider}")
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        for msg in conversation.messages:
            role = msg.role.upper()
            if msg.timestamp:
                role += f" [{msg.timestamp.strftime('%H:%M:%S')}]"

            lines.append(role)
            lines.append("-" * len(role))
            lines.append(msg.content)

            if include_tool_calls and msg.tool_calls:
                lines.append("")
                lines.append("Tool calls:")
                for tool_call in msg.tool_calls:
                    lines.append(f"  - {tool_call.get('name', 'Unknown')}")

            lines.append("")
            lines.append("")

        lines.append("-" * 40)
        lines.append(f"Exported from Victor on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    @staticmethod
    def from_message_list(
        messages: List[Dict[str, Any]],
        title: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> ConversationExport:
        """Create a ConversationExport from a list of message dicts.

        Args:
            messages: List of message dictionaries
            title: Optional title
            model: Optional model name
            provider: Optional provider name

        Returns:
            ConversationExport object
        """
        conv_messages = []
        for msg in messages:
            timestamp = None
            if "timestamp" in msg:
                if isinstance(msg["timestamp"], str):
                    timestamp = datetime.fromisoformat(msg["timestamp"])
                elif isinstance(msg["timestamp"], datetime):
                    timestamp = msg["timestamp"]

            conv_messages.append(
                ConversationMessage(
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                    timestamp=timestamp,
                    tool_calls=msg.get("tool_calls"),
                    metadata=msg.get("metadata"),
                )
            )

        return ConversationExport(
            messages=conv_messages,
            title=title,
            created_at=datetime.now(),
            model=model,
            provider=provider,
        )


class ConversationPersistence:
    """Persistent storage for conversations.

    Stores conversations in ~/.victor/conversations/ directory.

    Note: Previously named ConversationStore. Renamed to avoid confusion with
    ConversationStore in conversation_memory.py which handles in-session memory.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the store.

        Args:
            base_dir: Base directory for storage (defaults to {project}/.victor/conversations)
        """
        if base_dir is None:
            from victor.config.settings import get_project_paths

            base_dir = get_project_paths().conversations_export_dir
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._exporter = ConversationExporter()

    def save(
        self,
        conversation: ConversationExport,
        session_id: Optional[str] = None,
    ) -> str:
        """Save a conversation to storage.

        Args:
            conversation: The conversation to save
            session_id: Optional session ID (auto-generated if not provided)

        Returns:
            The session ID of the saved conversation
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set session ID
        conversation.session_id = session_id

        # Save as JSON
        file_path = self._base_dir / f"{session_id}.json"
        self._exporter.export_to_file(
            conversation,
            str(file_path),
            format=ExportFormat.JSON,
        )

        logger.info(f"Saved conversation: {session_id}")
        return session_id

    def load(self, session_id: str) -> Optional[ConversationExport]:
        """Load a conversation from storage.

        Args:
            session_id: The session ID to load

        Returns:
            ConversationExport or None if not found
        """
        file_path = self._base_dir / f"{session_id}.json"

        if not file_path.exists():
            logger.warning(f"Conversation not found: {session_id}")
            return None

        try:
            data = json_loads(file_path.read_text(encoding="utf-8"))
            return self._from_json(data)
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return None

    def _from_json(self, data: Dict[str, Any]) -> ConversationExport:
        """Convert JSON data to ConversationExport."""
        metadata = data.get("metadata", {})

        messages = []
        for msg in data.get("messages", []):
            timestamp = None
            if "timestamp" in msg:
                try:
                    timestamp = datetime.fromisoformat(msg["timestamp"])
                except (ValueError, TypeError):
                    pass

            messages.append(
                ConversationMessage(
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                    timestamp=timestamp,
                    tool_calls=msg.get("tool_calls"),
                    metadata=msg.get("metadata"),
                )
            )

        created_at = None
        if metadata.get("created_at"):
            try:
                created_at = datetime.fromisoformat(metadata["created_at"])
            except (ValueError, TypeError):
                pass

        return ConversationExport(
            messages=messages,
            title=metadata.get("title"),
            created_at=created_at,
            model=metadata.get("model"),
            provider=metadata.get("provider"),
            session_id=metadata.get("session_id"),
            metadata=metadata,
        )

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List available sessions.

        Args:
            limit: Maximum sessions to return

        Returns:
            List of session summaries
        """
        sessions = []

        for file_path in sorted(self._base_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json_loads(file_path.read_text(encoding="utf-8"))
                metadata = data.get("metadata", {})
                message_count = len(data.get("messages", []))

                sessions.append(
                    {
                        "session_id": file_path.stem,
                        "title": metadata.get("title", "Untitled"),
                        "created_at": metadata.get("created_at"),
                        "model": metadata.get("model"),
                        "provider": metadata.get("provider"),
                        "message_count": message_count,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to read session {file_path}: {e}")

        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a conversation.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self._base_dir / f"{session_id}.json"

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted conversation: {session_id}")
            return True

        return False

    def get_recent(self, count: int = 5) -> List[ConversationExport]:
        """Get recent conversations.

        Args:
            count: Number of conversations to return

        Returns:
            List of ConversationExport objects
        """
        conversations = []
        sessions = self.list_sessions(count)

        for session in sessions:
            conv = self.load(session["session_id"])
            if conv:
                conversations.append(conv)

        return conversations


# Global instances
_exporter: Optional[ConversationExporter] = None
_store: Optional[ConversationPersistence] = None


def get_exporter() -> ConversationExporter:
    """Get or create the global exporter."""
    global _exporter
    if _exporter is None:
        _exporter = ConversationExporter()
    return _exporter


def get_store() -> ConversationPersistence:
    """Get or create the global conversation persistence store."""
    global _store
    if _store is None:
        _store = ConversationPersistence()
    return _store


# Backward compatibility alias
ConversationStore = ConversationPersistence
