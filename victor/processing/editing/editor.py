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

"""Multi-file editor with diff preview and rollback.

Provides transaction-like editing with:
- Atomic multi-file operations
- Rich diff preview with syntax highlighting
- Rollback to previous state
- Backup management
- Dry-run mode
"""

import difflib
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel


class OperationType(str, Enum):
    """Type of file operation."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class EditOperation(BaseModel):
    """Represents a single file operation."""

    type: OperationType = Field(description="Type of operation")
    path: str = Field(description="File path")
    old_content: Optional[str] = Field(default=None, description="Original content")
    new_content: Optional[str] = Field(default=None, description="New content")
    new_path: Optional[str] = Field(default=None, description="New path for rename")
    backup_path: Optional[str] = Field(default=None, description="Backup file path")
    applied: bool = Field(default=False, description="Whether operation was applied")


class EditTransaction(BaseModel):
    """Represents a transaction of multiple file edits."""

    id: str = Field(description="Transaction ID")
    operations: List[EditOperation] = Field(default_factory=list, description="List of operations")
    committed: bool = Field(default=False, description="Whether transaction was committed")
    rolled_back: bool = Field(default=False, description="Whether transaction was rolled back")
    timestamp: datetime = Field(default_factory=datetime.now, description="Transaction timestamp")
    description: str = Field(default="", description="Transaction description")


class FileEditor:
    """Multi-file editor with atomic operations and rollback.

    Features:
    - Atomic multi-file edits
    - Rich diff preview
    - Automatic backups
    - Rollback capability
    - Dry-run mode
    - Transaction history
    """

    def __init__(
        self,
        backup_dir: Optional[str] = None,
        auto_backup: bool = True,
        console: Optional[Console] = None,
    ):
        """Initialize file editor.

        Args:
            backup_dir: Directory for backups (default: {project}/.victor/backups)
            auto_backup: Automatically backup files before editing
            console: Rich console for output
        """
        from victor.config.settings import get_project_paths

        self.backup_dir = Path(backup_dir or get_project_paths().backups_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.auto_backup = auto_backup
        self.console = console or Console()

        self.current_transaction: Optional[EditTransaction] = None
        self.transaction_history: List[EditTransaction] = []

    def start_transaction(self, description: str = "") -> str:
        """Start a new edit transaction.

        Args:
            description: Description of the transaction

        Returns:
            Transaction ID
        """
        if self.current_transaction:
            raise RuntimeError("Transaction already in progress")

        transaction_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.current_transaction = EditTransaction(id=transaction_id, description=description)

        self.console.print(f"\n[bold cyan]📝 Started transaction:[/] {transaction_id}")
        if description:
            self.console.print(f"[dim]{description}[/]")

        return transaction_id

    def add_create(self, path: str, content: str) -> None:
        """Add a file creation operation.

        Args:
            path: Path to create
            content: File content
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        path_obj = Path(path)
        if path_obj.exists():
            raise FileExistsError(f"File already exists: {path}")

        operation = EditOperation(type=OperationType.CREATE, path=path, new_content=content)

        self.current_transaction.operations.append(operation)
        self.console.print(f"[green]+ Create:[/] {path}")

    def add_modify(self, path: str, new_content: str) -> None:
        """Add a file modification operation.

        Args:
            path: Path to modify
            new_content: New file content
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Read current content
        old_content = path_obj.read_text()

        operation = EditOperation(
            type=OperationType.MODIFY,
            path=path,
            old_content=old_content,
            new_content=new_content,
        )

        self.current_transaction.operations.append(operation)
        self.console.print(f"[yellow]~ Modify:[/] {path}")

    def add_delete(self, path: str) -> None:
        """Add a file deletion operation.

        Args:
            path: Path to delete
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Read current content for backup
        old_content = path_obj.read_text() if path_obj.is_file() else None

        operation = EditOperation(type=OperationType.DELETE, path=path, old_content=old_content)

        self.current_transaction.operations.append(operation)
        self.console.print(f"[red]- Delete:[/] {path}")

    def add_rename(self, old_path: str, new_path: str) -> None:
        """Add a file rename operation.

        Args:
            old_path: Current path
            new_path: New path
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        old_path_obj = Path(old_path)
        if not old_path_obj.exists():
            raise FileNotFoundError(f"File not found: {old_path}")

        new_path_obj = Path(new_path)
        if new_path_obj.exists():
            raise FileExistsError(f"Target already exists: {new_path}")

        operation = EditOperation(type=OperationType.RENAME, path=old_path, new_path=new_path)

        self.current_transaction.operations.append(operation)
        self.console.print(f"[blue]→ Rename:[/] {old_path} → {new_path}")

    def preview_diff(self, context_lines: int = 3) -> None:
        """Show diff preview of all changes.

        Args:
            context_lines: Number of context lines in diff
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        self.console.print("\n[bold cyan]📋 Changes Preview:[/]")
        self.console.print("=" * 70)

        for i, op in enumerate(self.current_transaction.operations, 1):
            self.console.print(f"\n[bold]{i}. {op.type.value.upper()}:[/] {op.path}")
            self.console.print("-" * 70)

            if op.type == OperationType.CREATE:
                self._show_create_preview(op)
            elif op.type == OperationType.MODIFY:
                self._show_modify_preview(op, context_lines)
            elif op.type == OperationType.DELETE:
                self._show_delete_preview(op)
            elif op.type == OperationType.RENAME:
                self._show_rename_preview(op)

    def _show_create_preview(self, op: EditOperation) -> None:
        """Show preview for file creation."""
        lines = (op.new_content or "").split("\n")
        preview = "\n".join(lines[:20])
        if len(lines) > 20:
            preview += f"\n... ({len(lines) - 20} more lines)"

        # Detect language from file extension
        ext = Path(op.path).suffix
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".rb": "ruby",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
        }
        language = lang_map.get(ext, "text")

        syntax = Syntax(preview, language, theme="monokai", line_numbers=True)
        self.console.print(
            Panel(syntax, title=f"New File ({len(lines)} lines)", border_style="green")
        )

    def _show_modify_preview(self, op: EditOperation, context_lines: int) -> None:
        """Show diff preview for file modification."""
        old_lines = (op.old_content or "").splitlines(keepends=True)
        new_lines = (op.new_content or "").splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{op.path}",
            tofile=f"b/{op.path}",
            n=context_lines,
        )

        diff_text = "".join(diff)
        if diff_text:
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
            self.console.print(Panel(syntax, title="Diff", border_style="yellow"))
        else:
            self.console.print("[dim]No changes[/]")

    def _show_delete_preview(self, op: EditOperation) -> None:
        """Show preview for file deletion."""
        if op.old_content:
            lines = op.old_content.split("\n")
            preview = "\n".join(lines[:10])
            if len(lines) > 10:
                preview += f"\n... ({len(lines) - 10} more lines)"

            self.console.print(
                Panel(
                    f"[red]File will be deleted ({len(lines)} lines)[/]\n\n{preview}",
                    title="Deleted Content (preview)",
                    border_style="red",
                )
            )
        else:
            self.console.print("[red]File will be deleted[/]")

    def _show_rename_preview(self, op: EditOperation) -> None:
        """Show preview for file rename."""
        self.console.print(
            Panel(
                f"[blue]Old:[/] {op.path}\n[blue]New:[/] {op.new_path}",
                title="Rename",
                border_style="blue",
            )
        )

    def commit(self, dry_run: bool = False) -> bool:
        """Commit the transaction (apply all changes).

        Args:
            dry_run: If True, don't actually apply changes

        Returns:
            True if successful, False otherwise
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        if self.current_transaction.committed:
            raise RuntimeError("Transaction already committed")

        self.console.print(
            f"\n[bold cyan]{'🔍 Dry Run' if dry_run else '💾 Committing'}:[/] {len(self.current_transaction.operations)} operations"
        )

        if dry_run:
            self.console.print("[dim]Dry run mode - no changes will be applied[/]")
            return True

        # Apply operations
        try:
            for i, op in enumerate(self.current_transaction.operations, 1):
                self.console.print(f"\n[{i}/{len(self.current_transaction.operations)}] ", end="")
                self._apply_operation(op)
                op.applied = True

            self.current_transaction.committed = True
            self.transaction_history.append(self.current_transaction)

            self.console.print("\n[bold green]✓ Transaction committed successfully[/]")
            return True

        except Exception as e:
            self.console.print(f"\n[bold red]✗ Error applying changes:[/] {e}")
            self.console.print("[yellow]Rolling back...[/]")
            self.rollback()
            return False

        finally:
            self.current_transaction = None

    def _apply_operation(self, op: EditOperation) -> None:
        """Apply a single operation.

        Args:
            op: Operation to apply
        """
        path_obj = Path(op.path)

        # Backup if needed
        if self.auto_backup and op.type in [OperationType.MODIFY, OperationType.DELETE]:
            backup_path = self._create_backup(op.path)
            op.backup_path = str(backup_path)

        # Apply operation
        if op.type == OperationType.CREATE:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(op.new_content or "")
            self.console.print(f"[green]✓ Created:[/] {op.path}")

        elif op.type == OperationType.MODIFY:
            path_obj.write_text(op.new_content or "")
            self.console.print(f"[yellow]✓ Modified:[/] {op.path}")

        elif op.type == OperationType.DELETE:
            if path_obj.is_file():
                path_obj.unlink()
            elif path_obj.is_dir():
                shutil.rmtree(path_obj)
            self.console.print(f"[red]✓ Deleted:[/] {op.path}")

        elif op.type == OperationType.RENAME:
            if op.new_path is None:
                raise ValueError("new_path is required for rename operations")
            new_path_obj = Path(op.new_path)
            new_path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.rename(new_path_obj)
            self.console.print(f"[blue]✓ Renamed:[/] {op.path} → {op.new_path}")

    def _create_backup(self, path: str) -> Path:
        """Create backup of a file.

        Args:
            path: Path to backup

        Returns:
            Path to backup file
        """
        path_obj = Path(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path_obj.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name

        if path_obj.is_file():
            shutil.copy2(path_obj, backup_path)

        return backup_path

    def rollback(self) -> bool:
        """Rollback the current transaction.

        Returns:
            True if successful, False otherwise
        """
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        self.console.print("\n[bold yellow]⏪ Rolling back transaction...[/]")

        # Rollback in reverse order
        for op in reversed(self.current_transaction.operations):
            if not op.applied:
                continue

            try:
                self._rollback_operation(op)
            except Exception as e:
                self.console.print(f"[red]✗ Error rolling back {op.path}:[/] {e}")
                return False

        self.current_transaction.rolled_back = True
        self.transaction_history.append(self.current_transaction)
        self.current_transaction = None

        self.console.print("[bold green]✓ Rollback complete[/]")
        return True

    def _rollback_operation(self, op: EditOperation) -> None:
        """Rollback a single operation.

        Args:
            op: Operation to rollback
        """
        path_obj = Path(op.path)

        if op.type == OperationType.CREATE:
            # Delete created file
            if path_obj.exists():
                path_obj.unlink()
                self.console.print(f"[green]✓ Rolled back (deleted):[/] {op.path}")

        elif op.type == OperationType.MODIFY:
            # Restore from backup
            if op.backup_path:
                backup_path = Path(op.backup_path)
                if backup_path.exists():
                    shutil.copy2(backup_path, path_obj)
                    self.console.print(f"[yellow]✓ Rolled back (restored):[/] {op.path}")
            elif op.old_content is not None:
                path_obj.write_text(op.old_content)
                self.console.print(f"[yellow]✓ Rolled back (content):[/] {op.path}")

        elif op.type == OperationType.DELETE:
            # Restore from backup
            if op.backup_path:
                backup_path = Path(op.backup_path)
                if backup_path.exists():
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, path_obj)
                    self.console.print(f"[red]✓ Rolled back (restored):[/] {op.path}")
            elif op.old_content is not None:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                path_obj.write_text(op.old_content)
                self.console.print(f"[red]✓ Rolled back (restored):[/] {op.path}")

        elif op.type == OperationType.RENAME:
            # Rename back
            if op.new_path is None:
                raise ValueError("new_path is required for rename operations")
            new_path_obj = Path(op.new_path)
            if new_path_obj.exists():
                new_path_obj.rename(path_obj)
                self.console.print(f"[blue]✓ Rolled back (renamed):[/] {op.new_path} → {op.path}")

    def abort(self) -> None:
        """Abort the current transaction without applying changes."""
        if not self.current_transaction:
            raise RuntimeError("No active transaction")

        self.console.print("\n[bold red]✗ Transaction aborted[/]")
        self.current_transaction = None

    def get_transaction_summary(self) -> Dict[str, Any]:
        """Get summary of current transaction.

        Returns:
            Summary dictionary
        """
        if not self.current_transaction:
            return {"error": "No active transaction"}

        return {
            "id": self.current_transaction.id,
            "description": self.current_transaction.description,
            "operations": len(self.current_transaction.operations),
            "by_type": {
                "create": len(
                    [
                        op
                        for op in self.current_transaction.operations
                        if op.type == OperationType.CREATE
                    ]
                ),
                "modify": len(
                    [
                        op
                        for op in self.current_transaction.operations
                        if op.type == OperationType.MODIFY
                    ]
                ),
                "delete": len(
                    [
                        op
                        for op in self.current_transaction.operations
                        if op.type == OperationType.DELETE
                    ]
                ),
                "rename": len(
                    [
                        op
                        for op in self.current_transaction.operations
                        if op.type == OperationType.RENAME
                    ]
                ),
            },
            "committed": self.current_transaction.committed,
            "rolled_back": self.current_transaction.rolled_back,
        }
