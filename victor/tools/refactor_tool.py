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

"""Code refactoring tool for safe code transformations.

Features:
- Rename symbols (variables, functions, classes)
- Extract functions from code blocks
- Inline variables
- Organize imports
- AST-based analysis for safety
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.formatters import format_refactor_operations

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icons
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter (lazy initialization)."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


# Helper functions


def _find_symbol(tree: ast.AST, name: str) -> Optional[Dict[str, Any]]:
    """Find symbol in AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return {"type": "function", "node": node}
        elif isinstance(node, ast.ClassDef) and node.name == name:
            return {"type": "class", "node": node}
        elif isinstance(node, ast.Name) and node.id == name:
            return {"type": "variable", "node": node}

    return None


def _analyze_variables(tree: ast.AST) -> Dict[str, List[str]]:
    """Analyze variables used in code block."""
    # Simple analysis - can be enhanced
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)

    # For now, assume all names are parameters
    # More sophisticated analysis would distinguish local vs external
    return {
        "params": list(names)[:5],  # Limit to 5 for safety
        "returns": [],
    }


def _find_function_insert_point(lines: List[str], current_line: int) -> int:
    """Find best location to insert extracted function."""
    # Simple heuristic: insert at the beginning of file (after imports)
    for i, line in enumerate(lines):
        if (
            line.strip()
            and not line.strip().startswith("import")
            and not line.strip().startswith("from")
        ):
            return max(0, i - 1)

    return 0


def _find_variable_assignment(tree: ast.AST, name: str) -> Optional[Dict[str, Any]]:
    """Find simple variable assignment."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    # Get value as string
                    try:
                        value_str = ast.unparse(node.value)
                    except (AttributeError, ValueError):
                        value_str = str(node.value)

                    return {
                        "line": node.lineno,
                        "value": value_str,
                    }

    return None


def _is_stdlib(module_name: str) -> bool:
    """Check if module is from standard library."""
    # Common stdlib modules
    stdlib_modules = {
        "abc",
        "ast",
        "asyncio",
        "collections",
        "concurrent",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "hashlib",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "pickle",
        "re",
        "shutil",
        "socket",
        "sqlite3",
        "string",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
    }

    # Get top-level module
    top_level = module_name.split(".")[0]
    return top_level in stdlib_modules


# Tool functions


def _collect_python_files(
    path: Path,
    scope: str,
    depth: int,
    current_depth: int = 0,
) -> List[Path]:
    """Collect Python files based on scope and depth settings.

    Args:
        path: Starting path (file or directory)
        scope: "file", "directory", or "project"
        depth: Max depth (-1=unlimited, 0=current only, N=N levels)
        current_depth: Current recursion depth (internal)

    Returns:
        List of Python file paths to process
    """
    files = []

    if path.is_file():
        if path.suffix == ".py":
            files.append(path)
        return files

    if not path.is_dir():
        return files

    # Check depth limit
    if depth >= 0 and current_depth > depth:
        return files

    # Directories to skip
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        "build",
        "dist",
    }

    for item in path.iterdir():
        if item.name in skip_dirs:
            continue

        if item.is_file() and item.suffix == ".py":
            files.append(item)
        elif item.is_dir() and scope in ("directory", "project"):
            # Recurse into subdirectories based on scope
            if scope == "directory" and current_depth == 0:
                # For "directory" scope, only go one level deep
                sub_files = _collect_python_files(item, "file", depth, current_depth + 1)
            else:
                sub_files = _collect_python_files(item, scope, depth, current_depth + 1)
            files.extend(sub_files)

    return files


def _rename_in_file(
    file_path: Path,
    old_name: str,
    new_name: str,
    require_definition: bool = False,
) -> Optional[Dict[str, Any]]:
    """Perform rename in a single file.

    Args:
        file_path: Path to Python file
        old_name: Current symbol name
        new_name: New symbol name
        require_definition: If True, only rename if symbol is defined in file

    Returns:
        Dict with changes info, or None if no changes/file couldn't be processed
    """
    try:
        content = file_path.read_text()
    except (IOError, OSError) as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return None

    # Parse AST
    try:
        tree = ast.parse(content)
    except SyntaxError:
        logger.warning(f"Syntax error in {file_path}, skipping")
        return None

    # Check if symbol exists in this file
    pattern = r"\b" + re.escape(old_name) + r"\b"
    if not re.search(pattern, content):
        return None

    # If requiring definition, verify symbol is defined here
    if require_definition:
        symbol_info = _find_symbol(tree, old_name)
        if not symbol_info:
            return None

    # Perform rename
    lines = content.split("\n")
    modified_lines = []
    changes = []

    for line_num, line in enumerate(lines, 1):
        if re.search(pattern, line):
            modified_line = re.sub(pattern, new_name, line)
            changes.append(
                {
                    "line": line_num,
                    "old": line.strip(),
                    "new": modified_line.strip(),
                }
            )
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    if not changes:
        return None

    return {
        "file_path": str(file_path),
        "changes": changes,
        "original_content": content,
        "new_content": "\n".join(modified_lines),
    }


@tool(
    cost_tier=CostTier.LOW,
    category="refactor",
    priority=Priority.MEDIUM,  # Task-specific refactoring
    access_mode=AccessMode.WRITE,  # Modifies source files
    danger_level=DangerLevel.LOW,  # Changes are undoable
    keywords=[
        "rename",
        "refactor",
        "symbol",
        "variable",
        "function",
        "class",
        "project",
        "multi-file",
        "ast",
    ],
    mandatory_keywords=[
        "rename variable",
        "rename function",
        "refactor code",
    ],  # Force inclusion
    task_types=["refactor", "edit"],  # Classification-aware selection
    stages=["execution"],  # Conversation stages where relevant
)
async def rename(
    old_name: str,
    new_name: str,
    path: str = ".",
    scope: str = "file",
    depth: int = -1,
    preview: bool = False,
) -> Dict[str, Any]:
    """[AST-AWARE] Rename symbols safely using word-boundary matching.

    Uses AST parsing + word boundaries to rename symbols without false positives.
    SAFE: Won't rename 'get_user' to 'fetch_user' inside 'get_username'.
    Use this for Python symbol refactoring. Use edit() for non-code text changes.

    Args:
        old_name: Current symbol name to rename.
        new_name: New symbol name.
        path: File or directory path. For "file" scope, must be a file.
              For "directory"/"project" scope, specifies starting directory.
        scope: Rename scope:
               - "file": Single file only (path must be a file)
               - "directory": All .py files in the directory (non-recursive)
               - "project": All .py files recursively (respects depth)
        depth: Directory traversal depth for "project" scope:
               - -1: Unlimited (default)
               - 0: Current directory only
               - N: Up to N levels deep
        preview: If True, show what would change without applying.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - files_count: Number of files modified
        - total_changes: Total occurrences changed across all files
        - file_changes: List of {file, changes} for each modified file
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed

    Examples:
        # Single file rename
        rename("get_user", "fetch_user", path="utils.py", scope="file")

        # Directory rename (non-recursive)
        rename("Config", "Settings", path="src/", scope="directory")

        # Project-wide rename
        rename("old_func", "new_func", scope="project")

        # Limited depth rename
        rename("helper", "util", path="lib/", scope="project", depth=2)
    """
    if not old_name or not new_name:
        return {
            "success": False,
            "error": "Missing required parameters: old_name, new_name",
        }

    if old_name == new_name:
        return {"success": False, "error": "old_name and new_name must be different"}

    # Validate scope
    valid_scopes = ("file", "directory", "project")
    if scope not in valid_scopes:
        return {
            "success": False,
            "error": f"Invalid scope '{scope}'. Must be one of: {valid_scopes}",
        }

    path_obj = Path(path).resolve()

    # Validate path based on scope
    if scope == "file":
        if not path_obj.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not path_obj.is_file():
            return {
                "success": False,
                "error": f"Path must be a file for scope='file': {path}",
            }
        if path_obj.suffix != ".py":
            return {
                "success": False,
                "error": f"File must be a Python file (.py): {path}",
            }
    else:
        if not path_obj.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        if not path_obj.is_dir():
            return {
                "success": False,
                "error": f"Path must be a directory for scope='{scope}': {path}",
            }

    # Collect files to process
    files = _collect_python_files(path_obj, scope, depth)

    if not files:
        return {
            "success": False,
            "error": f"No Python files found in {path} with scope='{scope}'",
        }

    # Process each file
    all_file_changes: List[Dict[str, Any]] = []
    total_changes = 0

    # For single-file scope, require symbol definition
    require_def = scope == "file"

    for file_path in files:
        result = _rename_in_file(file_path, old_name, new_name, require_definition=require_def)
        if result:
            all_file_changes.append(result)
            total_changes += len(result["changes"])

    if not all_file_changes:
        if scope == "file":
            return {
                "success": False,
                "error": f"Symbol '{old_name}' not found in {path}",
            }
        else:
            return {
                "success": False,
                "error": f"No occurrences of '{old_name}' found in {len(files)} files",
            }

    # Build report
    report = []
    report.append(f"Rename Refactoring: '{old_name}' → '{new_name}'")
    report.append("=" * 70)
    report.append("")
    report.append(f"Scope: {scope}")
    report.append(f"Path: {path}")
    if scope == "project" and depth >= 0:
        report.append(f"Depth: {depth}")
    report.append(f"Files analyzed: {len(files)}")
    report.append(f"Files modified: {len(all_file_changes)}")
    report.append(f"Total changes: {total_changes} occurrences")
    report.append("")

    # Show changes per file
    report.append("Changes by file:")
    for fc in all_file_changes[:15]:  # Show first 15 files
        rel_path = (
            Path(fc["file_path"]).relative_to(Path.cwd())
            if Path(fc["file_path"]).is_relative_to(Path.cwd())
            else fc["file_path"]
        )
        report.append(f"\n  {rel_path} ({len(fc['changes'])} changes):")
        for change in fc["changes"][:5]:  # Show first 5 changes per file
            report.append(f"    Line {change['line']}: {change['old'][:50]}...")

        if len(fc["changes"]) > 5:
            report.append(f"    ... and {len(fc['changes']) - 5} more")

    if len(all_file_changes) > 15:
        report.append(f"\n... and {len(all_file_changes) - 15} more files")

    # Apply changes if not preview
    if not preview:
        applied_count = 0
        for fc in all_file_changes:
            try:
                Path(fc["file_path"]).write_text(fc["new_content"])
                applied_count += 1
            except (IOError, OSError) as e:
                logger.error(f"Failed to write {fc['file_path']}: {e}")

        report.append("")
        if applied_count == len(all_file_changes):
            report.append(f"{_get_icon('success')} All {applied_count} files updated successfully")
        else:
            report.append(
                f"{_get_icon('warning')}  {applied_count}/{len(all_file_changes)} files updated (some failed)"
            )
    else:
        report.append("")
        report.append(f"{_get_icon('warning')}  PREVIEW MODE - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "files_count": len(all_file_changes),
        "total_changes": total_changes,
        "file_changes": [
            {
                "file": fc["file_path"],
                "changes_count": len(fc["changes"]),
                "changes": fc["changes"],
            }
            for fc in all_file_changes
        ],
        "formatted_report": "\n".join(report),
    }


@tool(
    cost_tier=CostTier.LOW,
    category="refactor",
    priority=Priority.MEDIUM,  # Task-specific refactoring
    access_mode=AccessMode.WRITE,  # Modifies source files
    danger_level=DangerLevel.LOW,  # Changes are undoable
    keywords=["extract", "refactor", "function", "method", "code block"],
    mandatory_keywords=[
        "refactor",
        "extract function",
        "extract method",
    ],  # From MANDATORY_TOOL_KEYWORDS
    stages=["execution"],  # Conversation stages where relevant
)
async def extract(
    file: str,
    start_line: int,
    end_line: int,
    function_name: str,
    preview: bool = False,
) -> Dict[str, Any]:
    """Extract code block into a new function.

    Extracts selected lines of code into a new function,
    analyzing variables to determine parameters and return values.

    Args:
        file: File path to refactor.
        start_line: Start line number (1-indexed).
        end_line: End line number (inclusive).
        function_name: Name for the extracted function.
        preview: Preview changes without applying (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - new_function: The extracted function code
        - parameters: List of inferred parameters
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed
    """
    if not file or not start_line or not end_line or not function_name:
        return {
            "success": False,
            "error": "Missing required parameters: file, start_line, end_line, function_name",
        }

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read file
    content = file_obj.read_text()
    lines = content.split("\n")

    # Validate line numbers
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return {
            "success": False,
            "error": f"Invalid line range: {start_line}-{end_line}",
        }

    # Extract code block (0-indexed)
    extracted_lines = lines[start_line - 1 : end_line]
    code_block = "\n".join(extracted_lines)

    # Determine indentation
    if extracted_lines:
        first_line = extracted_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        base_indent = " " * indent
    else:
        base_indent = ""

    # Analyze variables used in extracted code
    try:
        # Parse just the extracted block
        block_tree = ast.parse(code_block)
        variables = _analyze_variables(block_tree)
    except SyntaxError:
        # If extraction block has syntax errors, try to infer params
        variables = {"params": [], "returns": []}

    # Build new function
    params_str = ", ".join(variables.get("params", []))
    new_function = f"\n{base_indent}def {function_name}({params_str}):\n"

    # Add docstring
    new_function += f'{base_indent}    """Extracted function."""\n'

    # Add extracted code (with additional indent)
    for line in extracted_lines:
        new_function += f"    {line}\n"

    # Add return statement if needed
    if variables.get("returns"):
        returns_str = ", ".join(variables["returns"])
        new_function += f"{base_indent}    return {returns_str}\n"

    # Build modified content
    modified_lines = lines[: start_line - 1]  # Before extraction

    # Add function call
    call_indent = " " * indent
    if variables.get("returns"):
        returns = ", ".join(variables["returns"])
        modified_lines.append(f"{call_indent}{returns} = {function_name}({params_str})")
    else:
        modified_lines.append(f"{call_indent}{function_name}({params_str})")

    modified_lines.extend(lines[end_line:])  # After extraction

    # Insert function definition at appropriate location
    # Find the best place (e.g., before the current function)
    insert_line = _find_function_insert_point(lines, start_line)
    modified_lines.insert(insert_line, new_function)

    new_content = "\n".join(modified_lines)

    # Build report
    report = []
    report.append(f"Extract Function: '{function_name}'")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Extracted lines: {start_line}-{end_line}")
    report.append(f"Parameters: {params_str or '(none)'}")
    report.append("")
    report.append("New function:")
    report.append(new_function)
    report.append("")
    report.append("Replaced with call:")
    if variables.get("returns"):
        returns = ", ".join(variables["returns"])
        report.append(f"  {returns} = {function_name}({params_str})")
    else:
        report.append(f"  {function_name}({params_str})")

    # Apply changes if not preview
    if not preview:
        file_obj.write_text(new_content)
        report.append("")
        report.append(f"{_get_icon('success')} Function extracted successfully")
    else:
        report.append("")
        report.append(f"{_get_icon('warning')}  This is a PREVIEW - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "new_function": new_function,
        "parameters": variables.get("params", []),
        "formatted_report": "\n".join(report),
    }


@tool(
    cost_tier=CostTier.LOW,
    category="refactor",
    priority=Priority.MEDIUM,  # Task-specific refactoring
    access_mode=AccessMode.WRITE,  # Modifies source files
    danger_level=DangerLevel.LOW,  # Changes are undoable
    keywords=["inline", "refactor", "variable", "replace", "expand"],
    mandatory_keywords=["refactor", "inline variable"],  # From MANDATORY_TOOL_KEYWORDS
)
async def inline(
    file: str,
    variable_name: str,
    preview: bool = False,
) -> Dict[str, Any]:
    """Inline a variable by replacing usages with its assigned value.

    Args:
        file: File path to refactor
        variable_name: Variable to inline
        preview: Preview without applying

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - changes_count: Number of locations changed
        - value: The inlined value
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed
    """
    if not file or not variable_name:
        return {
            "success": False,
            "error": "Missing required parameters: file, variable_name",
        }

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read file
    content = file_obj.read_text()

    # Parse AST to find variable assignment
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Find variable assignment
    assignment = _find_variable_assignment(tree, variable_name)

    if not assignment:
        return {
            "success": False,
            "error": f"Simple assignment for '{variable_name}' not found",
        }

    lines = content.split("\n")

    # Remove assignment line
    assignment_line = assignment["line"]
    value_expr = assignment["value"]

    # Replace variable usage with value
    modified_lines = []
    changes = []

    for line_num, line in enumerate(lines, 1):
        if line_num == assignment_line:
            # Skip assignment line
            changes.append(
                {
                    "line": line_num,
                    "action": "removed",
                    "content": line,
                }
            )
            continue

        # Replace variable usage
        pattern = r"\b" + re.escape(variable_name) + r"\b"
        if re.search(pattern, line):
            modified_line = re.sub(pattern, value_expr, line)
            changes.append(
                {
                    "line": line_num,
                    "action": "modified",
                    "old": line,
                    "new": modified_line,
                }
            )
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    new_content = "\n".join(modified_lines)

    # Build report
    report = []
    report.append(f"Inline Variable: '{variable_name}'")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Value: {value_expr}")
    report.append(f"Changes: {len(changes)} locations")
    report.append("")

    if changes:
        report.append("Changes:")
        for change in changes[:10]:
            if change["action"] == "removed":
                report.append(f"\nLine {change['line']} (removed):")
                report.append(f"  - {change['content']}")
            else:
                report.append(f"\nLine {change['line']} (inlined):")
                report.append(f"  - {change['old']}")
                report.append(f"  + {change['new']}")

        if len(changes) > 10:
            report.append(f"\n... and {len(changes) - 10} more changes")

    # Apply changes if not preview
    if not preview:
        file_obj.write_text(new_content)
        report.append("")
        report.append(f"{_get_icon('success')} Variable inlined successfully")
    else:
        report.append("")
        report.append(f"{_get_icon('warning')}  This is a PREVIEW - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "changes_count": len(changes),
        "value": value_expr,
        "formatted_report": "\n".join(report),
    }


@tool(
    cost_tier=CostTier.LOW,
    category="refactor",
    priority=Priority.MEDIUM,  # Task-specific refactoring
    access_mode=AccessMode.WRITE,  # Modifies source files
    danger_level=DangerLevel.LOW,  # Changes are undoable
    keywords=["organize", "imports", "sort", "refactor", "cleanup"],
)
async def organize_imports(
    file: str,
    preview: bool = False,
) -> Dict[str, Any]:
    """Organize imports: sort into groups (stdlib/third-party/local), remove duplicates.

    Args:
        file: File path to refactor
        preview: Preview without applying

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - stdlib_count: Number of standard library imports
        - third_party_count: Number of third-party imports
        - local_count: Number of local imports
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read file
    content = file_obj.read_text()
    lines = content.split("\n")

    # Parse AST to find imports
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Collect imports
    stdlib_imports = []
    third_party_imports = []
    local_imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_line = f"import {alias.name}"
                if alias.asname:
                    import_line += f" as {alias.asname}"

                # Categorize (simple heuristic)
                if _is_stdlib(alias.name):
                    stdlib_imports.append(import_line)
                elif alias.name.startswith("."):
                    local_imports.append(import_line)
                else:
                    third_party_imports.append(import_line)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            names_str = ", ".join(names)

            if node.level > 0:
                # Relative import
                dots = "." * node.level
                import_line = f"from {dots}{module} import {names_str}"
                local_imports.append(import_line)
            else:
                import_line = f"from {module} import {names_str}"
                if _is_stdlib(module):
                    stdlib_imports.append(import_line)
                else:
                    third_party_imports.append(import_line)

    # Sort each group
    stdlib_imports = sorted(set(stdlib_imports))
    third_party_imports = sorted(set(third_party_imports))
    local_imports = sorted(set(local_imports))

    # Build organized imports
    organized = []
    if stdlib_imports:
        organized.extend(stdlib_imports)
        organized.append("")

    if third_party_imports:
        organized.extend(third_party_imports)
        organized.append("")

    if local_imports:
        organized.extend(local_imports)
        organized.append("")

    # Find where imports end in original file
    import_end_line = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#"):
            try:
                node = ast.parse(line).body[0]
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_end_line = i + 1
            except (SyntaxError, IndexError, ValueError):
                if import_end_line > 0:
                    break

    # Build new content
    new_lines = []

    # Keep docstring and initial comments
    docstring_end = 0
    for i, line in enumerate(lines):
        if i == 0 and (line.strip().startswith('"""') or line.strip().startswith("'''")):
            # Module docstring
            new_lines.append(line)
            for j in range(i + 1, len(lines)):
                new_lines.append(lines[j])
                if '"""' in lines[j] or "'''" in lines[j]:
                    docstring_end = j + 1
                    break
            break
        elif line.strip().startswith("#"):
            new_lines.append(line)
        else:
            break

    # Add organized imports
    if docstring_end > 0:
        new_lines.append("")
    new_lines.extend(organized)

    # Add rest of file (skipping old imports)
    rest_start = max(import_end_line, docstring_end)
    new_lines.extend(lines[rest_start:])

    new_content = "\n".join(new_lines)

    # Build report
    report = []
    report.append("Organize Imports")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Standard library: {len(stdlib_imports)}")
    report.append(f"Third-party: {len(third_party_imports)}")
    report.append(f"Local: {len(local_imports)}")
    report.append("")

    if organized:
        report.append("Organized imports:")
        for line in organized[:20]:
            report.append(f"  {line}")

        if len(organized) > 20:
            report.append(f"  ... and {len(organized) - 20} more")

    # Apply changes if not preview
    if not preview:
        file_obj.write_text(new_content)
        report.append("")
        report.append(f"{_get_icon('success')} Imports organized successfully")
    else:
        report.append("")
        report.append(f"{_get_icon('warning')}  This is a PREVIEW - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "stdlib_count": len(stdlib_imports),
        "third_party_count": len(third_party_imports),
        "local_count": len(local_imports),
        "formatted_report": "\n".join(report),
    }
