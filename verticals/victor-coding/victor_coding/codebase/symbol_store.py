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

"""SQLite-based symbol store for queryable code structure.

Stores extracted symbols (classes, functions, interfaces, etc.) in SQLite
for efficient querying. This complements the embedding store by providing
structured access to code metadata.

Features:
- Multi-language support (Python, TypeScript, Go, Rust, Java, etc.)
- Queryable by name, type, category, file path
- Supports both OOP and procedural paradigms
- Stores method signatures, docstrings, line numbers
- Architecture pattern detection

Usage:
    store = SymbolStore("/path/to/project")
    await store.index_codebase()

    # Query by category
    providers = store.find_by_category("provider")

    # Query by type
    classes = store.find_by_type("class")

    # Query by pattern
    handlers = store.find_by_name_pattern("%Handler")
"""

import ast
import hashlib
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from victor_coding.codebase.ignore_patterns import DEFAULT_SKIP_DIRS, should_ignore_path

logger = logging.getLogger(__name__)


# Pattern categories for architecture detection (universal across languages)
ARCHITECTURE_PATTERNS = {
    "provider": ["Provider", "Backend", "Client", "Connector", "Adapter", "Gateway"],
    "service": ["Service", "UseCase", "Interactor", "Manager", "Facade"],
    "repository": ["Repository", "Store", "Cache", "DAO", "Registry"],
    "controller": ["Controller", "Handler", "Endpoint", "Router", "Route"],
    "model": ["Model", "Entity", "Schema", "DTO", "Record", "Domain"],
    "factory": ["Factory", "Builder", "Creator", "Producer"],
    "middleware": ["Middleware", "Interceptor", "Filter", "Guard", "Pipe"],
    "observer": ["Observer", "Listener", "Subscriber", "Watcher", "Hook"],
    "strategy": ["Strategy", "Policy", "Algorithm"],
    "component": ["Component", "Widget", "View", "Screen", "Page"],
    "config": ["Config", "Settings", "Options", "Preferences", "Environment"],
    "util": ["Util", "Utils", "Helper", "Helpers", "Common"],
    "test": ["Test", "Spec", "Mock", "Stub", "Fake"],
}


@dataclass
class SymbolInfo:
    """Information about a code symbol."""

    name: str
    symbol_type: str  # class, function, interface, struct, enum, method, etc.
    file_path: str
    line_number: int
    language: str
    category: Optional[str] = None  # Architecture category (provider, service, etc.)
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent_symbol: Optional[str] = None  # For methods: parent class name
    modifiers: List[str] = field(default_factory=list)  # public, private, async, static
    is_exported: bool = False
    content_hash: Optional[str] = None


@dataclass
class FileInfo:
    """Information about a source file."""

    path: str
    language: str
    size: int
    lines: int
    last_modified: float
    indexed_at: float
    content_hash: str
    symbol_count: int = 0
    import_count: int = 0


class SymbolStore:
    """SQLite-based storage for code symbols with multi-language support."""

    # Language extensions (source code files that can be parsed for symbols)
    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".scala": "scala",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".swift": "swift",
        ".dart": "dart",
        ".ex": "elixir",
        ".exs": "elixir",
        ".vue": "vue",
        ".svelte": "svelte",
    }

    # Config extensions (configuration and documentation files)
    # These are indexed for file counting and LOC but not parsed for symbols
    CONFIG_EXTENSIONS = {
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".hocon": "hocon",
        ".xml": "xml",
        ".md": "markdown",
        ".txt": "text",
        ".cfg": "config",
        ".conf": "config",
        ".props": "properties",
    }

    # Use shared default skip directories from ignore_patterns module
    # Hidden directories (starting with '.') are excluded automatically
    # by the shared should_ignore_path() utility
    SKIP_DIRS = DEFAULT_SKIP_DIRS

    def __init__(
        self,
        root_path: str,
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ):
        """Initialize symbol store.

        Args:
            root_path: Root directory of the codebase
            include_dirs: List of directories to include in the analysis.
            exclude_dirs: List of directories to exclude from the analysis.
        """
        self.root = Path(root_path).resolve()
        self.include_dirs = include_dirs

        self.effective_skip_dirs = self.SKIP_DIRS.copy()
        if exclude_dirs:
            self.effective_skip_dirs.update(exclude_dirs)

        self._init_db()

    def _resolve_search_path(self, path: str | Path) -> Path:
        """Normalize include/search paths against the resolved project root."""
        raw_path = Path(path)
        return raw_path.resolve() if raw_path.is_absolute() else (self.root / raw_path).resolve()

    def _relative_path(self, file_path: Path) -> str:
        """Return a stable project-relative path from any symlink-normalized file path."""
        return str(file_path.resolve().relative_to(self.root))

    @property
    def _db_path(self) -> Path:
        """Path to the canonical project-local SQLite database."""
        from victor_coding.compat.settings import get_project_paths

        return get_project_paths(self.root).project_db

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript("""
                -- Files table
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    language TEXT NOT NULL,
                    size INTEGER,
                    lines INTEGER,
                    last_modified REAL,
                    indexed_at REAL,
                    content_hash TEXT,
                    symbol_count INTEGER DEFAULT 0,
                    import_count INTEGER DEFAULT 0,
                    file_type TEXT DEFAULT 'source'
                );

                -- Symbols table
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    language TEXT NOT NULL,
                    category TEXT,
                    docstring TEXT,
                    signature TEXT,
                    parent_symbol TEXT,
                    modifiers TEXT,
                    is_exported INTEGER DEFAULT 0,
                    content_hash TEXT,
                    FOREIGN KEY (file_path) REFERENCES files(path)
                );

                -- Imports/dependencies table
                CREATE TABLE IF NOT EXISTS imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    import_name TEXT NOT NULL,
                    import_type TEXT DEFAULT 'module',
                    FOREIGN KEY (file_path) REFERENCES files(path)
                );

                -- Architecture patterns detected
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT,
                    line_number INTEGER,
                    description TEXT
                );

                -- Indexes for fast queries
                CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
                CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type);
                CREATE INDEX IF NOT EXISTS idx_symbols_category ON symbols(category);
                CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
                CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol);
                CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_path);
                CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);

                -- Metadata table
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

            # Migrate: add file_type column to old DBs that lack it
            cols = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
            if "file_type" not in cols:
                conn.execute('ALTER TABLE files ADD COLUMN file_type TEXT DEFAULT "source"')
            # Index on file_type (after migration ensures column exists)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type)")

    def should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored.

        Uses shared ignore logic from ignore_patterns module.
        Automatically excludes hidden directories (starting with '.').
        """
        return should_ignore_path(path, skip_dirs=self.effective_skip_dirs)

    def detect_language(self, path: Path) -> Optional[str]:
        """Detect language from file extension."""
        return self.LANGUAGE_EXTENSIONS.get(path.suffix.lower())

    def categorize_symbol(self, name: str) -> Optional[str]:
        """Determine architecture category from symbol name."""
        for category, patterns in ARCHITECTURE_PATTERNS.items():
            for pattern in patterns:
                if pattern in name:
                    return category
        return None

    async def index_codebase(self, force: bool = False) -> Dict[str, Any]:
        """Index the entire codebase with incremental update support.

        Handles:
        - New files: Index and add symbols
        - Modified files: Delete old symbols, re-index
        - Deleted files: Remove symbols from database
        - Unchanged files: Skip (unless force=True)

        Args:
            force: Force full reindex even if files haven't changed

        Returns:
            Statistics about the indexing operation
        """
        start_time = time.time()
        stats = {
            "files_indexed": 0,
            "files_skipped": 0,
            "files_deleted": 0,
            "files_with_errors": 0,  # Files that had parse errors but were still indexed
            "files_failed": 0,  # Files that completely failed to index
            "symbols_found": 0,
            "patterns_detected": 0,
            "languages": {},
            "errors": [],  # List of (file_path, error_type, error_msg) for debugging
        }

        print(f"🔍 Indexing symbols in {self.root}")

        # Collect all current source files
        current_files: Set[str] = set()
        source_files = []
        search_paths = (
            [self._resolve_search_path(d) for d in self.include_dirs]
            if self.include_dirs
            else [self.root]
        )

        for search_path in search_paths:
            if not search_path.is_dir():
                continue
            for ext in self.LANGUAGE_EXTENSIONS:
                for file_path in search_path.rglob(f"*{ext}"):
                    if not self.should_ignore(file_path) and file_path.is_file():
                        source_files.append(file_path)
                        current_files.add(self._relative_path(file_path))

        print(f"Found {len(source_files)} source files")

        with sqlite3.connect(str(self._db_path)) as conn:
            # Step 1: Handle deleted files - remove symbols for files that no longer exist
            cursor = conn.execute("SELECT path FROM files")
            indexed_files = {row[0] for row in cursor}
            deleted_files = indexed_files - current_files

            for deleted_path in deleted_files:
                self._delete_file_data(conn, deleted_path)
                stats["files_deleted"] += 1

            if deleted_files:
                print(f"🗑️  Removed {len(deleted_files)} deleted files from index")

            # Step 2: Index new and modified files
            for file_path in source_files:
                language = self.detect_language(file_path)
                if not language:
                    continue

                rel_path = self._relative_path(file_path)

                # Check if file needs reindexing
                if not force and not self._needs_reindex(conn, file_path):
                    stats["files_skipped"] += 1
                    continue

                # Delete old data for modified files before re-indexing
                if rel_path in indexed_files:
                    self._delete_file_data(conn, rel_path)

                # Extract symbols based on language - robust handling for imperfect codebases
                try:
                    symbols, imports, parse_error = self._extract_symbols_robust(
                        file_path, language
                    )

                    # Store file info (even if parsing had errors)
                    self._store_file(conn, file_path, language, len(symbols), len(imports))

                    # Store symbols
                    for symbol in symbols:
                        self._store_symbol(conn, symbol)
                        stats["symbols_found"] += 1

                    # Store imports
                    for imp in imports:
                        conn.execute(
                            "INSERT INTO imports (file_path, import_name) VALUES (?, ?)",
                            (rel_path, imp),
                        )

                    stats["files_indexed"] += 1
                    stats["languages"][language] = stats["languages"].get(language, 0) + 1

                    # Track files that had parse errors but were still indexed
                    if parse_error:
                        stats["files_with_errors"] += 1
                        stats["errors"].append((rel_path, "parse_error", parse_error))
                        logger.debug(f"Indexed {rel_path} with parse errors: {parse_error}")

                except UnicodeDecodeError as e:
                    # Binary file or encoding issue - skip but track
                    stats["files_failed"] += 1
                    stats["errors"].append((rel_path, "encoding", str(e)))
                    logger.debug(f"Skipping binary/encoding issue: {rel_path}")

                except Exception as e:
                    # Unexpected error - log at warning level for visibility
                    stats["files_failed"] += 1
                    error_type = type(e).__name__
                    stats["errors"].append((rel_path, error_type, str(e)))
                    logger.warning(f"Failed to index {rel_path}: {error_type}: {e}")

            # Step 3: Refresh architecture patterns (incremental-aware)
            if stats["files_indexed"] > 0 or stats["files_deleted"] > 0 or force:
                if force:
                    # Full reindex: clear all patterns and regenerate (destructive but complete)
                    conn.execute("DELETE FROM patterns")
                    patterns = self._detect_patterns(conn)
                    for pattern in patterns:
                        conn.execute(
                            """INSERT INTO patterns (pattern_name, pattern_type, file_path, symbol_name, line_number, description)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            pattern,
                        )
                        stats["patterns_detected"] += 1
                else:
                    # Incremental update: use INSERT OR REPLACE (preserves existing patterns)
                    # This prevents wiping patterns from unchanged files
                    patterns = self._detect_patterns(conn)
                    for pattern in patterns:
                        conn.execute(
                            """INSERT OR REPLACE INTO patterns (pattern_name, pattern_type, file_path, symbol_name, line_number, description)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            pattern,
                        )
                        stats["patterns_detected"] += 1

            # Update metadata
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("last_indexed", str(time.time())),
            )
            conn.commit()

        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)

        if stats["files_indexed"] > 0 or stats["files_deleted"] > 0:
            msg = f"✅ Indexed {stats['files_indexed']} files, {stats['symbols_found']} symbols in {elapsed:.2f}s"
            # Add warning for files with issues
            if stats["files_with_errors"] > 0:
                msg += f" ({stats['files_with_errors']} with parse errors)"
            if stats["files_failed"] > 0:
                msg += f" ({stats['files_failed']} failed)"
            print(msg)

            # Show total database stats for context
            db_stats = self.get_stats()
            print(f"  → {db_stats['total_symbols']:,} total symbols in database")

            # Show summary of failed files if any (helps users fix their code)
            if stats["files_failed"] > 0 and stats["files_failed"] <= 10:
                print("  ⚠️  Failed files:")
                for path, error_type, _error_msg in stats["errors"]:
                    if error_type != "parse_error":
                        print(f"     {path}: {error_type}")
            elif stats["files_failed"] > 10:
                print(
                    f"  ⚠️  {stats['files_failed']} files failed to index (run with --verbose for details)"
                )
        else:
            # Even when no files changed, show total stats
            db_stats = self.get_stats()
            print(f"✅ Index up to date ({stats['files_skipped']} files unchanged)")
            print(f"  → {db_stats['total_symbols']:,} total symbols in database")

        return stats

    def _delete_file_data(self, conn: sqlite3.Connection, rel_path: str) -> None:
        """Delete all data associated with a file.

        Args:
            conn: Database connection
            rel_path: Relative path of file to delete
        """
        conn.execute("DELETE FROM symbols WHERE file_path = ?", (rel_path,))
        conn.execute("DELETE FROM imports WHERE file_path = ?", (rel_path,))
        conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))

    def _needs_reindex(self, conn: sqlite3.Connection, file_path: Path) -> bool:
        """Check if file needs reindexing based on modification time."""
        rel_path = self._relative_path(file_path)
        cursor = conn.execute(
            "SELECT last_modified, content_hash FROM files WHERE path = ?", (rel_path,)
        )
        row = cursor.fetchone()

        if not row:
            return True

        current_mtime = file_path.stat().st_mtime
        return current_mtime > row[0]

    def _store_file(
        self,
        conn: sqlite3.Connection,
        file_path: Path,
        language: str,
        symbol_count: int,
        import_count: int,
    ) -> None:
        """Store file metadata including file type classification."""
        stat = file_path.stat()
        content = file_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        lines = content.count(b"\n") + 1
        rel_path = self._relative_path(file_path)

        # Determine file type: config if extension in CONFIG_EXTENSIONS, else source
        file_ext = file_path.suffix.lower()
        file_type = "config" if file_ext in self.CONFIG_EXTENSIONS else "source"

        conn.execute(
            """INSERT OR REPLACE INTO files
               (path, language, size, lines, last_modified, indexed_at, content_hash, symbol_count, import_count, file_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rel_path,
                language,
                stat.st_size,
                lines,
                stat.st_mtime,
                time.time(),
                content_hash,
                symbol_count,
                import_count,
                file_type,
            ),
        )

    def _store_symbol(self, conn: sqlite3.Connection, symbol: SymbolInfo) -> None:
        """Store a symbol in the database."""
        modifiers_str = ",".join(symbol.modifiers) if symbol.modifiers else None

        conn.execute(
            """INSERT INTO symbols
               (name, symbol_type, file_path, line_number, language, category,
                docstring, signature, parent_symbol, modifiers, is_exported, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol.name,
                symbol.symbol_type,
                symbol.file_path,
                symbol.line_number,
                symbol.language,
                symbol.category,
                symbol.docstring,
                symbol.signature,
                symbol.parent_symbol,
                modifiers_str,
                1 if symbol.is_exported else 0,
                symbol.content_hash,
            ),
        )

    def _extract_symbols_robust(
        self, file_path: Path, language: str
    ) -> Tuple[List[SymbolInfo], List[str], Optional[str]]:
        """Extract symbols with robust error handling for imperfect codebases.

        Users may have codebases with syntax errors, incomplete code, or other issues.
        This method still extracts what it can and reports errors without failing.

        Returns:
            Tuple of (symbols, imports, parse_error)
            - parse_error is None if successful, otherwise contains the error message
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        rel_path = self._relative_path(file_path)
        parse_error = None

        if language == "python":
            symbols, imports = self._extract_python_symbols_robust(content, rel_path)
            # Check if AST parsing failed by comparing with what we should have found
            if not symbols and "def " in content or "class " in content:
                # AST probably failed, try to detect syntax error for reporting
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    parse_error = f"SyntaxError line {e.lineno}: {e.msg}"
        else:
            symbols, imports = self._extract_generic_symbols(content, rel_path, language)

        return symbols, imports, parse_error

    def _extract_python_symbols_robust(
        self, content: str, rel_path: str
    ) -> Tuple[List[SymbolInfo], List[str]]:
        """Extract Python symbols with fallback to regex for files with syntax errors.

        First tries AST parsing. If that fails due to SyntaxError, falls back
        to regex-based extraction which can still find top-level definitions.
        """
        try:
            tree = ast.parse(content)
            return self._extract_python_symbols_from_ast(tree, rel_path, content)
        except SyntaxError:
            # AST failed - use regex fallback for partial extraction
            return self._extract_python_symbols_regex_fallback(content, rel_path)

    def _extract_python_symbols_regex_fallback(
        self, content: str, rel_path: str
    ) -> Tuple[List[SymbolInfo], List[str]]:
        """Extract Python symbols using regex when AST parsing fails.

        This allows indexing files with syntax errors, which is common in
        codebases under active development or with incomplete code.
        """
        symbols = []
        imports = []
        lines = content.split("\n")

        # Regex patterns for top-level definitions
        class_pattern = re.compile(r"^class\s+(\w+)\s*[:\(]")
        func_pattern = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(")
        import_pattern = re.compile(r"^(?:from\s+[\w.]+\s+)?import\s+(.+)")

        for line_num, line in enumerate(lines, start=1):
            stripped = line.lstrip()

            # Check for class definition
            class_match = class_pattern.match(stripped)
            if class_match and not line.startswith(" ") and not line.startswith("\t"):
                name = class_match.group(1)
                symbols.append(
                    SymbolInfo(
                        name=name,
                        symbol_type="class",
                        file_path=rel_path,
                        line_number=line_num,
                        language="python",
                        category=self.categorize_symbol(name),
                        docstring=None,
                        modifiers=["regex_extracted"],  # Mark as regex-extracted
                    )
                )
                continue

            # Check for function definition
            func_match = func_pattern.match(stripped)
            if func_match and not line.startswith(" ") and not line.startswith("\t"):
                name = func_match.group(1)
                modifiers = ["regex_extracted"]
                if "async " in stripped:
                    modifiers.append("async")
                symbols.append(
                    SymbolInfo(
                        name=name,
                        symbol_type="function",
                        file_path=rel_path,
                        line_number=line_num,
                        language="python",
                        category=self.categorize_symbol(name),
                        signature=f"{name}(...)",
                        modifiers=modifiers,
                    )
                )
                continue

            # Check for imports
            import_match = import_pattern.match(stripped)
            if import_match:
                import_text = import_match.group(1)
                # Handle multiple imports: import a, b, c
                for imp in import_text.split(","):
                    imp = imp.strip().split(" as ")[0].strip()
                    if imp and not imp.startswith("("):
                        imports.append(imp)

        return symbols, imports

    def _extract_python_symbols_from_ast(
        self, tree: ast.AST, rel_path: str, content: str
    ) -> Tuple[List[SymbolInfo], List[str]]:
        """Extract symbols from a parsed AST tree."""
        symbols = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                category = self.categorize_symbol(node.name)
                is_abstract = (
                    any(isinstance(b, ast.Name) and b.id == "ABC" for b in node.bases)
                    or "Abstract" in node.name
                )

                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        symbol_type="class",
                        file_path=rel_path,
                        line_number=node.lineno,
                        language="python",
                        category=category,
                        docstring=ast.get_docstring(node),
                        modifiers=["abstract"] if is_abstract else [],
                    )
                )

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Build signature
                args = [arg.arg for arg in node.args.args]
                signature = f"{node.name}({', '.join(args)})"

                modifiers = []
                if isinstance(node, ast.AsyncFunctionDef):
                    modifiers.append("async")
                if node.name.startswith("_"):
                    modifiers.append("private")

                # Check for decorators
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        if dec.id == "staticmethod":
                            modifiers.append("static")
                        elif dec.id == "classmethod":
                            modifiers.append("classmethod")
                        elif dec.id == "property":
                            modifiers.append("property")

                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        symbol_type="function",
                        file_path=rel_path,
                        line_number=node.lineno,
                        language="python",
                        category=self.categorize_symbol(node.name),
                        docstring=ast.get_docstring(node),
                        signature=signature,
                        modifiers=modifiers,
                    )
                )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return symbols, imports

    def _extract_generic_symbols(
        self, content: str, rel_path: str, language: str
    ) -> Tuple[List[SymbolInfo], List[str]]:
        """Extract symbols from any language robustly.

        Strategy:
        1. Try tree-sitter parsing (if language package installed)
        2. Fall back to regex patterns for unsupported languages or parse errors

        This handles imperfect codebases gracefully.
        """
        # Try tree-sitter first for better accuracy
        try:
            symbols, imports = self._extract_generic_symbols_treesitter(content, rel_path, language)
            if symbols:  # Got results, use them
                return symbols, imports
        except Exception as e:
            # Tree-sitter failed (not installed, parse error, etc.)
            # Fall through to regex fallback
            logger.debug(f"Tree-sitter failed for {language}: {e}, using regex fallback")

        # Regex fallback for all languages
        return self._extract_generic_symbols_regex(content, rel_path, language)

    def _extract_generic_symbols_treesitter(
        self, content: str, rel_path: str, language: str
    ) -> Tuple[List[SymbolInfo], List[str]]:
        """Extract symbols using tree-sitter for accurate parsing.

        Tree-sitter is error-tolerant and can parse files with syntax errors,
        making it ideal for imperfect codebases.
        """
        from victor_coding.codebase.tree_sitter_manager import (
            get_parser,
            LANGUAGE_MODULES,
        )

        # Check if language is supported
        if language not in LANGUAGE_MODULES:
            return [], []

        parser = get_parser(language)
        tree = parser.parse(content.encode("utf-8"))
        root = tree.root_node

        symbols = []
        imports = []

        # Language-specific node type queries
        # Tree-sitter node types vary by language grammar
        # Format: (node_type, name_field, symbol_type)
        # name_field can be a field name or None for special handling
        SYMBOL_QUERIES = {
            "javascript": [
                ("class_declaration", "name", "class"),
                ("function_declaration", "name", "function"),
                ("arrow_function", "name", "function"),
                ("method_definition", "name", "method"),
            ],
            "typescript": [
                ("class_declaration", "name", "class"),
                ("interface_declaration", "name", "interface"),
                ("type_alias_declaration", "name", "type"),
                ("function_declaration", "name", "function"),
                ("method_definition", "name", "method"),
                ("enum_declaration", "name", "enum"),
            ],
            "go": [
                # Go uses type_spec inside type_declaration - needs special handling
                ("type_spec", None, "type"),  # Special: extract type_identifier child
                ("function_declaration", "name", "function"),
                ("method_declaration", "name", "method"),
            ],
            "rust": [
                ("struct_item", "name", "struct"),
                ("enum_item", "name", "enum"),
                ("trait_item", "name", "trait"),
                ("impl_item", "type", "impl"),
                ("function_item", "name", "function"),
                ("mod_item", "name", "module"),
            ],
            "java": [
                ("class_declaration", "name", "class"),
                ("interface_declaration", "name", "interface"),
                ("enum_declaration", "name", "enum"),
                ("method_declaration", "name", "method"),
            ],
        }

        queries = SYMBOL_QUERIES.get(language, [])

        def walk_tree(node):
            """Recursively walk tree and extract symbols."""
            for query_type, name_field, default_symbol_type in queries:
                if node.type == query_type:
                    name = None
                    actual_symbol_type = default_symbol_type

                    if name_field is None:
                        # Special handling for Go type_spec: name is in type_identifier child
                        if language == "go" and node.type == "type_spec":
                            for child in node.children:
                                if child.type == "type_identifier":
                                    name = content[child.start_byte : child.end_byte]
                                    # Determine if it's struct, interface, or other
                                    for sibling in node.children:
                                        if sibling.type == "struct_type":
                                            actual_symbol_type = "struct"
                                            break
                                        elif sibling.type == "interface_type":
                                            actual_symbol_type = "interface"
                                            break
                                    break
                    else:
                        # Standard field-based name extraction
                        name_node = node.child_by_field_name(name_field)
                        if name_node:
                            name = content[name_node.start_byte : name_node.end_byte]

                    if name:
                        symbols.append(
                            SymbolInfo(
                                name=name,
                                symbol_type=actual_symbol_type,
                                file_path=rel_path,
                                line_number=node.start_point[0] + 1,
                                language=language,
                                category=self.categorize_symbol(name),
                                modifiers=["treesitter_extracted"],
                            )
                        )
                        break

            # Extract imports
            if node.type in (
                "import_statement",
                "import_declaration",
                "use_declaration",
            ):
                import_text = content[node.start_byte : node.end_byte]
                # Extract module name (simplified)
                import_match = re.search(r'["\']([^"\']+)["\']', import_text)
                if import_match:
                    imports.append(import_match.group(1))

            for child in node.children:
                walk_tree(child)

        walk_tree(root)
        return symbols, imports

    def _extract_generic_symbols_regex(
        self, content: str, rel_path: str, language: str
    ) -> Tuple[List[SymbolInfo], List[str]]:
        """Extract symbols from any language using regex patterns.

        This is the fallback for languages without tree-sitter support
        or when tree-sitter parsing fails.
        """
        symbols = []
        imports = []

        # Universal patterns for symbol detection
        patterns = [
            # Classes (JS/TS/Java/C#/PHP/Ruby)
            (
                r"(?:export\s+)?(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?class\s+([A-Z][a-zA-Z0-9_]*)",
                "class",
            ),
            # Interfaces (TS/Java/C#/Go)
            (r"(?:export\s+)?interface\s+([A-Z][a-zA-Z0-9_]*)", "interface"),
            # Structs (Go/Rust/C)
            (r"(?:pub\s+)?struct\s+([A-Z][a-zA-Z0-9_]*)", "struct"),
            # Type aliases (TS/Go/Rust)
            (r"(?:export\s+)?type\s+([A-Z][a-zA-Z0-9_]*)\s*[=<]", "type"),
            # Enums (TS/Rust/Java/C#)
            (r"(?:export\s+)?(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9_]*)", "enum"),
            # Traits (Rust)
            (r"(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9_]*)", "trait"),
            # Modules (Ruby/Elixir)
            (r"(?:defmodule|module)\s+([A-Z][a-zA-Z0-9_:]*)", "module"),
            # Functions (Go/Rust - top-level)
            (r"^(?:pub\s+)?fn\s+([a-z_][a-zA-Z0-9_]*)", "function"),
            (r"^func\s+(?:\([^)]+\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)", "function"),
            # React/Vue components
            (
                r"(?:export\s+)?(?:default\s+)?(?:const|function)\s+([A-Z][a-zA-Z0-9_]*)\s*[=\(]",
                "component",
            ),
        ]

        # Import patterns
        import_patterns = [
            r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",  # ES modules
            r"import\s+['\"]([^'\"]+)['\"]",  # Go imports
            r"require\s*\(['\"]([^'\"]+)['\"]\)",  # CommonJS
            r"use\s+([a-zA-Z_][a-zA-Z0-9_:]*)",  # Rust use
        ]

        lines = content.split("\n")

        for line_no, line in enumerate(lines, 1):
            # Check for symbols
            for pattern, symbol_type in patterns:
                match = re.search(pattern, line, re.MULTILINE)
                if match:
                    name = match.group(1)

                    # Extract modifiers
                    modifiers = []
                    if "export" in line.lower():
                        modifiers.append("export")
                    if "public" in line.lower():
                        modifiers.append("public")
                    if "private" in line.lower():
                        modifiers.append("private")
                    if "async" in line.lower():
                        modifiers.append("async")
                    if "static" in line.lower():
                        modifiers.append("static")
                    if "abstract" in line.lower():
                        modifiers.append("abstract")

                    # Get docstring from previous lines
                    docstring = self._extract_docstring_before(lines, line_no - 1)

                    symbols.append(
                        SymbolInfo(
                            name=name,
                            symbol_type=symbol_type,
                            file_path=rel_path,
                            line_number=line_no,
                            language=language,
                            category=self.categorize_symbol(name),
                            docstring=docstring,
                            modifiers=modifiers,
                            is_exported="export" in line.lower() or "pub " in line.lower(),
                        )
                    )
                    break  # One match per line

            # Check for imports
            for pattern in import_patterns:
                for match in re.finditer(pattern, line):
                    imports.append(match.group(1))

        return symbols, imports

    def _extract_docstring_before(self, lines: List[str], line_idx: int) -> Optional[str]:
        """Extract docstring/comment from lines before the symbol."""
        if line_idx <= 0:
            return None

        prev_line = lines[line_idx - 1].strip()

        # Check for various comment styles
        comment_markers = ["///", "/**", "/*", "//", "#", '"""', "'''"]
        for marker in comment_markers:
            if prev_line.startswith(marker):
                # Clean up the comment
                doc = prev_line.lstrip(marker).rstrip("*/").strip()
                return doc[:200] if doc else None

        return None

    def _detect_patterns(self, conn: sqlite3.Connection) -> List[Tuple]:
        """Detect architecture patterns from indexed symbols."""
        patterns = []

        # Check for Provider pattern
        cursor = conn.execute("""SELECT DISTINCT category, COUNT(*) as cnt
               FROM symbols WHERE category IS NOT NULL
               GROUP BY category HAVING cnt >= 2""")
        for row in cursor:
            category, count = row
            patterns.append(
                (
                    f"{category.title()} Pattern",
                    category,
                    "",  # file_path
                    None,  # symbol_name
                    None,  # line_number
                    f"Found {count} {category} components",
                )
            )

        # Check for base classes (inheritance pattern)
        cursor = conn.execute("""SELECT name, file_path, line_number FROM symbols
               WHERE symbol_type = 'class' AND
               (name LIKE 'Base%' OR name LIKE 'Abstract%' OR name LIKE 'I%')""")
        for row in cursor:
            name, file_path, line_no = row
            patterns.append(
                (
                    f"Base Class: {name}",
                    "inheritance",
                    file_path,
                    name,
                    line_no,
                    "Abstract base class for inheritance",
                )
            )

        return patterns

    # Query methods

    def find_by_category(self, category: str, limit: int = 50) -> List[SymbolInfo]:
        """Find symbols by architecture category."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                """SELECT name, symbol_type, file_path, line_number, language,
                          category, docstring, signature, parent_symbol, modifiers
                   FROM symbols WHERE category = ? LIMIT ?""",
                (category, limit),
            )
            return [self._row_to_symbol(row) for row in cursor]

    def find_by_type(self, symbol_type: str, limit: int = 100) -> List[SymbolInfo]:
        """Find symbols by type (class, function, interface, etc.)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                """SELECT name, symbol_type, file_path, line_number, language,
                          category, docstring, signature, parent_symbol, modifiers
                   FROM symbols WHERE symbol_type = ? LIMIT ?""",
                (symbol_type, limit),
            )
            return [self._row_to_symbol(row) for row in cursor]

    def find_by_name_pattern(self, pattern: str, limit: int = 50) -> List[SymbolInfo]:
        """Find symbols matching a name pattern (use % as wildcard)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                """SELECT name, symbol_type, file_path, line_number, language,
                          category, docstring, signature, parent_symbol, modifiers
                   FROM symbols WHERE name LIKE ? LIMIT ?""",
                (pattern, limit),
            )
            return [self._row_to_symbol(row) for row in cursor]

    def find_key_components(self, limit: int = 20) -> List[SymbolInfo]:
        """Find key architectural components ranked by graph connectivity.

        Uses ATTACH DATABASE to join the symbol index with the code graph,
        ranking classes by in-degree (how many other symbols reference them).
        Falls back to heuristic ranking if the graph database is unavailable.
        """
        graph_db = self._db_path.parent / "project.db"
        use_graph = graph_db.exists()

        with sqlite3.connect(str(self._db_path)) as conn:
            if use_graph:
                try:
                    conn.execute("ATTACH DATABASE ? AS graph_db", (str(graph_db),))
                    cursor = conn.execute(
                        """SELECT s.name, s.symbol_type, s.file_path, s.line_number,
                                  s.language, s.category, s.docstring, s.signature,
                                  s.parent_symbol, s.modifiers
                           FROM symbols s
                           LEFT JOIN graph_db.graph_node g
                             ON g.name = s.name
                             AND g.file = s.file_path
                             AND g.type = 'class'
                           LEFT JOIN (
                             SELECT dst, COUNT(*) as in_deg
                             FROM graph_db.graph_edge
                             GROUP BY dst
                           ) e ON e.dst = g.node_id
                           WHERE s.symbol_type IN ('class', 'interface', 'struct', 'trait')
                             AND s.file_path NOT LIKE 'tests/%'
                             AND s.file_path NOT LIKE 'vscode-%/out/%'
                             AND s.file_path NOT LIKE 'site/%'
                             AND s.file_path NOT LIKE 'archive/%'
                           ORDER BY COALESCE(e.in_deg, 0) DESC
                           LIMIT ?""",
                        (limit,),
                    )
                    results = [self._row_to_symbol(row) for row in cursor]
                    conn.execute("DETACH DATABASE graph_db")
                    if results:
                        return results
                except Exception:
                    pass  # Fall through to heuristic

            # Heuristic fallback: category + name pattern ranking
            cursor = conn.execute(
                """SELECT name, symbol_type, file_path, line_number, language,
                          category, docstring, signature, parent_symbol, modifiers
                   FROM symbols
                   WHERE category IS NOT NULL
                   AND symbol_type IN ('class', 'interface', 'struct', 'trait')
                   AND file_path NOT LIKE 'tests/%'
                   AND file_path NOT LIKE 'vscode-%/out/%'
                   ORDER BY
                     CASE category
                       WHEN 'service' THEN 1
                       WHEN 'controller' THEN 2
                       WHEN 'provider' THEN 3
                       WHEN 'factory' THEN 4
                       WHEN 'model' THEN 5
                       ELSE 6
                     END,
                     CASE
                       WHEN name LIKE '%Agent%' OR name LIKE '%Orchestrat%'
                         OR name LIKE '%Engine%' OR name LIKE '%Registry%'
                         THEN 0
                       ELSE 1
                     END,
                     name
                   LIMIT ?""",
                (limit,),
            )
            return [self._row_to_symbol(row) for row in cursor]

    def find_named_implementations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find named implementations grouped by domain.

        Detects patterns like:
        - src/storage/engines/impls/sst/ -> Storage Engines: SST
        - src/graph/engines/orion/ -> Graph Engines: ORION
        - src/providers/anthropic/ -> Providers: Anthropic

        Returns:
            Dict mapping domain names to lists of implementations with metadata.
        """
        # Skip these generic file/directory names (language-agnostic)
        # Entry points, modules, and utility directories that shouldn't be impl names
        SKIP_NAMES = {
            "mod",
            "lib",
            "main",
            "index",
            "init",
            "__init__",
            "core",
            "utils",
            "common",
            "factory",
            "helpers",
            "shared",
            "generic",
            "base",
            "impls",
            "tests",
            "test",
            "spec",
            "specs",
        }

        # Note: We detect files vs directories structurally - paths from the symbol
        # index always have the file as the last component, so we check position
        # rather than extensions. This is inherently language-agnostic.

        # Domain inference from path prefixes
        DOMAIN_PREFIXES = {
            "storage": ["storage", "db", "persistence", "cache"],
            "graph": ["graph", "network", "relation"],
            "index": ["index", "search", "query"],
            "compute": ["compute", "processing", "ml", "ai"],
            "network": ["network", "http", "rpc", "grpc"],
            "auth": ["auth", "security", "identity"],
        }

        results: Dict[str, List[Dict[str, Any]]] = {}

        with sqlite3.connect(str(self._db_path)) as conn:
            # Get all struct/class symbols with their paths
            cursor = conn.execute(
                """SELECT DISTINCT file_path, name, symbol_type, docstring, line_number
                   FROM symbols
                   WHERE symbol_type IN ('struct', 'class', 'trait', 'interface')
                   ORDER BY file_path"""
            )

            for row in cursor:
                file_path, name, sym_type, docstring, line_number = row
                path_lower = file_path.lower()
                parts = file_path.split("/")

                # Strategy 1: Look for engines/impls/NAME pattern (most specific)
                impl_name = None
                impl_type = None

                # Detect sub-component types from path context (generic patterns)
                # e.g., codecs/, encoding/, serialization/, compression/
                SUBCOMPONENT_DIRS = {
                    "codec",
                    "codecs",
                    "encoding",
                    "serialization",
                    "compression",
                    "format",
                    "protocol",
                    "ops",
                }

                for i, part in enumerate(parts):
                    # Check if path contains a subcomponent directory - classify accordingly
                    # e.g., engines/core/ops/codec/impls/baseline/ -> Codecs: BASELINE
                    part_lower = part.lower()
                    if part_lower in SUBCOMPONENT_DIRS:
                        # Look for impls/ after this subcomponent dir
                        for j in range(i + 1, len(parts)):
                            # j+1 < len(parts)-1 ensures next_part is a directory, not the file
                            if parts[j] == "impls" and j + 1 < len(parts) - 1:
                                next_part = parts[j + 1].lower()
                                if next_part not in SKIP_NAMES:
                                    impl_name = parts[j + 1].upper()
                                    impl_type = (
                                        part_lower + "s"
                                        if not part_lower.endswith("s")
                                        else part_lower
                                    )
                                    break
                        if impl_name:
                            break
                        continue

                    # engines/impls/sst/... -> SST (true storage engine)
                    # i+1 < len(parts)-1 ensures next_part is a directory, not the file
                    if part == "impls" and i + 1 < len(parts) - 1:
                        next_part = parts[i + 1].lower()
                        if next_part not in SKIP_NAMES:
                            impl_name = parts[i + 1].upper()
                            impl_type = "engines"
                            break
                    # engines/orion/... -> ORION (when no impls dir)
                    elif part == "engines" and i + 1 < len(parts) - 1:
                        next_part = parts[i + 1].lower()
                        if next_part not in SKIP_NAMES:
                            impl_name = parts[i + 1].upper()
                            impl_type = "engines"
                            break
                    # providers/anthropic/... -> ANTHROPIC
                    elif part == "providers" and i + 1 < len(parts) - 1:
                        next_part = parts[i + 1].lower()
                        if next_part not in SKIP_NAMES:
                            impl_name = parts[i + 1].upper()
                            impl_type = "providers"
                            break
                    # backends/local_rocksdb/... -> LOCAL_ROCKSDB
                    elif part == "backends" and i + 1 < len(parts) - 1:
                        next_part = parts[i + 1].lower()
                        if next_part not in SKIP_NAMES:
                            impl_name = parts[i + 1].upper()
                            impl_type = "backends"
                            break
                    # adapters/some_adapter/... -> SOME_ADAPTER
                    elif part == "adapters" and i + 1 < len(parts) - 1:
                        next_part = parts[i + 1].lower()
                        if next_part not in SKIP_NAMES:
                            impl_name = parts[i + 1].upper()
                            impl_type = "adapters"
                            break

                if not impl_name or not impl_type:
                    continue

                # Infer domain from path
                domain = "other"
                for d, prefixes in DOMAIN_PREFIXES.items():
                    if any(p in path_lower for p in prefixes):
                        domain = d
                        break

                # Build domain key
                domain_key = f"{domain.title()} {impl_type.title()}"

                if domain_key not in results:
                    results[domain_key] = []

                # Check if we already have this implementation
                existing = [r for r in results[domain_key] if r["name"] == impl_name]
                if not existing:
                    results[domain_key].append(
                        {
                            "name": impl_name,
                            "path": file_path,
                            "line": line_number,
                            "description": (docstring or "")[:60] if docstring else "",
                            "primary_symbol": name,
                        }
                    )

        return results

    def find_performance_hints(self) -> Dict[str, List[Dict[str, str]]]:
        """Extract performance hints from docstrings and comments.

        Looks for patterns like:
        - "~5ms", "5ms latency", "< 10ms"
        - "1M+ ops/sec", "10K vectors"
        - "O(n log n)", "O(1)"

        Returns:
            Dict mapping file paths to lists of performance hints.
        """
        import re

        # Patterns to detect performance metrics
        METRIC_PATTERNS = [
            (r"[~<>≈]?\s*\d+(?:\.\d+)?\s*(?:ms|μs|ns|s)\b", "latency"),
            (r"\d+[KMB]?\+?\s*(?:ops?/sec|vectors?|items?|entries)", "throughput"),
            (r"O\([^)]+\)", "complexity"),
            (r"(?:latency|throughput|performance)[:\s]+[^.]{10,50}", "description"),
        ]

        results: Dict[str, List[Dict[str, str]]] = {}

        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute("""SELECT file_path, docstring FROM symbols
                   WHERE docstring IS NOT NULL AND docstring != ''""")

            for file_path, docstring in cursor:
                if not docstring:
                    continue

                hints = []
                for pattern, hint_type in METRIC_PATTERNS:
                    matches = re.findall(pattern, docstring, re.IGNORECASE)
                    for match in matches:
                        hints.append({"type": hint_type, "value": match.strip()})

                if hints:
                    if file_path not in results:
                        results[file_path] = []
                    results[file_path].extend(hints)

        return results

    def get_detected_patterns(self) -> List[Dict[str, Any]]:
        """Get all detected architecture patterns."""
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "SELECT pattern_name, pattern_type, file_path, symbol_name, line_number, description FROM patterns"
            )
            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "file_path": row[2],
                    "symbol_name": row[3],
                    "line_number": row[4],
                    "description": row[5],
                }
                for row in cursor
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the symbol store.

        Returns comprehensive stats including:
        - total_files: All indexed files
        - total_symbols: Code symbols (classes, functions, methods)
        - graph_nodes: Total graph nodes (files + symbols + imports + etc.)
        - symbols_by_type: Breakdown by symbol type (class, function, etc.)
        - files_by_language: Breakdown by programming language
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            stats = {}

            # File count by language
            cursor = conn.execute("SELECT language, COUNT(*) FROM files GROUP BY language")
            stats["files_by_language"] = dict(cursor.fetchall())

            # Symbol count by type
            cursor = conn.execute("SELECT symbol_type, COUNT(*) FROM symbols GROUP BY symbol_type")
            symbols_by_type = dict(cursor.fetchall())

            # Total files
            cursor = conn.execute("SELECT COUNT(*) FROM files")
            stats["total_files"] = cursor.fetchone()[0]

            # Total code symbols (classes, functions, methods - what developers write)
            cursor = conn.execute("SELECT COUNT(*) FROM symbols")
            stats["total_symbols"] = cursor.fetchone()[0]

            # Total graph nodes (files + symbols + imports as approximation)
            cursor = conn.execute("SELECT COUNT(*) FROM imports")
            import_count = cursor.fetchone()[0]
            stats["graph_nodes"] = stats["total_files"] + stats["total_symbols"] + import_count

            # Symbol breakdown by type with clear naming
            stats["classes"] = symbols_by_type.get("class", 0)
            stats["functions"] = symbols_by_type.get("function", 0)
            stats["methods"] = symbols_by_type.get("method", 0)
            stats["symbols_by_type"] = symbols_by_type

            # Symbol count by category
            cursor = conn.execute(
                "SELECT category, COUNT(*) FROM symbols WHERE category IS NOT NULL GROUP BY category"
            )
            stats["symbols_by_category"] = dict(cursor.fetchall())

            # Patterns
            cursor = conn.execute("SELECT COUNT(*) FROM patterns")
            stats["total_patterns"] = cursor.fetchone()[0]

            # Last indexed
            cursor = conn.execute("SELECT value FROM metadata WHERE key = 'last_indexed'")
            row = cursor.fetchone()
            stats["last_indexed"] = float(row[0]) if row else None

            return stats

    def _row_to_symbol(self, row: tuple) -> SymbolInfo:
        """Convert a database row to SymbolInfo."""
        # Parse modifiers from comma-separated string
        modifiers_str = row[9] if len(row) > 9 else None
        modifiers = modifiers_str.split(",") if modifiers_str else []

        return SymbolInfo(
            name=row[0],
            symbol_type=row[1],
            file_path=row[2],
            line_number=row[3],
            language=row[4],
            category=row[5],
            docstring=row[6],
            signature=row[7],
            parent_symbol=row[8],
            modifiers=modifiers,
        )

    def clear(self) -> None:
        """Clear all data from the store."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript("""
                DELETE FROM symbols;
                DELETE FROM files;
                DELETE FROM imports;
                DELETE FROM patterns;
                DELETE FROM metadata;
            """)
