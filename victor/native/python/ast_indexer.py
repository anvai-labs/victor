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

"""Pure Python AST indexer implementation.

Provides Python fallback for AST indexer hot paths:
- is_stdlib_module(): stdlib detection for PageRank filtering
- extract_identifiers(): regex-based identifier extraction

These are performance-critical operations called thousands of times
during codebase indexing. The Rust implementation uses:
- Perfect hash (phf) for O(1) stdlib lookup
- SIMD-optimized regex for identifier extraction
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from victor.native.observability import InstrumentedAccelerator

# =============================================================================
# STANDARD LIBRARY MODULES
# =============================================================================
# This set matches the STDLIB_MODULES frozenset in indexer.py
# Used to exclude stdlib imports from the codebase graph to avoid
# inflating PageRank scores with external dependencies.

STDLIB_MODULES = frozenset(
    {
        # Core builtins
        "abc",
        "asyncio",
        "builtins",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "gc",
        "hashlib",
        "heapq",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "pprint",
        "queue",
        "random",
        "re",
        "secrets",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "ssl",
        "string",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "traceback",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
        "xml",
        "zipfile",
        "zlib",
        # Typing extensions
        "typing_extensions",
        # Common third-party (excluded from graph like stdlib)
        "numpy",
        "pandas",
        "requests",
        "aiohttp",
        "httpx",
        "pydantic",
        "pytest",
        "mock",
        "unittest.mock",
        # Additional stdlib modules (Python 3.9+)
        "argparse",
        "array",
        "ast",
        "atexit",
        "base64",
        "binascii",
        "bisect",
        "bz2",
        "calendar",
        "cmath",
        "codecs",
        "concurrent",
        "configparser",
        "contextvars",
        "csv",
        "ctypes",
        "curses",
        "dbm",
        "difflib",
        "dis",
        "doctest",
        "email",
        "encodings",
        "errno",
        "faulthandler",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "fractions",
        "ftplib",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "graphlib",
        "grp",
        "gzip",
        "hmac",
        "html",
        "http",
        "imaplib",
        "imghdr",
        "ipaddress",
        "keyword",
        "linecache",
        "locale",
        "lzma",
        "mailbox",
        "mimetypes",
        "mmap",
        "modulefinder",
        "multiprocessing",
        "netrc",
        "nis",
        "nntplib",
        "numbers",
        "optparse",
        "parser",
        "pdb",
        "pkgutil",
        "poplib",
        "posix",
        "posixpath",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "site",
        "smtpd",
        "smtplib",
        "sndhdr",
        "socketserver",
        "spwd",
        "stat",
        "statistics",
        "stringprep",
        "sunau",
        "symtable",
        "sysconfig",
        "syslog",
        "tabnanny",
        "tarfile",
        "telnetlib",
        "termios",
        "test",
        "textwrap",
        "timeit",
        "tkinter",
        "token",
        "tokenize",
        "trace",
        "tracemalloc",
        "tty",
        "turtle",
        "types",
        "unicodedata",
        "uu",
        "venv",
        "wave",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xdrlib",
        "xmlrpc",
        "zipapp",
        "zipimport",
        "zoneinfo",
    }
)

# Compiled regex pattern for identifier extraction
# Matches: [A-Za-z_][A-Za-z0-9_]*
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PythonAstIndexer(InstrumentedAccelerator):
    """Pure Python implementation of AstIndexerProtocol.

    Provides fallback implementations for AST indexer hot paths.
    The Rust implementation provides significant speedups:
    - is_stdlib_module: 5-10x faster with perfect hash
    - extract_identifiers: 3-5x faster with SIMD regex
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"

    def get_version(self) -> Optional[str]:
        return self._version

    def is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module name is a standard library module.

        Args:
            module_name: Full module name (e.g., "os.path", "typing")

        Returns:
            True if the module is in the stdlib or common third-party set
        """
        with self._timed_call("stdlib_check"):
            if not module_name:
                return False

            # Check exact match first
            if module_name in STDLIB_MODULES:
                return True

            # Check top-level package (e.g., "os.path" -> "os")
            top_level = module_name.split(".")[0]
            return top_level in STDLIB_MODULES

    def batch_is_stdlib_modules(self, module_names: List[str]) -> List[bool]:
        """Check multiple module names for stdlib membership.

        Args:
            module_names: List of module names to check

        Returns:
            List of booleans, one per module name
        """
        with self._timed_call("batch_stdlib_check"):
            return [self._is_stdlib_single(name) for name in module_names]

    def _is_stdlib_single(self, module_name: str) -> bool:
        """Internal stdlib check without instrumentation."""
        if not module_name:
            return False
        if module_name in STDLIB_MODULES:
            return True
        top_level = module_name.split(".")[0]
        return top_level in STDLIB_MODULES

    def extract_identifiers(self, source: str) -> List[str]:
        """Extract all unique identifier references from source code.

        Uses regex pattern [A-Za-z_][A-Za-z0-9_]*.

        Args:
            source: Source code text

        Returns:
            List of unique identifiers found
        """
        with self._timed_call("extract_identifiers"):
            if not source:
                return []

            # Use set for deduplication
            identifiers = set(_IDENTIFIER_PATTERN.findall(source))
            return list(identifiers)

    def extract_identifiers_with_positions(self, source: str) -> List[Tuple[str, int, int]]:
        """Extract identifiers with their positions.

        Args:
            source: Source code text

        Returns:
            List of (identifier, start_offset, end_offset) tuples
        """
        with self._timed_call("extract_identifiers_with_positions"):
            if not source:
                return []

            results = []
            for match in _IDENTIFIER_PATTERN.finditer(source):
                results.append((match.group(0), match.start(), match.end()))

            return results

    def filter_stdlib_imports(self, imports: List[str]) -> Tuple[List[str], List[str]]:
        """Partition imports into stdlib and non-stdlib.

        Args:
            imports: List of import module names

        Returns:
            Tuple of (stdlib_imports, non_stdlib_imports)
        """
        with self._timed_call("filter_stdlib_imports"):
            stdlib = []
            non_stdlib = []

            for module in imports:
                if self._is_stdlib_single(module):
                    stdlib.append(module)
                else:
                    non_stdlib.append(module)

            return stdlib, non_stdlib
