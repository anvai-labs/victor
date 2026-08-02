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

"""Pure Python symbol extractor implementation.

Provides symbol extraction using Python's ast module for Python source
and regex patterns for other languages.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Set

from victor.native.observability import InstrumentedAccelerator
from victor.native.protocols import Symbol, SymbolType

# Standard library modules (Python 3.10+)
# This should be kept in sync with the Rust implementation
STDLIB_MODULES: frozenset[str] = frozenset(
    {
        "abc",
        "aifc",
        "argparse",
        "array",
        "ast",
        "asynchat",
        "asyncio",
        "asyncore",
        "atexit",
        "audioop",
        "base64",
        "bdb",
        "binascii",
        "binhex",
        "bisect",
        "builtins",
        "bz2",
        "calendar",
        "cgi",
        "cgitb",
        "chunk",
        "cmath",
        "cmd",
        "code",
        "codecs",
        "codeop",
        "collections",
        "colorsys",
        "compileall",
        "concurrent",
        "configparser",
        "contextlib",
        "contextvars",
        "copy",
        "copyreg",
        "cProfile",
        "crypt",
        "csv",
        "ctypes",
        "curses",
        "dataclasses",
        "datetime",
        "dbm",
        "decimal",
        "difflib",
        "dis",
        "distutils",
        "doctest",
        "email",
        "encodings",
        "enum",
        "errno",
        "faulthandler",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "fractions",
        "ftplib",
        "functools",
        "gc",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "graphlib",
        "grp",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "idlelib",
        "imaplib",
        "imghdr",
        "imp",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "lib2to3",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "mailbox",
        "mailcap",
        "marshal",
        "math",
        "mimetypes",
        "mmap",
        "modulefinder",
        "multiprocessing",
        "netrc",
        "nis",
        "nntplib",
        "numbers",
        "operator",
        "optparse",
        "os",
        "ossaudiodev",
        "pathlib",
        "pdb",
        "pickle",
        "pickletools",
        "pipes",
        "pkgutil",
        "platform",
        "plistlib",
        "poplib",
        "posix",
        "posixpath",
        "pprint",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "queue",
        "quopri",
        "random",
        "re",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "site",
        "smtpd",
        "smtplib",
        "sndhdr",
        "socket",
        "socketserver",
        "spwd",
        "sqlite3",
        "ssl",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "sunau",
        "symtable",
        "sys",
        "sysconfig",
        "syslog",
        "tabnanny",
        "tarfile",
        "telnetlib",
        "tempfile",
        "termios",
        "test",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "tkinter",
        "token",
        "tokenize",
        "tomllib",
        "trace",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "turtledemo",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uu",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xdrlib",
        "xml",
        "xmlrpc",
        "zipapp",
        "zipfile",
        "zipimport",
        "zlib",
        "zoneinfo",
        # Common typing extensions
        "typing_extensions",
    }
)

# Regex for extracting identifiers
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PythonSymbolExtractor(InstrumentedAccelerator):
    """Pure Python implementation of SymbolExtractorProtocol.

    Uses Python's ast module for Python source code.
    Falls back to regex for other languages.
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"

    def get_version(self) -> Optional[str]:
        return self._version

    def extract_functions(self, source: str, lang: str) -> List[Symbol]:
        """Extract function definitions from source."""
        with self._timed_call("extract_functions", lang=lang):
            if lang == "python":
                return self._extract_python_functions(source)
            # Other languages use tree-sitter via victor-coding (29 languages)
            return []

    def extract_classes(self, source: str, lang: str) -> List[Symbol]:
        """Extract class definitions from source."""
        with self._timed_call("extract_classes", lang=lang):
            if lang == "python":
                return self._extract_python_classes(source)
            # Other languages use tree-sitter via victor-coding (29 languages)
            return []

    def extract_imports(self, source: str, lang: str) -> List[str]:
        """Extract import statements from source."""
        with self._timed_call("extract_imports", lang=lang):
            if lang == "python":
                return self._extract_python_imports(source)
            # Other languages use tree-sitter via victor-coding (29 languages)
            return []

    def extract_references(self, source: str) -> List[str]:
        """Extract all identifier references from source."""
        with self._timed_call("reference_extraction"):
            return IDENTIFIER_PATTERN.findall(source)

    def is_stdlib_module(self, name: str) -> bool:
        """Check if a module name is from the standard library."""
        with self._timed_call("stdlib_check"):
            # Get the top-level module name
            top_level = name.split(".")[0]
            return top_level in STDLIB_MODULES

    def _extract_python_functions(self, source: str) -> List[Symbol]:
        """Extract functions from Python source using ast."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Determine visibility
                visibility = "private" if node.name.startswith("_") else "public"
                if node.name.startswith("__") and not node.name.endswith("__"):
                    visibility = "protected"

                # Get decorators
                decorators = tuple(self._get_decorator_name(d) for d in node.decorator_list)

                # Get docstring
                docstring = ast.get_docstring(node) or ""

                # Build signature
                signature = self._build_function_signature(node)

                symbols.append(
                    Symbol(
                        name=node.name,
                        type=SymbolType.FUNCTION,
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=signature,
                        docstring=docstring,
                        decorators=decorators,
                        visibility=visibility,
                    )
                )

        return symbols

    def _extract_python_classes(self, source: str) -> List[Symbol]:
        """Extract classes from Python source using ast."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Determine visibility
                visibility = "private" if node.name.startswith("_") else "public"

                # Get decorators
                decorators = tuple(self._get_decorator_name(d) for d in node.decorator_list)

                # Get docstring
                docstring = ast.get_docstring(node) or ""

                # Build signature with base classes
                bases = [self._get_name(b) for b in node.bases]
                signature = f"class {node.name}"
                if bases:
                    signature += f"({', '.join(bases)})"
                signature += ":"

                symbols.append(
                    Symbol(
                        name=node.name,
                        type=SymbolType.CLASS,
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=signature,
                        docstring=docstring,
                        decorators=decorators,
                        visibility=visibility,
                    )
                )

        return symbols

    def _extract_python_imports(self, source: str) -> List[str]:
        """Extract imports from Python source using ast."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        return sorted(imports)

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get the name of a decorator."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return "unknown"

    def _get_name(self, node: ast.expr) -> str:
        """Get the name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        return "unknown"

    def _build_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Build a function signature string."""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = []

        # Positional args
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {self._get_annotation(node.args.vararg.annotation)}"
            args.append(arg_str)

        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {self._get_annotation(node.args.kwarg.annotation)}"
            args.append(arg_str)

        signature = f"{prefix} {node.name}({', '.join(args)})"

        # Return annotation
        if node.returns:
            signature += f" -> {self._get_annotation(node.returns)}"

        return signature

    def _get_annotation(self, node: ast.expr) -> str:
        """Get string representation of a type annotation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_annotation(node.slice)}]"
        elif isinstance(node, ast.Tuple):
            return ", ".join(self._get_annotation(e) for e in node.elts)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return f"{self._get_annotation(node.left)} | {self._get_annotation(node.right)}"
        return "..."
