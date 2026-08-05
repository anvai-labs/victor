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

"""Tests for SymbolStore codebase indexing with multi-language support.

Tests cover:
- Python AST extraction with regex fallback for syntax errors
- TypeScript/JavaScript tree-sitter extraction
- Go, Rust, Java tree-sitter extraction
- Robust handling of imperfect codebases
- Architecture pattern detection
"""

import asyncio

import pytest

pytest.importorskip("victor_coding.codebase.symbol_store")

from victor_coding.codebase.symbol_store import SymbolStore
from victor_coding.compat.settings import get_project_paths


class TestPythonExtraction:
    """Tests for Python symbol extraction using AST with regex fallback."""

    def test_extract_python_class(self, tmp_path):
        """Test extracting Python class definitions."""
        code = '''
class UserService:
    """A service for user operations."""

    def get_user(self, user_id: str) -> dict:
        pass
'''
        (tmp_path / "service.py").write_text(code)
        store = SymbolStore(str(tmp_path))

        stats = asyncio.run(store.index_codebase())

        assert stats["files_indexed"] == 1
        classes = store.find_by_type("class")
        assert len(classes) == 1
        assert classes[0].name == "UserService"
        assert classes[0].docstring == "A service for user operations."

    def test_extract_python_function(self, tmp_path):
        """Test extracting Python function definitions."""
        code = '''
def process_data(data: list) -> dict:
    """Process input data."""
    return {"result": data}

async def fetch_data(url: str) -> bytes:
    """Fetch data from URL."""
    pass
'''
        (tmp_path / "utils.py").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        functions = store.find_by_type("function")
        assert len(functions) == 2
        names = [f.name for f in functions]
        assert "process_data" in names
        assert "fetch_data" in names

        # Check async modifier
        async_func = next(f for f in functions if f.name == "fetch_data")
        assert async_func is not None
        assert "async" in async_func.modifiers

    def test_python_syntax_error_fallback(self, tmp_path):
        """Test regex fallback for Python files with syntax errors."""
        code = '''
class GoodClass:
    """Valid class."""
    pass

def broken_function(
    # Missing closing paren - syntax error!

class AnotherClass:
    """Still extracted via regex."""
    pass
'''
        (tmp_path / "broken.py").write_text(code)
        store = SymbolStore(str(tmp_path))

        stats = asyncio.run(store.index_codebase())

        # File should be indexed despite syntax error
        assert stats["files_indexed"] == 1
        assert stats["files_with_errors"] == 1

        # Symbols should still be extracted via regex
        classes = store.find_by_type("class")
        assert len(classes) >= 2
        class_names = [c.name for c in classes]
        assert "GoodClass" in class_names
        assert "AnotherClass" in class_names

    def test_python_import_detection(self, tmp_path):
        """Test Python import detection."""
        code = """
import os
import sys
from pathlib import Path
from typing import Dict, List
"""
        (tmp_path / "imports.py").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())
        # Imports are tracked internally, verify file was indexed
        stats = store.get_stats()
        assert stats["total_files"] == 1


class TestTypeScriptExtraction:
    """Tests for TypeScript symbol extraction using tree-sitter."""

    def test_extract_typescript_class(self, tmp_path):
        """Test extracting TypeScript class definitions."""
        code = """
export class UserService {
    private db: Database;

    async getUser(id: string): Promise<User> {
        return this.db.find(id);
    }
}
"""
        (tmp_path / "service.ts").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        classes = store.find_by_type("class")
        assert len(classes) == 1
        assert classes[0].name == "UserService"

    def test_extract_typescript_interface(self, tmp_path):
        """Test extracting TypeScript interface definitions."""
        code = """
interface Config {
    apiUrl: string;
    timeout: number;
}

interface User {
    id: string;
    name: string;
}
"""
        (tmp_path / "types.ts").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        interfaces = store.find_by_type("interface")
        assert len(interfaces) == 2
        names = [i.name for i in interfaces]
        assert "Config" in names
        assert "User" in names

    def test_extract_typescript_enum(self, tmp_path):
        """Test extracting TypeScript enum definitions."""
        code = """
enum Status {
    Active = 'active',
    Inactive = 'inactive',
    Pending = 'pending'
}
"""
        (tmp_path / "enums.ts").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        enums = store.find_by_type("enum")
        assert len(enums) == 1
        assert enums[0].name == "Status"


class TestJavaScriptExtraction:
    """Tests for JavaScript symbol extraction."""

    def test_extract_javascript_class(self, tmp_path):
        """Test extracting JavaScript class definitions."""
        code = """
class Component {
    constructor(props) {
        this.props = props;
    }

    render() {
        return null;
    }
}
"""
        (tmp_path / "component.js").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        classes = store.find_by_type("class")
        assert len(classes) >= 1
        assert any(c.name == "Component" for c in classes)


class TestGoExtraction:
    """Tests for Go symbol extraction using tree-sitter."""

    @pytest.mark.skip(reason="Tree-sitter Go parser not available in CI environment")
    def test_extract_go_struct(self, tmp_path):
        """Test extracting Go struct definitions."""
        code = """
package main

type User struct {
    ID   string
    Name string
}

type Config struct {
    Port int
    Host string
}
"""
        (tmp_path / "types.go").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        # Go structs are extracted with symbol_type="struct"
        structs = store.find_by_type("struct")
        assert len(structs) == 2
        names = [s.name for s in structs]
        assert "User" in names
        assert "Config" in names

    def test_extract_go_function(self, tmp_path):
        """Test extracting Go function definitions."""
        code = """
package main

func main() {
    println("Hello")
}

func processData(data []byte) error {
    return nil
}
"""
        (tmp_path / "main.go").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        functions = store.find_by_type("function")
        names = [f.name for f in functions]
        assert "main" in names or "processData" in names


class TestRustExtraction:
    """Tests for Rust symbol extraction using tree-sitter."""

    def test_extract_rust_struct(self, tmp_path):
        """Test extracting Rust struct definitions."""
        code = """
pub struct User {
    pub id: String,
    pub name: String,
}

struct Config {
    port: u16,
    host: String,
}
"""
        (tmp_path / "types.rs").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        structs = store.find_by_type("struct")
        assert len(structs) >= 2
        names = [s.name for s in structs]
        assert "User" in names
        assert "Config" in names

    def test_extract_rust_enum(self, tmp_path):
        """Test extracting Rust enum definitions."""
        code = """
pub enum Status {
    Active,
    Inactive,
    Pending,
}
"""
        (tmp_path / "enums.rs").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        enums = store.find_by_type("enum")
        assert len(enums) >= 1
        assert any(e.name == "Status" for e in enums)

    def test_extract_rust_trait(self, tmp_path):
        """Test extracting Rust trait definitions."""
        code = """
pub trait Handler {
    fn handle(&self, request: Request) -> Response;
}
"""
        (tmp_path / "traits.rs").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        traits = store.find_by_type("trait")
        assert len(traits) >= 1
        assert any(t.name == "Handler" for t in traits)


class TestJavaExtraction:
    """Tests for Java symbol extraction using tree-sitter."""

    def test_extract_java_class(self, tmp_path):
        """Test extracting Java class definitions."""
        code = """
public class UserService {
    private Database db;

    public User getUser(String id) {
        return db.find(id);
    }
}
"""
        (tmp_path / "UserService.java").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        classes = store.find_by_type("class")
        assert len(classes) >= 1
        assert any(c.name == "UserService" for c in classes)

    def test_extract_java_interface(self, tmp_path):
        """Test extracting Java interface definitions."""
        code = """
public interface Repository {
    Object find(String id);
    void save(Object entity);
}
"""
        (tmp_path / "Repository.java").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        interfaces = store.find_by_type("interface")
        assert len(interfaces) >= 1
        assert any(i.name == "Repository" for i in interfaces)


class TestArchitecturePatternDetection:
    """Tests for architecture pattern detection."""

    def test_detect_provider_pattern(self, tmp_path):
        """Test detecting provider pattern in class names."""
        code = """
class DatabaseProvider:
    pass

class CacheProvider:
    pass

class AuthProvider:
    pass
"""
        (tmp_path / "providers.py").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        providers = store.find_by_category("provider")
        assert len(providers) == 3

    def test_detect_service_pattern(self, tmp_path):
        """Test detecting service pattern in class names."""
        code = """
class UserService:
    pass

class PaymentService:
    pass
"""
        (tmp_path / "services.py").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        services = store.find_by_category("service")
        assert len(services) == 2

    def test_detect_repository_pattern(self, tmp_path):
        """Test detecting repository pattern in class names."""
        code = """
class UserRepository:
    pass

class ProductRepository:
    pass
"""
        (tmp_path / "repositories.py").write_text(code)
        store = SymbolStore(str(tmp_path))

        asyncio.run(store.index_codebase())

        repos = store.find_by_category("repository")
        assert len(repos) == 2


class TestRobustIndexing:
    """Tests for robust codebase indexing with imperfect files."""

    def test_skip_binary_files(self, tmp_path):
        """Test that binary files are skipped gracefully."""
        # Create a fake binary file with .py extension
        binary_content = bytes([0x89, 0x50, 0x4E, 0x47, 0x00, 0x00])
        (tmp_path / "binary.py").write_bytes(binary_content)

        # Create a valid Python file
        (tmp_path / "valid.py").write_text("class Valid: pass")

        store = SymbolStore(str(tmp_path))
        stats = asyncio.run(store.index_codebase())

        # Valid file should be indexed, binary should be skipped
        assert stats["files_indexed"] >= 1
        classes = store.find_by_type("class")
        assert any(c.name == "Valid" for c in classes)

    def test_handle_encoding_errors(self, tmp_path):
        """Test handling files with encoding issues."""
        # Write a file with invalid UTF-8 (using errors='ignore' should handle it)
        (tmp_path / "encoding.py").write_bytes(b"class Test:\n    \xff\xfe pass")

        store = SymbolStore(str(tmp_path))
        asyncio.run(store.index_codebase())

        # Should handle encoding error gracefully
        # File may or may not be indexed depending on encoding handling

    def test_incremental_indexing(self, tmp_path):
        """Test incremental re-indexing only changed files."""
        (tmp_path / "file1.py").write_text("class One: pass")

        store = SymbolStore(str(tmp_path))
        stats1 = asyncio.run(store.index_codebase())
        assert stats1["files_indexed"] == 1

        # Re-index without changes
        stats2 = asyncio.run(store.index_codebase())
        assert stats2["files_skipped"] == 1
        assert stats2["files_indexed"] == 0

    def test_handle_deleted_files(self, tmp_path):
        """Test handling of deleted files during re-indexing."""
        file_path = tmp_path / "deleteme.py"
        file_path.write_text("class ToDelete: pass")

        store = SymbolStore(str(tmp_path))
        asyncio.run(store.index_codebase())

        # Verify class was indexed
        classes = store.find_by_type("class")
        assert any(c.name == "ToDelete" for c in classes)

        # Delete file and re-index
        file_path.unlink()
        stats = asyncio.run(store.index_codebase())

        assert stats["files_deleted"] == 1

        # Class should no longer be found
        classes = store.find_by_type("class")
        assert not any(c.name == "ToDelete" for c in classes)


class TestQueryMethods:
    """Tests for symbol query methods."""

    def test_find_by_name_pattern(self, tmp_path):
        """Test finding symbols by name pattern."""
        code = """
class UserService: pass
class UserRepository: pass
class ProductService: pass
"""
        (tmp_path / "classes.py").write_text(code)
        store = SymbolStore(str(tmp_path))
        asyncio.run(store.index_codebase())

        # Find all User-related classes
        user_classes = store.find_by_name_pattern("User%")
        assert len(user_classes) == 2

        # Find all services
        services = store.find_by_name_pattern("%Service")
        assert len(services) == 2

    def test_find_key_components(self, tmp_path):
        """Test finding key architectural components."""
        code = """
class UserService: pass
class DatabaseProvider: pass
class UserRepository: pass
class Config: pass
"""
        (tmp_path / "app.py").write_text(code)
        store = SymbolStore(str(tmp_path))
        asyncio.run(store.index_codebase())

        key = store.find_key_components(limit=10)
        # Should find components with architectural categories
        assert len(key) >= 2

    def test_get_stats(self, tmp_path):
        """Test getting store statistics."""
        (tmp_path / "python.py").write_text("class PyClass: pass")
        (tmp_path / "typescript.ts").write_text("class TsClass {}")

        store = SymbolStore(str(tmp_path))
        asyncio.run(store.index_codebase())

        stats = store.get_stats()

        assert stats["total_files"] == 2
        assert stats["total_symbols"] >= 2
        assert "python" in stats["files_by_language"]
        assert "typescript" in stats["files_by_language"]


class TestCanonicalProjectDb:
    """Regressions for canonical project database path usage."""

    def test_compat_settings_exposes_project_db(self, tmp_path):
        paths = get_project_paths(str(tmp_path))

        assert paths.project_db == tmp_path / ".victor" / "project.db"
        assert not hasattr(paths, "conversation_db")

    def test_symbol_store_uses_canonical_project_db(self, tmp_path):
        store = SymbolStore(str(tmp_path))

        assert store._db_path == tmp_path / ".victor" / "project.db"

    def test_symbol_store_normalizes_symlinked_include_dirs(self, tmp_path):
        alias_path = tmp_path.parent / f"{tmp_path.name}-alias"
        alias_path.symlink_to(tmp_path, target_is_directory=True)
        (tmp_path / "app.py").write_text("class App: pass")

        store = SymbolStore(str(tmp_path), include_dirs=[str(alias_path)])
        stats = asyncio.run(store.index_codebase())

        assert stats["files_indexed"] == 1
        classes = store.find_by_type("class")
        assert any(symbol.name == "App" for symbol in classes)
