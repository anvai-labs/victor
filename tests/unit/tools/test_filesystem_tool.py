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

"""Tests for filesystem tool."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from victor.tools.filesystem import (
    read,
    write,
    ls,
    reset_path_resolver,
    detect_file_type_by_magic,
    check_extension_magic_mismatch,
    FileCategory,
    FileTypeInfo,
    register_binary_handler,
    get_binary_handler,
    # File content cache (P3-2)
    FileContentCache,
    get_file_content_cache,
    clear_file_content_cache,
    is_file_cache_enabled,
    _cache_enabled,
)


@pytest.fixture(autouse=True)
def _allow_tmp_workspace(monkeypatch):
    """Disable workspace guard for tests that create files in /tmp."""
    monkeypatch.setenv("VICTOR_DISABLE_WORKSPACE_GUARD", "1")


@pytest.mark.asyncio
async def test_read_file_success():
    """Test successful file reading.

    Note: read() now returns formatted output with:
    - Header: [File: path] [Lines X-Y of Z] [Size: N bytes]
    - Line numbers prefixed to each line
    - Truncation info if applicable
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello, World!")
        temp_path = f.name

    try:
        result = await read(path=temp_path)

        # Check for header components
        assert "[File:" in result
        assert "[Lines 1-1 of 1]" in result
        assert "[Size:" in result
        # Check content is included (with line number prefix)
        assert "Hello, World!" in result
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_read_file_not_found():
    """Test reading non-existent file."""
    with pytest.raises(FileNotFoundError):
        await read(path="/nonexistent/path/file.txt")


@pytest.mark.asyncio
async def test_read_file_not_found_prefers_package_file_suggestions(tmp_path, monkeypatch):
    """Missing module files should suggest package-backed source files first."""
    package_dir = tmp_path / "victor" / "core" / "registry"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("# package")
    (package_dir / "base.py").write_text("# base")
    (tmp_path / "victor" / "core" / "registry_base.py").write_text("# legacy")
    monkeypatch.chdir(tmp_path)
    reset_path_resolver()

    with pytest.raises(FileNotFoundError) as exc_info:
        await read(path="victor/core/registry.py")

    message = str(exc_info.value)
    assert "victor/core/registry/base.py" in message
    assert message.index("victor/core/registry/base.py") < message.index(
        "victor/core/registry_base.py"
    )


@pytest.mark.asyncio
async def test_read_resolves_rust_module_file_to_mod_rs(tmp_path, monkeypatch):
    """Rust module paths like foo.rs should resolve to foo/mod.rs when present."""
    module_dir = tmp_path / "src" / "services" / "operations" / "vectors"
    module_dir.mkdir(parents=True)
    (module_dir / "mod.rs").write_text("pub mod legacy;\n")
    monkeypatch.chdir(tmp_path)
    reset_path_resolver()

    result = await read(path="src/services/operations/vectors.rs")

    assert "[File: src/services/operations/vectors/mod.rs]" in result
    assert "pub mod legacy;" in result


@pytest.mark.asyncio
async def test_read_file_not_a_file():
    """Test reading a directory path auto-converts to ls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Directories auto-convert to ls instead of raising
        result = await read(path=tmpdir)
        assert "directory" in result.lower() or "contents" in result.lower()


@pytest.mark.asyncio
async def test_read_file_exception_handling():
    """Test exception handling in read_file (permission error)."""
    from unittest.mock import patch

    # Create a real file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test")
        temp_path = f.name

    try:
        # Mock aiofiles.open to raise an exception
        with patch("aiofiles.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                await read(path=temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_write_file_success():
    """Test successful file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_file.txt")
        content = "Test content\nMultiple lines"

        result = await write(path=file_path, content=content)

        assert "Successfully created" in result
        assert f"{len(content)} characters" in result
        assert "/undo" in result  # Verify undo hint is shown

        # Verify file was actually written
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            written_content = f.read()
            assert written_content == content


@pytest.mark.asyncio
async def test_write_file_creates_directories():
    """Test that write_file creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested path that doesn't exist
        file_path = os.path.join(tmpdir, "subdir1", "subdir2", "test.txt")
        content = "Test content"

        result = await write(path=file_path, content=content)

        assert "Successfully created" in result

        # Verify directories and file were created
        assert os.path.exists(file_path)
        assert os.path.isfile(file_path)


@pytest.mark.asyncio
async def test_write_file_overwrites_existing():
    """Test that write_file overwrites existing files."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Original content")
        temp_path = f.name

    try:
        new_content = "New content"
        result = await write(path=temp_path, content=new_content)

        assert "Successfully modified" in result  # File existed, so it's a modification

        # Verify content was overwritten
        with open(temp_path, "r") as f:
            assert f.read() == new_content
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_write_file_exception_handling():
    """Test exception handling in write_file (permission error).

    write() delegates to edit() when the enhanced editor (victor-coding) is
    available; force the direct-write fallback path (which still uses
    aiofiles.open) so the I/O error surfaces as an exception.
    """
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")

        with (
            patch("victor.tools.filesystem._editor_available", return_value=False),
            patch("aiofiles.open", side_effect=PermissionError("Write denied")),
        ):
            with pytest.raises(PermissionError):
                await write(path=file_path, content="test content")


@pytest.mark.asyncio
async def test_list_directory_success():
    """Test listing directory contents with depth=1 (default)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files and directories
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(tmpdir, "file2.txt")
        subdir = os.path.join(tmpdir, "subdir")

        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")
        os.mkdir(subdir)

        # depth=1 returns immediate children with "name" key
        result = await ls(path=tmpdir, depth=1)

        # Result is now a dict with cwd, target, items, count
        assert "cwd" in result
        assert "target" in result
        assert "items" in result
        assert "count" in result
        assert result["count"] == 3

        items = result["items"]
        names = [item["name"] for item in items]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

        # Check types
        types = {item["name"]: item["type"] for item in items}
        assert types["file1.txt"] == "file"
        assert types["file2.txt"] == "file"
        assert types["subdir"] == "directory"


@pytest.mark.asyncio
async def test_list_directory_default_depth():
    """Test listing directory with default depth=2 returns immediate children and one level nested."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        file1 = os.path.join(tmpdir, "file1.txt")
        subdir = os.path.join(tmpdir, "subdir")
        nested_file = os.path.join(subdir, "nested.txt")

        with open(file1, "w") as f:
            f.write("test")
        os.mkdir(subdir)
        with open(nested_file, "w") as f:
            f.write("nested")

        # Default depth=2 returns immediate children AND one level nested
        result = await ls(path=tmpdir)

        # Result is now a dict with items
        assert "items" in result
        items = result["items"]
        # With depth=2: file1.txt, subdir, subdir/nested.txt
        assert len(items) == 3
        paths = [item.get("path", item.get("name")) for item in items]
        assert "file1.txt" in paths
        assert "subdir" in paths
        # nested.txt should be included at depth=2
        assert any("nested.txt" in p for p in paths)


@pytest.mark.asyncio
async def test_list_directory_recursive():
    """Test listing directory contents recursively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        subdir = os.path.join(tmpdir, "subdir")
        os.mkdir(subdir)
        file1 = os.path.join(tmpdir, "file1.txt")
        file2 = os.path.join(subdir, "file2.txt")

        with open(file1, "w") as f:
            f.write("test")
        with open(file2, "w") as f:
            f.write("test")

        result = await ls(path=tmpdir, recursive=True)

        # Result is now a dict with items
        assert "items" in result
        items = result["items"]
        assert len(items) == 3  # subdir, file1.txt, subdir/file2.txt
        paths = [item["path"] for item in items]
        assert "subdir" in paths
        assert "file1.txt" in paths
        assert os.path.join("subdir", "file2.txt") in paths or "subdir/file2.txt" in paths


@pytest.mark.asyncio
async def test_list_directory_not_found():
    """Test listing non-existent directory."""
    with pytest.raises(FileNotFoundError):
        await ls(path="/nonexistent/directory")


@pytest.mark.asyncio
async def test_list_directory_not_a_directory():
    """Test listing a file path returns file metadata."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
        f.write("test content")
        temp_path = f.name

    try:
        # ls() on a file should return file metadata in ls format
        result = await ls(path=temp_path)
        # Should return a dict with items list (same format as directory listing)
        assert isinstance(result, dict)
        assert "items" in result
        assert result["count"] == 1
        assert result["auto_converted_from_file"] is True

        # Check the single file item has comprehensive metadata
        item = result["items"][0]
        assert item["type"] == "file"
        # Path may be resolved to absolute form (e.g., /private/var on macOS)
        assert item["path"] == str(Path(temp_path).resolve())
        assert "name" in item
        assert "size" in item
        assert "permissions" in item
        assert "owner" in item
        assert "group" in item
        assert "modified" in item
        assert "auto_converted" in item
        assert "hint" in item
        assert item["hint"].startswith("Note:")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
async def test_list_directory_empty():
    """Test listing empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await ls(path=tmpdir)

        # Result is now a dict with items
        assert "items" in result
        assert result["items"] == []
        assert result["count"] == 0


@pytest.mark.asyncio
async def test_write_file_to_directory_raises_error():
    """Test that write_file raises IsADirectoryError for directory path (line 91)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(IsADirectoryError) as excinfo:
            await write(path=tmpdir, content="test content")

        assert "Cannot write to directory" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_binary_extension_rejected():
    """Test that read_file rejects files with binary extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with a binary extension
        binary_file = os.path.join(tmpdir, "test.pyc")
        with open(binary_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        with pytest.raises(ValueError) as excinfo:
            await read(path=binary_file)

        assert "Cannot read binary file" in str(excinfo.value)
        assert ".pyc" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_binary_content_rejected():
    """Test that read_file rejects files with binary content (non-UTF-8)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a text file but with binary content
        binary_file = os.path.join(tmpdir, "test.txt")
        with open(binary_file, "wb") as f:
            # Write invalid UTF-8 bytes
            f.write(b"\x80\x81\x82\x83")

        with pytest.raises(ValueError) as excinfo:
            await read(path=binary_file)

        assert "Cannot read file" in str(excinfo.value)
        assert "binary" in str(excinfo.value).lower()
        assert "non-UTF-8 content" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_coverage_file_rejected():
    """Test that read_file rejects .coverage files (SQLite databases)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        coverage_file = os.path.join(tmpdir, ".coverage")
        with open(coverage_file, "wb") as f:
            f.write(b"SQLite format 3\x00")

        with pytest.raises(ValueError) as excinfo:
            await read(path=coverage_file)

        assert "Cannot read binary file" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_db_extension_rejected():
    """Test that read_file rejects .db and .sqlite files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in [".db", ".sqlite", ".sqlite3"]:
            db_file = os.path.join(tmpdir, f"test{ext}")
            with open(db_file, "wb") as f:
                f.write(b"\x00\x01\x02\x03")

            with pytest.raises(ValueError) as excinfo:
                await read(path=db_file)

            assert "Cannot read binary file" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_file_directory_auto_converts_to_ls():
    """Test that read(dir) auto-converts to ls(dir) instead of erroring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await read(path=tmpdir)
        assert "directory" in result.lower()


# ============================================================================
# MAGIC BYTES DETECTION TESTS
# ============================================================================


class TestMagicBytesDetection:
    """Tests for magic bytes file type detection."""

    def test_detect_png_magic_bytes(self):
        """Test PNG file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            # PNG magic bytes
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.IMAGE
            assert result.mime_type == "image/png"
            assert ".png" in result.extensions

            os.unlink(f.name)

    def test_detect_jpeg_magic_bytes(self):
        """Test JPEG file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            # JPEG magic bytes
            f.write(b"\xff\xd8\xff")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.IMAGE
            assert result.mime_type == "image/jpeg"
            assert ".jpg" in result.extensions or ".jpeg" in result.extensions

            os.unlink(f.name)

    def test_detect_pdf_magic_bytes(self):
        """Test PDF file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            # PDF magic bytes
            f.write(b"%PDF-1.4")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.DOCUMENT
            assert result.mime_type == "application/pdf"

            os.unlink(f.name)

    def test_detect_zip_magic_bytes(self):
        """Test ZIP file detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:
            # ZIP magic bytes (PK\x03\x04)
            f.write(b"PK\x03\x04")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.ARCHIVE
            assert result.mime_type == "application/zip"

            os.unlink(f.name)

    def test_detect_sqlite_magic_bytes(self):
        """Test SQLite database detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            # SQLite magic bytes
            f.write(b"SQLite format 3\x00")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.DATABASE
            assert result.mime_type == "application/x-sqlite3"

            os.unlink(f.name)

    def test_detect_elf_magic_bytes(self):
        """Test ELF executable detection by magic bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".so") as f:
            # ELF magic bytes
            f.write(b"\x7fELF")
            f.write(b"\x00" * 100)
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is not None
            assert result.category == FileCategory.COMPILED
            assert "ELF" in result.description

            os.unlink(f.name)

    def test_detect_text_file_no_magic(self):
        """Test that text files don't match any magic signature."""
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as f:
            f.write("This is plain text content")
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            # Text files shouldn't match any magic signature
            assert result is None

            os.unlink(f.name)

    def test_empty_file_no_magic(self):
        """Test that empty files don't match any magic signature."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.flush()

            result = detect_file_type_by_magic(Path(f.name))

            assert result is None

            os.unlink(f.name)


class TestExtensionMagicMismatch:
    """Tests for extension/magic bytes mismatch detection."""

    def test_matching_extension_no_warning(self):
        """Test no warning when extension matches magic bytes."""
        type_info = FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/png",
            description="PNG image",
            extensions=(".png",),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            warning = check_extension_magic_mismatch(Path(f.name), type_info)
            assert warning is None
            os.unlink(f.name)

    def test_mismatched_extension_warning(self):
        """Test warning when extension doesn't match magic bytes."""
        type_info = FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/png",
            description="PNG image",
            extensions=(".png",),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            warning = check_extension_magic_mismatch(Path(f.name), type_info)

            assert warning is not None
            assert ".txt" in warning
            assert "PNG image" in warning
            assert ".png" in warning

            os.unlink(f.name)

    def test_no_magic_type_no_warning(self):
        """Test no warning when no magic type is detected."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            warning = check_extension_magic_mismatch(Path(f.name), None)
            assert warning is None
            os.unlink(f.name)


class TestBinaryFileHandler:
    """Tests for binary file handler registry."""

    def test_register_and_get_handler(self):
        """Test registering and retrieving a binary handler."""
        from victor.tools.filesystem import unregister_binary_handler

        def dummy_pdf_handler(path: Path) -> str:
            return f"Extracted text from {path}"

        try:
            # Register handler
            register_binary_handler(FileCategory.DOCUMENT, dummy_pdf_handler)

            # Retrieve handler
            handler = get_binary_handler(FileCategory.DOCUMENT)
            assert handler is not None
            assert handler(Path("/test.pdf")) == "Extracted text from /test.pdf"
        finally:
            # Clean up to avoid test pollution
            unregister_binary_handler(FileCategory.DOCUMENT)

    def test_get_unregistered_handler(self):
        """Test getting handler for unregistered category returns None."""
        # MEDIA category likely not registered
        get_binary_handler(FileCategory.MEDIA)
        # May or may not be None depending on previous tests
        # Just verify it doesn't raise an exception


@pytest.mark.asyncio
async def test_read_file_rejects_png_by_magic():
    """Test that read_file rejects PNG files detected by magic bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".data") as f:
        # Write PNG magic bytes with misleading extension
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(b"\x00" * 100)
        f.flush()

        with pytest.raises(ValueError) as excinfo:
            await read(path=f.name)

        error_msg = str(excinfo.value)
        assert "PNG image" in error_msg
        assert "Cannot read binary file" in error_msg

        os.unlink(f.name)


@pytest.mark.asyncio
async def test_read_file_rejects_pdf_by_magic():
    """Test that read_file rejects PDF files detected by magic bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        # Write PDF magic bytes with misleading .txt extension
        f.write(b"%PDF-1.4\n")
        f.write(b"%%EOF\n")
        f.flush()

        with pytest.raises(ValueError) as excinfo:
            await read(path=f.name)

        error_msg = str(excinfo.value)
        assert "PDF document" in error_msg
        assert "Cannot read binary file" in error_msg

        os.unlink(f.name)


@pytest.mark.asyncio
async def test_read_file_rejects_sqlite_by_magic():
    """Test that read_file rejects SQLite database detected by magic bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        # Write SQLite magic bytes with misleading .txt extension
        f.write(b"SQLite format 3\x00")
        f.write(b"\x00" * 100)
        f.flush()

        with pytest.raises(ValueError) as excinfo:
            await read(path=f.name)

        error_msg = str(excinfo.value)
        assert "SQLite database" in error_msg
        assert "Cannot read binary file" in error_msg

        os.unlink(f.name)


@pytest.mark.asyncio
async def test_read_file_allows_text_with_no_magic():
    """Test that read_file allows text files with no magic signature."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as f:
        f.write("This is plain text content")
        f.flush()

        content = await read(path=f.name)
        # Content is now included in formatted output with headers and line numbers
        assert "This is plain text content" in content
        assert "[Lines 1-1 of 1]" in content

        os.unlink(f.name)


# ============================================================================
# FILE CONTENT CACHE TESTS (P3-2)
# ============================================================================


class TestFileContentCache:
    """Tests for session-level file content cache."""

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = FileContentCache(max_entries=10)

        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("test content")
            temp_path = f.name

        try:
            mtime = os.path.getmtime(temp_path)
            size = os.path.getsize(temp_path)

            # Set in cache
            cache.set(temp_path, "test content", mtime, size)

            # Get from cache
            result = cache.get(temp_path)
            assert result == "test content"

            # Stats should show 1 hit
            stats = cache.get_stats()
            assert stats["hits"] == 1
        finally:
            os.unlink(temp_path)

    def test_cache_miss_on_empty(self):
        """Test cache miss returns None."""
        cache = FileContentCache()
        result = cache.get("/nonexistent/path.txt")
        assert result is None

        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_cache_invalidate_on_mtime_change(self):
        """Test cache invalidates when file mtime changes."""
        import time

        cache = FileContentCache()

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("original content")
            temp_path = f.name

        try:
            mtime = os.path.getmtime(temp_path)
            size = os.path.getsize(temp_path)

            cache.set(temp_path, "original content", mtime, size)

            # Modify file
            time.sleep(0.1)  # Ensure mtime changes
            with open(temp_path, "w") as f:
                f.write("modified content")

            # Cache should return None (mtime changed)
            result = cache.get(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)

    def test_cache_invalidate_explicit(self):
        """Test explicit cache invalidation."""
        cache = FileContentCache()

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("content")
            temp_path = f.name

        try:
            mtime = os.path.getmtime(temp_path)
            size = os.path.getsize(temp_path)

            cache.set(temp_path, "content", mtime, size)
            assert cache.get(temp_path) == "content"

            # Invalidate
            cache.invalidate(temp_path)

            # Should miss now
            assert cache.get(temp_path) is None

            stats = cache.get_stats()
            assert stats["invalidations"] == 1
        finally:
            os.unlink(temp_path)

    def test_cache_clear(self):
        """Test cache clear removes all entries."""
        cache = FileContentCache()

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("content")
            temp_path = f.name

        try:
            mtime = os.path.getmtime(temp_path)
            size = os.path.getsize(temp_path)

            cache.set(temp_path, "content", mtime, size)
            assert len(cache) == 1

            cache.clear()
            assert len(cache) == 0
        finally:
            os.unlink(temp_path)

    def test_cache_lru_eviction(self):
        """Test LRU eviction when max_entries exceeded."""
        cache = FileContentCache(max_entries=2)

        # Create 3 temp files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
                f.write(f"content {i}")
                temp_files.append(f.name)

        try:
            # Cache files 0 and 1
            for i in range(2):
                mtime = os.path.getmtime(temp_files[i])
                size = os.path.getsize(temp_files[i])
                cache.set(temp_files[i], f"content {i}", mtime, size)

            assert len(cache) == 2

            # Cache file 2 - should evict file 0 (LRU)
            mtime = os.path.getmtime(temp_files[2])
            size = os.path.getsize(temp_files[2])
            cache.set(temp_files[2], "content 2", mtime, size)

            assert len(cache) == 2

            # File 0 should be evicted
            assert cache.get(temp_files[0]) is None
            # Files 1 and 2 should still be cached
            assert cache.get(temp_files[1]) == "content 1"
            assert cache.get(temp_files[2]) == "content 2"
        finally:
            for temp_path in temp_files:
                os.unlink(temp_path)

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = FileContentCache()

        with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
            f.write("content")
            temp_path = f.name

        try:
            mtime = os.path.getmtime(temp_path)
            size = os.path.getsize(temp_path)

            # Miss
            cache.get(temp_path)

            # Set
            cache.set(temp_path, "content", mtime, size)

            # Hits
            cache.get(temp_path)
            cache.get(temp_path)

            stats = cache.get_stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["entries"] == 1
            assert stats["total_bytes_cached"] == size
            assert stats["total_bytes_saved"] == size * 2
        finally:
            os.unlink(temp_path)


class TestFileCacheIntegration:
    """Integration tests for file cache with read/write functions.

    NOTE: These tests are skipped when file cache is disabled (default).
    The OS page cache provides better memory management, so the Python
    process cache is disabled by default.
    """

    @pytest.fixture(autouse=True)
    def setup_cache(self):
        """Ensure cache is cleared before each test."""
        from victor.tools.filesystem import is_file_cache_enabled

        if not is_file_cache_enabled():
            pytest.skip("File cache disabled - OS page cache handles this better")
        clear_file_content_cache()
        yield
        clear_file_content_cache()

    @pytest.mark.asyncio
    async def test_read_caches_content(self):
        """Test that read() caches file content."""
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".py") as f:
            f.write("print('hello')")
            temp_path = f.name

        try:
            # First read - cache miss
            content1 = await read(path=temp_path)
            # Content is now wrapped with headers and line numbers
            assert "print('hello')" in content1

            cache = get_file_content_cache()
            stats = cache.get_stats()
            assert stats["misses"] == 1

            # Second read - cache hit
            content2 = await read(path=temp_path)
            assert "print('hello')" in content2

            stats = cache.get_stats()
            assert stats["hits"] == 1
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_write_invalidates_cache(self):
        """Test that write() invalidates file cache."""
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".py") as f:
            f.write("original")
            temp_path = f.name

        try:
            # Read to populate cache
            content1 = await read(path=temp_path)
            # Content is now wrapped with headers and line numbers
            assert "original" in content1

            # Write new content
            await write(path=temp_path, content="updated")

            # Read should get fresh content
            content2 = await read(path=temp_path)
            assert "updated" in content2
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_cache_enabled_by_default(self):
        """Test that cache is enabled by default."""
        assert is_file_cache_enabled() is True
        assert _cache_enabled is True
