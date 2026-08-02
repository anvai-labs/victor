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

"""Demo of multi-file editing with diff preview and rollback.

This shows how Victor safely edits multiple files with:
- Transaction-based editing
- Rich diff preview
- Automatic backups
- Rollback capability
- Dry-run mode

Usage:
    python examples/multi_file_editing_demo.py
"""

import asyncio
import tempfile
from pathlib import Path
from victor.processing.editing import FileEditor


def main():
    """Demo multi-file editing."""
    print("🎯 Multi-File Editing Demo\n")
    print("=" * 70)

    # Create a temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Demo 1: Create new files
        print("\n\n1️⃣ Creating new files...")
        print("-" * 70)

        editor = FileEditor(backup_dir=str(tmpdir / "backups"))
        editor.start_transaction("Create authentication module")

        # Add file creations
        editor.add_create(
            path=str(tmpdir / "auth.py"),
            content="""\"\"\"Authentication module.\"\"\"

def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate user credentials.

    Args:
        username: Username
        password: Password

    Returns:
        True if authenticated, False otherwise
    \"\"\"
    # TODO: Implement actual authentication
    return username == "admin" and password == "secret"


def create_session(user_id: int) -> str:
    \"\"\"Create user session.

    Args:
        user_id: User ID

    Returns:
        Session token
    \"\"\"
    import secrets
    return secrets.token_hex(32)
""",
        )

        editor.add_create(
            path=str(tmpdir / "auth_test.py"),
            content="""\"\"\"Tests for authentication module.\"\"\"

import pytest
from auth import authenticate, create_session


def test_authenticate_valid():
    \"\"\"Test authentication with valid credentials.\"\"\"
    assert authenticate("admin", "secret") is True


def test_authenticate_invalid():
    \"\"\"Test authentication with invalid credentials.\"\"\"
    assert authenticate("user", "wrong") is False


def test_create_session():
    \"\"\"Test session creation.\"\"\"
    session = create_session(123)
    assert len(session) == 64  # 32 bytes hex = 64 chars
""",
        )

        # Preview changes
        editor.preview_diff()

        # Commit
        success = editor.commit()

        if success:
            print("\n✅ Files created successfully!")
            print(f"   - {tmpdir / 'auth.py'}")
            print(f"   - {tmpdir / 'auth_test.py'}")

        # Demo 2: Modify existing files
        print("\n\n2️⃣ Modifying existing files...")
        print("-" * 70)

        editor = FileEditor(backup_dir=str(tmpdir / "backups"))
        editor.start_transaction("Add password hashing to auth module")

        # Read current content (verify file exists)
        _ = (tmpdir / "auth.py").read_text()

        # Modify it
        new_auth_content = """\"\"\"Authentication module with password hashing.\"\"\"

import hashlib
import secrets


def hash_password(password: str, salt: bytes = None) -> tuple[str, bytes]:
    \"\"\"Hash password with salt.

    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hashed_password, salt)
    \"\"\"
    if salt is None:
        salt = secrets.token_bytes(32)

    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return hashed.hex(), salt


def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate user credentials with password hashing.

    Args:
        username: Username
        password: Password

    Returns:
        True if authenticated, False otherwise
    \"\"\"
    # TODO: Load stored hash and salt from database
    # This is a simplified version
    stored_hash = "abc123"  # In production: load from DB
    stored_salt = b"salt"   # In production: load from DB

    hashed, _ = hash_password(password, stored_salt)
    return username == "admin" and hashed == stored_hash


def create_session(user_id: int) -> str:
    \"\"\"Create user session.

    Args:
        user_id: User ID

    Returns:
        Session token
    \"\"\"
    return secrets.token_hex(32)
"""

        editor.add_modify(path=str(tmpdir / "auth.py"), new_content=new_auth_content)

        # Preview diff
        editor.preview_diff()

        # Commit
        success = editor.commit()

        if success:
            print("\n✅ File modified successfully!")

        # Demo 3: Dry run mode
        print("\n\n3️⃣ Dry run mode (preview without applying)...")
        print("-" * 70)

        editor = FileEditor(backup_dir=str(tmpdir / "backups"))
        editor.start_transaction("Add logging to auth module")

        logged_content = new_auth_content.replace(
            "import secrets",
            "import secrets\nimport logging\n\nlogger = logging.getLogger(__name__)",
        )

        editor.add_modify(path=str(tmpdir / "auth.py"), new_content=logged_content)

        editor.preview_diff()

        # Dry run - doesn't actually apply changes
        editor.commit(dry_run=True)

        # Verify file wasn't changed
        current_content = (tmpdir / "auth.py").read_text()
        unchanged = current_content == new_auth_content
        print(f"\n✅ Dry run complete! File unchanged: {unchanged}")

        # Demo 4: Rollback on error
        print("\n\n4️⃣ Rollback capability (simulated error)...")
        print("-" * 70)

        editor = FileEditor(backup_dir=str(tmpdir / "backups"))
        editor.start_transaction("Attempt problematic changes")

        editor.add_modify(
            path=str(tmpdir / "auth.py"),
            new_content="# This change will be rolled back\n",
        )

        # Simulate manual rollback (in practice, this happens automatically on errors)
        editor.preview_diff()
        print("\n[Simulating error during commit...]")
        editor.rollback()

        # Verify file was restored
        current_content = (tmpdir / "auth.py").read_text()
        restored = current_content == new_auth_content
        print(f"✅ Rollback complete! File restored: {restored}")

        # Demo 5: Multiple operations in one transaction
        print("\n\n5️⃣ Multiple operations in one transaction...")
        print("-" * 70)

        editor = FileEditor(backup_dir=str(tmpdir / "backups"))
        editor.start_transaction("Reorganize auth module")

        # Create new utils file
        editor.add_create(
            path=str(tmpdir / "auth_utils.py"),
            content="""\"\"\"Authentication utilities.\"\"\"

import hashlib
import secrets


def hash_password(password: str, salt: bytes = None) -> tuple[str, bytes]:
    \"\"\"Hash password with salt.\"\"\"
    if salt is None:
        salt = secrets.token_bytes(32)

    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return hashed.hex(), salt
""",
        )

        # Update main auth file to import from utils
        refactored_auth = """\"\"\"Authentication module.\"\"\"

import secrets
from auth_utils import hash_password


def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate user credentials.\"\"\"
    stored_hash = "abc123"
    stored_salt = b"salt"

    hashed, _ = hash_password(password, stored_salt)
    return username == "admin" and hashed == stored_hash


def create_session(user_id: int) -> str:
    \"\"\"Create user session.\"\"\"
    return secrets.token_hex(32)
"""

        editor.add_modify(path=str(tmpdir / "auth.py"), new_content=refactored_auth)

        # Update tests
        updated_tests = """\"\"\"Tests for authentication module.\"\"\"

import pytest
from auth import authenticate, create_session
from auth_utils import hash_password


def test_authenticate_valid():
    \"\"\"Test authentication with valid credentials.\"\"\"
    assert authenticate("admin", "secret") is True


def test_authenticate_invalid():
    \"\"\"Test authentication with invalid credentials.\"\"\"
    assert authenticate("user", "wrong") is False


def test_create_session():
    \"\"\"Test session creation.\"\"\"
    session = create_session(123)
    assert len(session) == 64


def test_hash_password():
    \"\"\"Test password hashing.\"\"\"
    hashed, salt = hash_password("test123")
    assert len(hashed) == 64  # 32 bytes hex
    assert len(salt) == 32
"""

        editor.add_modify(path=str(tmpdir / "auth_test.py"), new_content=updated_tests)

        # Preview all changes
        editor.preview_diff()

        # Show transaction summary
        summary = editor.get_transaction_summary()
        print("\n📊 Transaction Summary:")
        print(f"   ID: {summary['id']}")
        print(f"   Total operations: {summary['operations']}")
        print("   By type:")
        for op_type, count in summary["by_type"].items():
            if count > 0:
                print(f"     - {op_type}: {count}")

        # Commit all changes atomically
        success = editor.commit()

        if success:
            print("\n✅ All changes committed atomically!")

        # Demo 6: Rename operation
        print("\n\n6️⃣ Rename operation...")
        print("-" * 70)

        editor = FileEditor(backup_dir=str(tmpdir / "backups"))
        editor.start_transaction("Rename utils to helpers")

        editor.add_rename(
            old_path=str(tmpdir / "auth_utils.py"),
            new_path=str(tmpdir / "auth_helpers.py"),
        )

        editor.preview_diff()
        success = editor.commit()

        if success:
            print("\n✅ File renamed successfully!")

        # Demo 7: Delete operation
        print("\n\n7️⃣ Delete operation...")
        print("-" * 70)

        editor = FileEditor(backup_dir=str(tmpdir / "backups"))
        editor.start_transaction("Remove test file")

        editor.add_delete(str(tmpdir / "auth_test.py"))

        editor.preview_diff()
        success = editor.commit()

        if success:
            print("\n✅ File deleted successfully!")

        # Show final state
        print("\n\n📁 Final file state:")
        print("-" * 70)
        remaining_files = list(tmpdir.glob("*.py"))
        for file in remaining_files:
            print(f"   - {file.name}")

        print("\n\n💾 Backup files created:")
        print("-" * 70)
        backup_dir = tmpdir / "backups"
        if backup_dir.exists():
            backups = list(backup_dir.glob("*.backup"))
            for backup in backups:
                print(f"   - {backup.name}")

    print("\n\n✨ Demo Complete!")
    print("\nMulti-file editing provides:")
    print("  ✓ Transaction-based safety (atomic operations)")
    print("  ✓ Rich diff preview with syntax highlighting")
    print("  ✓ Automatic backups before modifications")
    print("  ✓ Complete rollback capability")
    print("  ✓ Dry-run mode for testing")
    print("  ✓ Multiple operation types (create/modify/delete/rename)")
    print("  ✓ All-or-nothing commits")


if __name__ == "__main__":
    main()
