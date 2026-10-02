#!/usr/bin/env python3
"""Synchronize package version strings from VERSION files.

victor-ai and victor-contracts have independent version files:
  - VERSION          → victor-ai version
  - victor-contracts/VERSION → victor-contracts version

Updates:
  - pyproject.toml (victor-ai) version
  - victor-contracts/pyproject.toml version
  - victor-ai's victor-contracts dependency lower bound
  - the victor_native artifact line (rust/pyproject.toml, the
    python-bindings crate manifest, Cargo.lock, and the root ``native``
    extra) — the native wheel version rides the victor-ai release so a
    release never re-uploads an already-published native version
    (0.11.0's publish 400'd exactly that way on a stale 0.8.2 wheel)

Usage:
  python scripts/sync_version.py          # Sync both packages
  python scripts/sync_version.py --ai     # Sync victor-ai only
  python scripts/sync_version.py --sdk    # Sync victor-contracts only
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sync_ai(root: Path = ROOT):
    """Sync victor-ai version (and the native artifact line) from VERSION."""
    version_file = root / "VERSION"
    if not version_file.exists():
        print("ERROR: VERSION file not found")
        sys.exit(1)

    version = version_file.read_text().strip()
    print(f"Syncing victor-ai to version {version}")

    ai_toml = root / "pyproject.toml"
    text = ai_toml.read_text()
    text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    ai_toml.write_text(text)
    print(f"  Updated {ai_toml}")

    sync_native(root, version)


def sync_native(root: Path, version: str) -> None:
    """Point every victor_native version reference at the release version.

    Synchronized by policy: the native wheel version tracks the victor-ai
    release. Four spots must agree (check_version_sync.py enforces this):
    rust/pyproject.toml, the python-bindings crate manifest, Cargo.lock's
    victor_native entry, and the root ``native`` extra's lower bound.
    """
    rust_pyproject = root / "rust" / "pyproject.toml"
    text = rust_pyproject.read_text()
    new = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert new != text or f'version = "{version}"' in text, f"no version line in {rust_pyproject}"
    rust_pyproject.write_text(new)

    crate = root / "rust" / "crates" / "python-bindings" / "Cargo.toml"
    text = crate.read_text()
    new = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert new != text or f'version = "{version}"' in text, f"no version line in {crate}"
    crate.write_text(new)

    lock = root / "rust" / "Cargo.lock"
    text = lock.read_text()
    new = re.sub(
        r'(name = "victor_native"\nversion = )"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
    )
    assert new != text, f"no victor_native lock entry in {lock}"
    lock.write_text(new)

    extra = root / "pyproject.toml"
    text = extra.read_text()
    new = re.sub(
        r'("victor-native>=)[^"]+(")',
        rf'\g<1>{version}\g<2>',
        text,
        count=1,
    )
    assert new != text, f"no victor-native extra bound in {extra}"
    extra.write_text(new)
    print(f"  Synced native artifact line to {version}")


def sync_sdk():
    """Sync victor-contracts version from its own VERSION file."""
    sdk_version_file = ROOT / "victor-contracts" / "VERSION"
    if not sdk_version_file.exists():
        print("ERROR: victor-contracts/VERSION file not found")
        sys.exit(1)

    version = sdk_version_file.read_text().strip()
    print(f"Syncing victor-contracts to version {version}")

    sdk_toml = ROOT / "victor-contracts" / "pyproject.toml"
    text = sdk_toml.read_text()
    text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    sdk_toml.write_text(text)
    print(f"  Updated {sdk_toml}")

    # Update victor-ai's SDK dependency lower bound
    ai_toml = ROOT / "pyproject.toml"
    text = ai_toml.read_text()
    text = re.sub(
        r'"victor-contracts[><=!~,. 0-9]+"',
        f'"victor-contracts>={version},<1.0"',
        text,
        count=1,
    )
    ai_toml.write_text(text)
    print(f"  Updated SDK dependency bound in {ai_toml}")


def main():
    if "--ai" in sys.argv:
        sync_ai()
    elif "--sdk" in sys.argv:
        sync_sdk()
    else:
        sync_ai()
        sync_sdk()

    print("\nSync complete. Run 'python scripts/check_version_sync.py' to verify.")


if __name__ == "__main__":
    main()
