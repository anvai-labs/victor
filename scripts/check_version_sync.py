#!/usr/bin/env python3
"""Verify that victor-ai and victor-contracts versions are consistent.

victor-ai and victor-contracts maintain independent version numbers.

Checks:
  1. VERSION file exists and matches victor-ai pyproject.toml
  2. victor-contracts/VERSION exists and matches victor-contracts pyproject.toml
  3. victor-ai's dependency on victor-contracts uses a compatible range
  4. Native manifests, lock and installation extra agree on the independent version

Exit code 0 on success, 1 on mismatch.
"""

import re
import sys
import tomllib
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_version(path: Path) -> str:
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    return path.read_text().strip()


def extract_toml_version(toml_path: Path) -> str:
    """Extract version = "X.Y.Z" from a pyproject.toml."""
    text = toml_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print(f"ERROR: No version found in {toml_path}")
        sys.exit(1)
    return match.group(1)


def extract_sdk_dependency(toml_path: Path) -> str:
    """Extract the victor-contracts dependency spec from victor-ai's pyproject.toml."""
    text = toml_path.read_text()
    match = re.search(r'"(victor-contracts[^"]*)"', text)
    if not match:
        print(f"ERROR: No victor-contracts dependency found in {toml_path}")
        sys.exit(1)
    return match.group(1)


def native_version_errors(root: Path) -> list[str]:
    """Reject partial native-version bumps before any artifacts are published."""

    def read(relative: str) -> dict[str, Any]:
        return tomllib.loads((root / relative).read_text())

    version = read("rust/pyproject.toml")["project"]["version"]
    cargo = read("rust/crates/python-bindings/Cargo.toml")["package"]["version"]
    locked = [
        package["version"]
        for package in read("rust/Cargo.lock")["package"]
        if package["name"] == "victor_native"
    ]
    extra = read("pyproject.toml")["project"]["optional-dependencies"]["native"]
    errors = []
    if cargo != version or locked != [version]:
        errors.append("native Python, Cargo manifest and lock versions must agree")
    if f"victor-native>={version}" not in extra:
        errors.append("the native extra must require the current native artifact version")
    return errors


def main() -> None:
    errors = native_version_errors(ROOT)

    # Check victor-ai version
    ai_version = read_version(ROOT / "VERSION")
    ai_toml_version = extract_toml_version(ROOT / "pyproject.toml")
    print(f"victor-ai VERSION file: {ai_version}")
    print(f"victor-ai pyproject.toml: {ai_toml_version}")
    if ai_version != ai_toml_version:
        errors.append(f"victor-ai version ({ai_toml_version}) != VERSION ({ai_version})")

    # Check victor-contracts version
    sdk_version = read_version(ROOT / "victor-contracts" / "VERSION")
    sdk_toml_version = extract_toml_version(ROOT / "victor-contracts" / "pyproject.toml")
    print(f"victor-contracts VERSION file: {sdk_version}")
    print(f"victor-contracts pyproject.toml: {sdk_toml_version}")
    if sdk_version != sdk_toml_version:
        errors.append(f"victor-contracts version ({sdk_toml_version}) != VERSION ({sdk_version})")

    # Check victor-ai's dependency on victor-contracts
    dep_spec = extract_sdk_dependency(ROOT / "pyproject.toml")
    print(f"victor-ai depends on: {dep_spec}")

    # Verify the SDK version satisfies the dependency spec
    if ">=" in dep_spec:
        # Extract lower bound: "victor-contracts>=X.Y.Z,<A.B"
        lower_match = re.search(r">=([0-9.]+)", dep_spec)
        if lower_match:
            lower_bound = lower_match.group(1)
            if sdk_version < lower_bound:
                errors.append(
                    f"victor-contracts version ({sdk_version}) is below "
                    f"victor-ai's lower bound ({lower_bound})"
                )
    elif "==" in dep_spec:
        # Exact pin — still valid, just not decoupled
        pin_match = re.search(r"==([0-9.]+)", dep_spec)
        if pin_match and pin_match.group(1) != sdk_version:
            errors.append(
                f"victor-ai pins victor-contracts=={pin_match.group(1)} "
                f"but SDK version is {sdk_version}"
            )

    if errors:
        print()
        for err in errors:
            print(f"FAIL: {err}")
        print("\nTo fix: update VERSION files and run 'make sync-version'")
        sys.exit(1)

    print("\nAll versions consistent.")
    sys.exit(0)


if __name__ == "__main__":
    main()
