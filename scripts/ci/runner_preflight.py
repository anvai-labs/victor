#!/usr/bin/env python3
"""Fail fast when a CI host does not meet a job's declared ABI contract."""

from __future__ import annotations

import argparse
import platform
import shutil
from pathlib import Path
from typing import Mapping, Sequence


def parse_version(value: str) -> tuple[int, ...]:
    """Parse a numeric dotted version for deterministic comparisons."""
    parts = value.strip().split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"expected a numeric dotted version, got {value!r}")
    return tuple(int(part) for part in parts)


def version_at_least(actual: str, minimum: str) -> bool:
    """Return whether ``actual`` is greater than or equal to ``minimum``."""
    actual_parts = parse_version(actual)
    minimum_parts = parse_version(minimum)
    width = max(len(actual_parts), len(minimum_parts))
    return actual_parts + (0,) * (width - len(actual_parts)) >= minimum_parts + (0,) * (
        width - len(minimum_parts)
    )


def read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    """Read freedesktop OS metadata without executing shell input."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def compatibility_errors(
    *,
    os_release: Mapping[str, str],
    libc_name: str,
    glibc_version: str,
    required_os: str | None,
    minimum_os_version: str | None,
    minimum_glibc: str | None,
    required_commands: Sequence[str],
) -> list[str]:
    """Return every unmet host requirement so operators get one diagnosis."""
    errors: list[str] = []
    actual_os = os_release.get("ID", "unknown").lower()
    actual_os_version = os_release.get("VERSION_ID", "unknown")

    if required_os and actual_os != required_os.lower():
        errors.append(f"OS {actual_os!r} does not match required {required_os!r}")
    if minimum_os_version:
        try:
            compatible_os = version_at_least(actual_os_version, minimum_os_version)
        except ValueError:
            compatible_os = False
        if not compatible_os:
            errors.append(f"OS version {actual_os_version!r} is below {minimum_os_version!r}")
    if minimum_glibc and libc_name.lower() != "glibc":
        errors.append(f"libc {libc_name!r} does not satisfy the glibc requirement")
    elif minimum_glibc:
        try:
            compatible_glibc = version_at_least(glibc_version, minimum_glibc)
        except ValueError:
            compatible_glibc = False
        if not compatible_glibc:
            errors.append(f"glibc {glibc_version!r} is below {minimum_glibc!r}")

    for command in required_commands:
        if shutil.which(command) is None:
            errors.append(f"required command {command!r} is unavailable")
    return errors


def build_parser() -> argparse.ArgumentParser:
    """Build the runner-contract command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-os")
    parser.add_argument("--min-os-version")
    parser.add_argument("--min-glibc")
    parser.add_argument("--require-command", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect this host and return a CI-friendly status code."""
    args = build_parser().parse_args(argv)
    os_release = read_os_release()
    libc_name, libc_version = platform.libc_ver()
    print(
        "runner preflight: "
        f"os={os_release.get('ID', 'unknown')} "
        f"version={os_release.get('VERSION_ID', 'unknown')} "
        f"libc={libc_name or 'unknown'}-{libc_version or 'unknown'}"
    )
    errors = compatibility_errors(
        os_release=os_release,
        libc_name=libc_name,
        glibc_version=libc_version,
        required_os=args.require_os,
        minimum_os_version=args.min_os_version,
        minimum_glibc=args.min_glibc,
        required_commands=args.require_command,
    )
    if errors:
        for error in errors:
            print(f"runner preflight failed: {error}")
        return 1
    print("runner preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
