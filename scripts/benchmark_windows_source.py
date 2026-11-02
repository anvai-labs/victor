"""Measure bounded Windows source inventory/read/hash, separately from parsing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import psutil

from victor.core.graph_rag.source_inventory import source_files
from victor.core.graph_rag.source_reader import project_source_reader


def measure(root: Path, expected_files: int, byte_limit: int) -> dict[str, float | int | str]:
    """Measure end-to-end inventory and acquisition with a digest correctness gate."""
    started, cpu_started = time.perf_counter(), time.process_time()
    digest = hashlib.sha256()
    total = 0
    peak_rss = 0
    process = psutil.Process()
    with project_source_reader(root) as reader:
        names = source_files(reader.descriptor, {".py"})
        if len(names) != expected_files:
            raise ValueError("Windows benchmark source coverage changed")
        for name in names:
            data, info = reader(name, byte_limit)
            if len(data) != byte_limit or info.st_size != byte_limit:
                raise ValueError("Windows benchmark source bytes changed")
            digest.update(name.encode())
            digest.update(data)
            total += len(data)
            peak_rss = max(peak_rss, process.memory_info().rss)
        reader.verify_root_path()
    elapsed = time.perf_counter() - started
    return {
        "files": len(names),
        "bytes": total,
        "wall_seconds": elapsed,
        "parent_cpu_seconds": time.process_time() - cpu_started,
        "sampled_parent_peak_rss_bytes": peak_rss,
        "bytes_per_second": total / elapsed,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=4096)
    parser.add_argument("--bytes-per-file", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if os.name != "nt":
        parser.error("Qualification benchmark requires actual Windows")
    if not 1 <= arguments.files <= 20_000 or not 64 <= arguments.bytes_per_file <= 1024 * 1024:
        parser.error("Benchmark corpus exceeds its bounds")
    with tempfile.TemporaryDirectory(prefix="victor-windows-source-benchmark-") as directory:
        root = Path(directory)
        data = b"def f(value):\n    return value + 1\n"
        data += b"#" * (arguments.bytes_per_file - len(data))
        for index in range(arguments.files):
            folder = root / f"sources-{index % 64:02d}"
            folder.mkdir(exist_ok=True)
            (folder / f"unit-{index:05d}.py").write_bytes(data)
        filesystem = measure(root, arguments.files, len(data))
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        git = measure(root, arguments.files, len(data))
        if filesystem["sha256"] != git["sha256"]:
            raise ValueError("Windows Git/filesystem source acquisition parity failed")
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "filesystem": filesystem,
        "git": git,
        "scope": "inventory/read/hash; parsing and detector inference excluded",
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
