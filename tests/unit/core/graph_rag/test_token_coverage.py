"""Physical coverage cannot be inferred from thresholded or virtual-unit totals."""

import copy

import pytest

from victor.core.graph_rag.token_coverage import (
    FileAccounting,
    physical_file_summary,
    proven_below_threshold,
)


def test_per_file_summary_requires_all_declared_rows_and_acquired_byte_counts():
    manifest = {"a.rs": {"bytes": 12, "native_lines": 1}}
    raw = {
        "summary": {
            "by": "tokens",
            "totalFiles": 1,
            "files": [{"path": "a.rs", "format": "rust", "bytes": 12, "lines": 1, "tokens": 4}],
        }
    }
    assert physical_file_summary(raw, manifest, metric="tokens")["a.rs"].tokens == 4
    for key, value in [("bytes", 11), ("path", "foreign.rs"), ("tokens", True)]:
        broken = copy.deepcopy(raw)
        broken["summary"]["files"][0][key] = value
        with pytest.raises(ValueError):
            physical_file_summary(broken, manifest, metric="tokens")
    raw["summary"]["totalFiles"] = 2
    with pytest.raises(ValueError, match="truncated"):
        physical_file_summary(raw, manifest, metric="tokens")


def test_only_proven_ineligible_omissions_can_complete_coverage():
    info = {"native_lines": 10}
    assert proven_below_threshold(
        FileAccounting("rust", 12, 100), info, min_tokens=300, min_lines=3
    )
    assert not proven_below_threshold(
        FileAccounting("rust", 400, 1000), info, min_tokens=300, min_lines=3
    )
    assert not proven_below_threshold(
        FileAccounting("astro", 0, 1000), info, min_tokens=300, min_lines=3
    )
    assert proven_below_threshold(FileAccounting("astro", 0, 12), info, min_tokens=300, min_lines=3)
    assert proven_below_threshold(
        FileAccounting("rust", 400, 1000), {"native_lines": 1}, min_tokens=300, min_lines=3
    )
