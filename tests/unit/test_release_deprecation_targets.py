"""Release guards for live Victor AI deprecation notices."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = ROOT / "scripts" / "ci" / "select_changed_tests.py"
spec = importlib.util.spec_from_file_location("release_changed_test_selector", SELECTOR_PATH)
assert spec is not None
assert spec.loader is not None
selector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = selector
spec.loader.exec_module(selector)


@pytest.mark.parametrize("relative_path", selector.DEPRECATION_NOTICE_FILES)
def test_live_victor_ai_removal_notices_target_0_11(relative_path: str) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")

    stale_notice = re.search(
        r"(?:will be removed|backward compatibility until).*0\.10\.0",
        text,
        flags=re.IGNORECASE,
    )
    assert stale_notice is None, f"{relative_path} still promises removal in 0.10.0"
    assert "0.11.0" in text, f"{relative_path} does not state the deferred removal target"
