"""Make the out-of-package ``web/`` tree importable for these tests.

``web/server/`` is deliberately not part of the installed ``victor`` package —
it is the repo-local web UI backend, imported as a namespace package from the
repo root (``uvicorn web.server.main:app``). Bare ``pytest`` does not put the
repo root on ``sys.path`` (``python -m pytest`` does, via CWD), so bootstrap it
here before test collection imports ``web.server.session_store``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
