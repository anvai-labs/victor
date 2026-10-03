# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""A/B-testing CLI command module: cold-start import hygiene.

The module previously imported ``victor.experiments.ab_testing.paths`` at
module level, dragging the ab_testing package (numpy + scipy via metrics.py)
into the CLI cold-start chain. The import is lazy now; this pins that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ab_testing_command_import_keeps_numpy_and_scipy_off_the_chain():
    """Importing the command module must not pull the experiments package."""
    code = (
        "import sys\n"
        "import victor.ui.commands.ab_testing\n"
        "offenders = [m for m in ('numpy', 'scipy') if m in sys.modules]\n"
        "assert not offenders, f'eager imports leaked: {offenders}'\n"
        "assert 'victor.experiments.ab_testing' not in sys.modules, (\n"
        "    'ab_testing package imported eagerly'\n"
        ")\n"
        "print('LAZY_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (
        result.returncode == 0 and "LAZY_OK" in result.stdout
    ), f"ab_testing CLI module eagerly imports heavy deps:\n{result.stderr[-1200:]}"


def test_list_experiments_lazily_imports_the_paths_helper(monkeypatch):
    """The lazy import inside _list_experiments resolves at call time."""
    import victor.ui.commands.ab_testing as mod

    called = {}

    def fake_path():
        called["hit"] = True
        return Path("/nonexistent")  # triggers the 'No experiments found' path

    import victor.experiments.ab_testing.paths as paths_mod

    monkeypatch.setattr(paths_mod, "get_default_ab_test_db_path", fake_path)

    from victor.ui.commands.ab_testing import _list_experiments

    try:
        _list_experiments(None)
    except Exception:
        # 'No experiments found' exits 0 — exit plumbing varies by typer
        # version (SystemExit / click.exceptions.Exit / typer.exceptions.Exit);
        # the point under test is the lazy paths-helper import, not taxonomy.
        pass
    assert called.get("hit") is True
