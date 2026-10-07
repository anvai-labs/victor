"""Guard: keep the CLI cold-start import chain free of heavy optional deps.

Background: ``python -m victor.ui.cli --help`` once took ~5s because importing
the CLI eagerly pulled ``scipy`` (~180ms) and ``jsonschema``/``rfc3987``
(~660ms). Those expensive optional deps were deferred to first use:

* ``victor/tools/base.py`` — ``jsonschema`` moved function-local into
  ``BaseTool.validate_parameters_detailed``.
* ``victor/experiments/ab_testing/{statistics,metrics}.py`` — ``scipy`` replaced
  with a ``find_spec`` availability flag + a lazy ``stats`` proxy.

This test pins that: after importing the CLI module in a fresh interpreter,
neither may be resident. A regression that re-introduces a module-level
``import scipy`` / ``import jsonschema`` in victor source will turn this red.

NOTE on ``lancedb``/``pyarrow``: these ARE still loaded at CLI import time, but
by EXTERNAL vertical packages (``victor-rag``/``victor-dataanalysis``) during the
mandatory plugin bootstrap (``cli.py::_register_plugin_commands`` ->
``ensure_bootstrapped``) — not by victor-ai source. Deferring the bootstrap would
break plugin-command resolution, so they are intentionally NOT asserted here.
Reducing that remaining cost is tracked as follow-up work.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Heavy optional deps that must NOT be eagerly imported by victor-ai source at
# CLI load time. (lancedb/pyarrow excluded — see module docstring.)
FORBIDDEN_AT_CLI_IMPORT = {
    "scipy",  # victor/experiments/ab_testing — deferred to first use
    "jsonschema",  # victor/tools/base.py — deferred into validate_parameters_detailed
    "rfc3987",  # transitive of jsonschema (format checker)
    "rfc3987_syntax",
}


class TestCliImportWeightGuard:
    """Prevent heavy optional deps from re-entering the CLI cold-start path."""

    def test_cli_import_does_not_eagerly_load_heavy_deps(self) -> None:
        """A fresh interpreter importing victor.ui.cli must leave heavy deps un-imported.

        Two-stage attribution: `import victor` alone must already be clean, so a
        violation is blamed on victor-ai source rather than the plugin
        bootstrap's external verticals (which legitimately load lancedb/pyarrow
        during `victor.ui.cli` import — see docstring NOTE).
        """
        forbidden = tuple(sorted(FORBIDDEN_AT_CLI_IMPORT))
        code = (
            "import sys, json\n"
            f"forbidden = {forbidden!r}\n"
            "import victor\n"
            "stage1 = [m for m in forbidden if m in sys.modules]\n"
            "import victor.ui.cli\n"
            "stage2 = [m for m in forbidden if m in sys.modules]\n"
            "print('LOADED:' + json.dumps({'import_victor': stage1, 'import_cli': stage2}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert (
            result.returncode == 0
        ), f"`import victor.ui.cli` failed in a fresh subprocess:\n{result.stderr}"
        loaded_line = next(
            (ln for ln in result.stdout.splitlines() if ln.startswith("LOADED:")),
            "",
        )
        stages = json.loads(loaded_line[len("LOADED:") :] or "{}")
        assert stages.get("import_victor") == [], (
            "`import victor` eagerly imported heavy optional deps — defer them to "
            f"first use instead of importing at module scope: {stages.get('import_victor')}. "
            "See this module's docstring for the why and the fix pattern."
        )
        assert stages.get("import_cli") == [], (
            "CLI cold-start eagerly imported heavy optional deps — defer them to "
            f"first use instead of importing at module scope: {stages.get('import_cli')}. "
            "See this module's docstring for the why and the fix pattern."
        )

    def test_cli_import_survives_without_numpy(self) -> None:
        """Bare-install simulation: with numpy un-importable, the whole CLI
        chain — import, similarity dispatch, FEP-0012 artifact load + predict —
        must still work (core is numpy-optional; deployment reshaping Phase 1).

        sys.modules["numpy"] = None makes `import numpy` raise ImportError, the
        same failure mode a machine without numpy sees. Unlike a sys.modules
        membership check, this proves FUNCTION under absence, not merely that
        no eager import happened (guarded optional imports legitimately succeed
        on numpy-installed dev machines).
        """
        code = (
            "import sys\n"
            "sys.modules['numpy'] = None\n"
            "import victor\n"
            "import victor.ui.cli\n"
            "from victor.processing.native import batch_cosine_similarity\n"
            "scores = batch_cosine_similarity([1.0, 0.0, 1.0], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])\n"
            "assert len(scores) == 2 and abs(scores[0] - 1.0) < 1e-6, scores\n"
            "from victor.ml.model import EdgeClassifierModel\n"
            "import glob\n"
            "artifact = glob.glob('victor/models/edge_classifier_v1.npz')[0]\n"
            "model = EdgeClassifierModel.load(artifact)\n"
            "label, conf = model.predict('stage_detection', 'run the parser tests')\n"
            "assert conf >= 0.0\n"
            "print('BARE_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0 and "BARE_OK" in result.stdout, (
            "CLI chain broke without numpy — a core-path module hard-imports "
            f"numpy:\n{result.stderr[-1500:]}"
        )

    def test_cli_import_survives_without_textual(self) -> None:
        """Bare-install simulation: with textual un-importable, the CLI chain
        must still work (textual is a `tui` extra, not core)."""
        code = (
            "import sys\n"
            "sys.modules['textual'] = None\n"
            "import victor.ui.cli\n"
            "print('TEXTUAL_BARE_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0 and "TEXTUAL_BARE_OK" in result.stdout, (
            "CLI chain broke without textual — a core-path module hard-imports "
            f"textual:\n{result.stderr[-1500:]}"
        )
