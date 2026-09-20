"""Runtime entry points must reject retired Python versions before loading services."""

import importlib
import sys

import pytest


@pytest.mark.parametrize("package", ["victor", "victor_contracts"])
@pytest.mark.parametrize("minor", [10, 11, 12, 13])
def test_runtime_python_floor(monkeypatch, package, minor):
    module = importlib.import_module(package)
    monkeypatch.setattr(sys, "version_info", (3, minor, 0))
    if minor < 12:
        with pytest.raises(RuntimeError, match=r"requires Python 3\.12\+"):
            module._ensure_supported_python()
    else:
        module._ensure_supported_python()
