"""Vertical defaults and generated manifests follow the supported host baseline."""

import tomllib

from packaging.specifiers import SpecifierSet

from victor.core.verticals.package_schema import (
    VICTOR_VERTICAL_TOML_TEMPLATE,
    VerticalCompatibility,
)


def test_default_and_template_reject_retired_interpreters():
    template = tomllib.loads(VICTOR_VERTICAL_TOML_TEMPLATE)
    for requirement in (
        VerticalCompatibility().python_version,
        template["vertical"]["compatibility"]["python_version"],
    ):
        supported = SpecifierSet(requirement)
        assert "3.10" not in supported
        assert "3.11" not in supported
        assert "3.12" in supported
        assert "3.13" in supported
