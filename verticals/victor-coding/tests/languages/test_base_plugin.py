# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Shared helpers on BaseLanguagePlugin."""

from types import SimpleNamespace

import pytest

from victor_coding.languages.base import BaseLanguagePlugin
from victor_coding.languages.plugins import additional, cpp, go, java, javascript
from victor_coding.languages.plugins import python as python_plugin
from victor_coding.languages.plugins import rust, typescript


class TestGetNodeText:
    def test_decodes_bytes(self):
        assert BaseLanguagePlugin._get_node_text(SimpleNamespace(text=b"foo_bar")) == "foo_bar"

    def test_invalid_utf8_is_ignored_not_raised(self):
        assert BaseLanguagePlugin._get_node_text(SimpleNamespace(text=b"ok\xff")) == "ok"

    def test_str_text_passes_through(self):
        assert BaseLanguagePlugin._get_node_text(SimpleNamespace(text="already")) == "already"

    @pytest.mark.parametrize("node", [None, SimpleNamespace()])
    def test_missing_node_or_text_is_none(self, node):
        assert BaseLanguagePlugin._get_node_text(node) is None


PLUGIN_MODULES = [additional, cpp, go, java, javascript, python_plugin, rust, typescript]


def _plugin_classes():
    for module in PLUGIN_MODULES:
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, BaseLanguagePlugin)
                and value is not BaseLanguagePlugin
                and value.__module__ == module.__name__
            ):
                yield value


def test_every_plugin_uses_the_shared_helper():
    """The helper used to be copied into 13 plugin classes; new copies must not reappear."""
    classes = list(_plugin_classes())
    assert len(classes) >= 13
    for cls in classes:
        assert "_get_node_text" not in vars(cls), f"{cls.__name__} redefines _get_node_text"
        assert cls._get_node_text(SimpleNamespace(text=b"x")) == "x"
