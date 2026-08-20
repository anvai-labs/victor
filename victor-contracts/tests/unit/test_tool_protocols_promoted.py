# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for promoted tool protocols — TDD RED phase."""

from __future__ import annotations


from victor_contracts.verticals.protocols.tools import (
    MessageData,
    ProviderRegistryProtocol,
    ToolRegistryProtocol,
    ToolResultData,
)


class TestToolResultData:
    def test_fields(self):
        r = ToolResultData(success=True, output="ok")
        assert r.success is True
        assert r.output == "ok"
        assert r.error is None
        assert r.metadata is None

    def test_failure_with_error(self):
        r = ToolResultData(success=False, error="not found")
        assert r.success is False
        assert r.error == "not found"

    def test_with_metadata(self):
        r = ToolResultData(success=True, output="x", metadata={"key": "val"})
        assert r.metadata == {"key": "val"}


class TestMessageData:
    def test_fields(self):
        m = MessageData(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_optional_name(self):
        m = MessageData(role="assistant", content="hi", name="bot")
        assert m.name == "bot"

    def test_default_name_none(self):
        m = MessageData(role="system", content="sys")
        assert m.name is None


class TestToolRegistryProtocol:
    def test_structural_check(self):
        class FakeRegistry:
            def register(self, name, tool):
                pass

            def get(self, name):
                return None

            def list_tools(self):
                return []

            def is_enabled(self, name):
                return True

        assert isinstance(FakeRegistry(), ToolRegistryProtocol)

    def test_rejects_incomplete(self):
        class Incomplete:
            def register(self, name, tool):
                pass

        assert not isinstance(Incomplete(), ToolRegistryProtocol)


class TestProviderRegistryProtocol:
    def test_structural_check(self):
        class FakeProviderReg:
            def get_provider(self, name):
                return None

            def list_providers(self):
                return []

        assert isinstance(FakeProviderReg(), ProviderRegistryProtocol)


class TestZeroDependency:
    def test_no_pydantic_import(self):
        """Promoted types must not import from pydantic."""
        import ast
        import sys

        tool_mod = sys.modules.get("victor_contracts.verticals.protocols.tools")
        source = tool_mod.__file__
        with open(source) as f:
            tree = ast.parse(f.read())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any("pydantic" in imp for imp in imports)
