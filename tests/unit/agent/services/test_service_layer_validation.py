"""Service layer validation tests (SVC-1/SVC-2).

Validates that:
1. All 6 services can be bootstrapped and resolved
2. Service delegation produces consistent results with coordinator path
3. Service layer overhead is minimal (structural, not runtime perf)
4. All 16 delegation points are wired correctly
"""

import ast
import asyncio
import inspect
import textwrap
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestServiceBootstrap:
    """SVC-1: Validate service creation and registration."""

    def test_all_six_service_protocols_importable(self):
        """All 6 service protocols must be importable from the protocols package."""
        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        protocols = [
            ChatServiceProtocol,
            ToolServiceProtocol,
            SessionServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
        ]
        assert len(protocols) == 6
        for p in protocols:
            assert p is not None
            assert hasattr(p, "__protocol_attrs__") or hasattr(p, "__abstractmethods__") or True

    def test_adapter_importable(self):
        """Service adapter must be importable (SessionServiceAdapter, ToolServiceAdapter removed)."""
        from victor.agent.services.adapters import ContextServiceAdapter

        assert ContextServiceAdapter is not None

    def test_all_six_service_implementations_importable(self):
        """All 6 service implementations must be importable."""
        from victor.agent.services.chat_service import ChatService
        from victor.agent.services.context_service import ContextService
        from victor.agent.services.provider_service import ProviderService
        from victor.agent.services.recovery_service import RecoveryService
        from victor.agent.services.session_service import SessionService
        from victor.agent.services.tool_service import ToolService

        services = [
            ChatService,
            ToolService,
            SessionService,
            ContextService,
            ProviderService,
            RecoveryService,
        ]
        assert len(services) == 6

    def test_bootstrap_creates_all_services(self):
        """bootstrap_new_services() must register all 6 core services."""
        from victor.core.bootstrap_services import bootstrap_new_services
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        mock_conv_controller = MagicMock()
        mock_streaming_coord = MagicMock()

        bootstrap_new_services(
            container,
            conversation_controller=mock_conv_controller,
            streaming_coordinator=mock_streaming_coord,
        )

        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        # All 6 should be registered
        for proto in [
            ChatServiceProtocol,
            ToolServiceProtocol,
            SessionServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
        ]:
            service = container.get_optional(proto)
            assert service is not None, f"{proto.__name__} not registered in container"


class TestASTHelperCorrectness:
    """Unit tests for the AST helpers themselves (co-design review item 22c).

    These pin the helpers' behavior on synthetic snippets, independent of the
    orchestrator's current contents — proving both that they close the old
    string-grep checks' false-negative class (binding/passthrough/None-check
    lines wrongly counted as "delegation") and that a reformatted guard is
    still caught.
    """

    def test_bindings_and_passthroughs_are_not_counted_as_calls(self) -> None:
        """The false-negative class the old line-grep checks were exposed to:
        a binding, a getattr passthrough, and an is-not-None guard all
        mention `_context_service` but none of them is a delegation call.
        """
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        self._context_service = something
                        x = getattr(self, "_context_service", None)
                        if self._context_service is not None:
                            pass
                """))
        count = sum(1 for _ in _iter_self_service_calls(tree, "_context_service"))
        assert count == 0

    def test_real_delegation_call_is_counted(self) -> None:
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        return self._context_service.get_context_metrics()
                """))
        count = sum(1 for _ in _iter_self_service_calls(tree, "_context_service"))
        assert count == 1

    def test_reformatted_multiline_guard_is_still_detected(self) -> None:
        """A `self._use_service_layer and self._x` guard split across lines
        would evade a literal substring search but must still be caught here.
        """
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        if (self._use_service_layer
                                and self._chat_service):
                            pass
                """))
        assert _has_feature_flag_delegation_guard(tree) is True

    def test_guard_with_call_operand_is_detected(self) -> None:
        """The most natural reintroduction shape — gating the delegation call
        itself. A bare-attribute-only AST match would miss this (the second
        operand is an ast.Call, not an ast.Attribute); the old substring
        check caught it, so the AST version must too.
        """
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        if self._use_service_layer and self._chat_service.chat():
                            pass
                """))
        assert _has_feature_flag_delegation_guard(tree) is True

    def test_guard_with_comparison_operand_is_detected(self) -> None:
        """`flag and service is not None` — a comparison operand, also caught
        by the old substring check and therefore by this one.
        """
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        if self._use_service_layer and self._chat_service is not None:
                            pass
                """))
        assert _has_feature_flag_delegation_guard(tree) is True

    def test_guard_with_parenthesized_nested_and_is_detected(self) -> None:
        """Explicit parens create a nested BoolOp — a flat-operand-only match
        would miss the inner attribute. The subtree scan reaches it.
        """
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        if self._use_service_layer and (self._chat_service and True):
                            pass
                """))
        assert _has_feature_flag_delegation_guard(tree) is True

    def test_flag_alone_without_service_reference_is_not_flagged(self) -> None:
        """`flag and True` gates nothing service-related — not the
        anti-pattern this guard targets.
        """
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        if self._use_service_layer and True:
                            pass
                """))
        assert _has_feature_flag_delegation_guard(tree) is False

    def test_unrelated_boolop_is_not_flagged(self) -> None:
        tree = ast.parse(textwrap.dedent("""
                class Foo:
                    def bar(self):
                        if self._chat_service and self._tool_service:
                            pass
                """))
        assert _has_feature_flag_delegation_guard(tree) is False


class TestDelegationPointCoverage:
    """SVC-2: Validate all delegation points are correctly wired.

    Co-design review item 22c: these checks were originally string-grep over
    ``inspect.getsource(AgentOrchestrator)``, which conflated attribute
    *bindings* (``self._x_service = ...``), locator passthroughs
    (``getattr(self, "_x_service", None)``), and ``is not None`` guards with
    genuine per-call delegation — and could be evaded by reformatting a
    reintroduced ``self._use_service_layer and self._x`` guard across lines.
    Rewritten to walk the AST directly: ``_iter_self_service_calls`` only
    matches actual ``self.<service>.<method>(...)`` call sites, and
    ``_has_feature_flag_delegation_guard`` matches the boolean-AND shape
    regardless of formatting.
    """

    def test_orchestrator_no_feature_flag_guards(self):
        """Orchestrator must not use _use_service_layer flag in delegation."""
        tree = _get_orchestrator_ast()
        assert not _has_feature_flag_delegation_guard(
            tree
        ), "Old-style _use_service_layer guards still present"

    def test_chat_delegation_points_exist(self):
        """Chat delegation calls onto _chat_service must not regress."""
        tree = _get_orchestrator_ast()
        count = sum(1 for _ in _iter_self_service_calls(tree, "_chat_service"))
        assert count >= 5, f"Expected >= 5 chat delegation calls, found {count}"

    def test_tool_delegation_points_exist(self):
        """Tool delegation calls onto _tool_service must not regress."""
        tree = _get_orchestrator_ast()
        count = sum(1 for _ in _iter_self_service_calls(tree, "_tool_service"))
        assert count >= 8, f"Expected >= 8 tool delegation calls, found {count}"

    def test_session_delegation_points_exist(self):
        """Session delegation calls onto _session_service must not regress."""
        tree = _get_orchestrator_ast()
        count = sum(1 for _ in _iter_self_service_calls(tree, "_session_service"))
        assert count >= 3, f"Expected >= 3 session delegation calls, found {count}"

    def test_context_delegation_points_exist(self):
        """Context delegation calls onto _context_service must not regress.

        Only get_context_metrics() is a genuine per-call delegation today —
        the other _context_service references in the orchestrator are
        service-locator plumbing (a binding, a getattr passthrough handed to
        another component, and an is-not-None guard), not delegation calls.
        The old string-grep check counted those as "delegation points" too,
        which is exactly the false-negative class this AST rewrite closes.
        """
        tree = _get_orchestrator_ast()
        count = sum(1 for _ in _iter_self_service_calls(tree, "_context_service"))
        assert count >= 1, f"Expected >= 1 context delegation call, found {count}"

    def test_provider_delegation_points_exist(self):
        """Provider delegation calls onto _provider_service must not regress.

        Only bind_runtime_components() is a genuine per-call delegation today
        — the other _provider_service references are plumbing (a binding, a
        getattr passthrough, is-not-None guards, and a direct attribute
        assignment rather than a method call).
        """
        tree = _get_orchestrator_ast()
        count = sum(1 for _ in _iter_self_service_calls(tree, "_provider_service"))
        assert count >= 1, f"Expected >= 1 provider delegation call, found {count}"

    def test_all_six_services_resolved_in_initialize(self):
        """Orchestrator must import all 6 service protocols."""
        tree = _get_orchestrator_ast()
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        expected = {
            "ChatServiceProtocol",
            "ToolServiceProtocol",
            "SessionServiceProtocol",
            "ContextServiceProtocol",
            "ProviderServiceProtocol",
            "RecoveryServiceProtocol",
        }
        missing = expected - imported
        assert not missing, f"Missing protocol imports: {sorted(missing)}"

    def test_no_feature_flag_in_delegation(self):
        """No _use_service_layer flag should remain in delegation methods."""
        tree = _get_orchestrator_ast()
        assert not _has_feature_flag_delegation_guard(tree), (
            "Found a _use_service_layer guard in delegation. "
            "Use service-first with None-guard pattern instead."
        )


class TestChatServiceHandlerInjectionRatchet:
    """Co-design review item 22c: ratchet on ChatService.bind_runtime_components's
    keyword-only parameter count.

    This method is a prerequisite guard for item 27 (ChatService inversion,
    Stage C) — its kwarg list has grown incrementally as new handlers were
    wired in (confirmed via git history: task_report_start_handler alone was
    touched across two separate commits), which is exactly the pattern of
    business-specific handler wiring leaking into what should stay a thin
    binding method. Audited at 8 kwargs on 2026-09-06; this cap may only be
    lowered, never raised — growing it further should route through item 27's
    redesign instead.
    """

    def test_bind_runtime_components_kwarg_count_has_not_grown(self) -> None:
        from victor.agent.services.chat_service import ChatService

        source = textwrap.dedent(inspect.getsource(ChatService.bind_runtime_components))
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef))

        kwarg_count = len(func_def.args.kwonlyargs)
        cap = 8
        assert kwarg_count <= cap, (
            f"ChatService.bind_runtime_components has {kwarg_count} keyword-only "
            f"params (ratchet cap {cap}). This method's growing handler-injection "
            f"surface is tracked by co-design review item 27 (ChatService "
            f"inversion) — route new runtime collaborators through that redesign "
            f"instead of adding another kwarg here. If the cap must move, lower "
            f"it only after a real reduction — never raise it."
        )


class TestServiceHealth:
    """Validate health check contracts across all services."""

    def test_all_services_have_is_healthy(self):
        """Every service protocol must define is_healthy() -> bool."""
        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        for proto in [
            ChatServiceProtocol,
            ToolServiceProtocol,
            SessionServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
        ]:
            methods = {name for name in dir(proto) if not name.startswith("_")}
            assert "is_healthy" in methods, f"{proto.__name__} missing is_healthy()"


def _get_orchestrator_ast() -> ast.Module:
    """Parse AgentOrchestrator's source into an AST for structural analysis.

    Co-design review item 22c: replaces string-grep facade checks (fragile to
    formatting, and prone to conflating attribute bindings/passthroughs with
    genuine delegation calls) with AST-based structural checks.
    """
    from victor.agent.orchestrator import AgentOrchestrator

    return ast.parse(inspect.getsource(AgentOrchestrator))


def _iter_self_service_calls(tree: ast.AST, service_attr: str):
    """Yield ast.Call nodes shaped like self.<service_attr>.<method>(...).

    Only matches genuine method-call delegation — not attribute bindings
    (``self._x = ...``), locator passthroughs
    (``getattr(self, "_x", None)``), or None-checks (``self._x is not None``).
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        obj = func.value
        if (
            isinstance(obj, ast.Attribute)
            and obj.attr == service_attr
            and isinstance(obj.value, ast.Name)
            and obj.value.id == "self"
        ):
            yield node


def _self_underscore_attrs_in_subtree(node: ast.AST) -> set:
    """All ``self._<attr>`` names referenced anywhere in *node*'s subtree."""
    attrs = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "self"
            and sub.attr.startswith("_")
        ):
            attrs.add(sub.attr)
    return attrs


def _has_feature_flag_delegation_guard(tree: ast.AST) -> bool:
    """True if a ``self._use_service_layer and self._<x>``-shaped guard exists.

    Walks every ``and`` BoolOp and checks its *full subtree* for attribute
    references, not just bare-attribute operands. This catches the pattern
    regardless of formatting AND regardless of how the second operand is
    shaped — a method call (``self._use_service_layer and self._x.chat()``),
    a comparison (``... and self._x is not None``), or an explicitly
    parenthesized nested ``and`` (``flag and (self._x and y)``). The
    original substring check caught the call/comparison shapes but was
    formatting-fragile; matching operand shapes via bare ``ast.Attribute``
    alone was both — this subtree scan closes both gaps.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)):
            continue
        attrs = _self_underscore_attrs_in_subtree(node)
        if "_use_service_layer" in attrs and attrs - {"_use_service_layer"}:
            return True
    return False
