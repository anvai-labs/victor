# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Rust language plugin."""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    BuildSystem,
    CallEdge,
    CommentStyle,
    DocCommentPattern,
    EdgeDetectionResult,
    Formatter,
    LanguageCapabilities,
    LanguageConfig,
    Linter,
    QueryPattern,
    TestRunner,
    TreeSitterQueries,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


class RustPlugin(BaseLanguagePlugin):
    """Rust language plugin.

    Supports:
    - Testing: cargo test
    - Formatting: rustfmt
    - Linting: clippy
    - Building: cargo build
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="rust",
            display_name="Rust",
            aliases=["rs"],
            extensions=[".rs"],
            filenames=["Cargo.toml", "Cargo.lock"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"'],
            indent_size=4,
            use_tabs=False,
            package_managers=["cargo"],
            build_systems=["cargo"],
            test_frameworks=["cargo test"],
            language_server="rust-analyzer",
            language_server_name="rust-analyzer",
            tree_sitter_language="rust",
            doc_comment_pattern=DocCommentPattern(
                line_prefixes=["///", "//!"],
            ),
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=True,
            supports_inline=True,
            supports_organize_imports=True,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=True,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Create tree-sitter queries for Rust symbol/call extraction."""
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(struct_item name: (type_identifier) @name)"),
                QueryPattern("class", "(enum_item name: (type_identifier) @name)"),
                QueryPattern("class", "(trait_item name: (type_identifier) @name)"),
                QueryPattern("function", "(function_item name: (identifier) @name)"),
                # impl blocks wrap their methods in a `declaration_list`
                # body — the previous pattern (without declaration_list)
                # only compiled because of a query-cache collision masking
                # the structural mismatch.
                QueryPattern(
                    "function",
                    "(impl_item body: (declaration_list (function_item name: (identifier) @name)))",
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (field_expression field: (field_identifier) @callee))
                (call_expression function: (scoped_identifier name: (identifier) @callee))
            """,
            references="""
                (call_expression function: (identifier) @name)
                (call_expression function: (field_expression field: (field_identifier) @name))
                (call_expression function: (scoped_identifier name: (identifier) @name))
                (identifier) @name
                (type_identifier) @name
            """,
            implements="""
                (impl_item
                    trait: (type_identifier) @interface
                    type: (type_identifier) @child)
            """,
            composition="""
                (struct_item
                    name: (type_identifier) @owner
                    body: (field_declaration_list
                        (field_declaration
                            type: (type_identifier) @type)))
            """,
            enclosing_scopes=[
                ("function_item", "name"),
                ("impl_item", "type"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get cargo test runner."""
        cargo_toml = project_root / "Cargo.toml"

        if not cargo_toml.exists():
            return None

        return TestRunner(
            name="cargo test",
            command=["cargo", "test"],
            file_pattern="*_test.rs",
            discover_args=["--no-run"],
            run_args=["--", "--nocapture"],
            coverage_args=[
                "--",
                "--show-output",
            ],  # Use cargo-tarpaulin for real coverage
            parallel_args=["--", "--test-threads=auto"],
            output_format="text",
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get rustfmt formatter."""
        return Formatter(
            name="rustfmt",
            command=["cargo", "fmt"],
            check_args=["--check"],
            config_file="rustfmt.toml",
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get clippy linter."""
        return Linter(
            name="clippy",
            command=["cargo", "clippy"],
            fix_args=["--fix", "--allow-dirty"],
            output_format="text",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get cargo build system."""
        cargo_toml = project_root / "Cargo.toml"

        if not cargo_toml.exists():
            return None

        return BuildSystem(
            name="cargo",
            build_command=["cargo", "build"],
            run_command=["cargo", "run"],
            clean_command=["cargo", "clean"],
            install_command=["cargo", "install", "--path", "."],
            debug_args=[],
            release_args=["--release"],
            manifest_file="Cargo.toml",
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Rust source code.

        Emits one CallEdge per call/method/macro invocation. For method
        calls (`obj.method()`), populates ``receiver_type`` via a best-effort
        local-scope walk: parameter type annotations, ``let`` bindings with
        explicit types, ``let`` bindings to ``Type::new()``-style
        constructors, and ``self`` resolved against the enclosing ``impl T``
        block. Cases requiring full type inference (method chains, struct
        field types, generics) leave ``receiver_type`` as ``None`` so the
        downstream resolver can fall back to name-only matching.
        """
        calls: List[CallEdge] = []
        # Pre-pass: collect intra-file struct field declarations so
        # `self.field.method()` and longer field chains can resolve their
        # receiver type. Also collect impl-method return types so method
        # chains (`a.foo().bar()`) resolve through `foo`'s return type.
        # Rust allows forward references, so we walk the whole tree once
        # before the main pass.
        struct_fields = self._collect_struct_fields(tree.root_node)
        impl_returns = self._collect_impl_method_returns(tree.root_node)
        for call_node, caller_name, caller_line, receiver_type in self._find_call_nodes(
            tree.root_node, struct_fields, impl_returns
        ):
            callee_name = self._extract_callee_name(call_node)
            if callee_name and caller_name:
                calls.append(
                    CallEdge(
                        caller_name=caller_name,
                        callee_name=callee_name,
                        caller_line=caller_line,
                        receiver_type=receiver_type,
                        is_method_call=self._is_method_syntax_call(call_node),
                    )
                )

        logger.debug(f"Detected {len(calls)} CALLS edges in {file_path.name}")

        return EdgeDetectionResult(
            calls=calls,
            metadata={
                "language": "rust",
                "file": str(file_path),
            },
        )

    def _collect_struct_fields(self, root: "Node") -> dict:
        """Walk the file once and return ``{struct_name: {field_name: type}}``.

        Cross-file struct field types are out of scope for v1; this only
        looks at ``struct_item`` nodes declared in the same file as the
        receiver site. That's sufficient for the common
        ``impl Foo { fn x(&self) { self.field.method() } }`` pattern,
        which is overwhelmingly intra-file in idiomatic Rust.
        """
        result: dict = {}

        def walk(node: "Node") -> None:
            if node.type == "struct_item":
                name_node = node.child_by_field_name("name")
                body = node.child_by_field_name("body")
                if name_node is not None and body is not None:
                    struct_name = self._get_node_text(name_node)
                    if struct_name:
                        fields: dict = {}
                        for child in body.children:
                            if child.type != "field_declaration":
                                continue
                            fname_node = child.child_by_field_name("name")
                            ftype_node = child.child_by_field_name("type")
                            if fname_node is None or ftype_node is None:
                                continue
                            fname = self._get_node_text(fname_node)
                            ftype = self._extract_type_str(ftype_node)
                            if fname and ftype:
                                fields[fname] = ftype
                        if fields:
                            result[struct_name] = fields
            for child in node.children:
                walk(child)

        walk(root)
        return result

    def _collect_impl_method_returns(self, root: "Node") -> dict:
        """Walk the file once and return ``{(impl_type, method_name): return_type}``.

        Used to resolve method chains like ``a.foo().bar()``: once we know
        ``a: T``, looking up ``(T, foo)`` gives the return type that
        ``bar`` is called on. Methods with no ``->`` (returning unit)
        are simply absent from the map, so the lookup falls back to
        ``None`` and the chain degrades to name-only resolution.

        Cross-file return types are out of scope for v1 (same constraint
        as struct fields). Idiomatic Rust keeps impls near their type's
        public surface, so intra-file coverage is high.
        """
        result: dict = {}

        def walk(node: "Node", current_impl: Optional[str]) -> None:
            if node.type == "impl_item":
                impl_type = self._extract_impl_type(node)
                for child in node.children:
                    walk(child, impl_type)
                return
            if node.type == "function_item" and current_impl:
                name_node = node.child_by_field_name("name")
                ret_node = node.child_by_field_name("return_type")
                if name_node is not None and ret_node is not None:
                    method_name = self._get_node_text(name_node)
                    ret_type = self._extract_type_str(ret_node)
                    if method_name and ret_type:
                        result[(current_impl, method_name)] = ret_type
                # Don't recurse into the function body for return-type
                # collection — inner closures/items can't define impl methods.
                return
            for child in node.children:
                walk(child, current_impl)

        walk(root, None)
        return result

    def _find_call_nodes(
        self,
        root: "Node",
        struct_fields: Optional[dict] = None,
        impl_returns: Optional[dict] = None,
    ) -> List[tuple["Node", str, Optional[int], Optional[str]]]:
        """Walk the Rust AST tracking impl/function/local scope per call site.

        Returns 4-tuples ``(call_node, enclosing_function_name, line,
        receiver_type)``. ``receiver_type`` is set only for method calls
        whose receiver could be resolved through the local-scope walk; plain
        function calls, macro invocations, and method calls on untracked
        variables get ``None``.

        We do not reuse ``ConfigurableASTTraverser`` here because that
        helper only carries the enclosing function name; receiver-type
        inference needs impl context, parameter types, and ``let``-binding
        order, none of which the shared traverser models.
        """
        results: List[tuple["Node", str, Optional[int], Optional[str]]] = []
        _SKIP = {
            "string_literal",
            "raw_string_literal",
            "line_comment",
            "block_comment",
        }

        def walk(
            node: "Node",
            scope_stack: List[dict],
            caller_name: str,
            current_impl: Optional[str],
        ) -> None:
            nt = node.type
            if nt in _SKIP:
                return

            if nt == "impl_item":
                impl_type = self._extract_impl_type(node)
                for child in node.children:
                    walk(child, scope_stack, caller_name, impl_type)
                return

            if nt == "function_item":
                name_node = node.child_by_field_name("name")
                fn_name = self._get_node_text(name_node) if name_node else ""
                new_scope = {"bindings": {}, "impl_type": current_impl}
                self._populate_parameter_bindings(node, new_scope)
                for child in node.children:
                    walk(
                        child,
                        scope_stack + [new_scope],
                        fn_name or caller_name,
                        current_impl,
                    )
                return

            if nt == "let_declaration":
                if scope_stack:
                    binding = self._extract_let_binding(node)
                    if binding:
                        scope_stack[-1]["bindings"][binding[0]] = binding[1]
                # Still recurse — nested calls in RHS need to be captured.
                for child in node.children:
                    walk(child, scope_stack, caller_name, current_impl)
                return

            # Macros (println!, format!, vec!, write!, ...) are not function
            # calls -- they expand at compile time, and emitting them as
            # CallEdges causes the resolver to fan out to user-defined
            # functions with the same leaf name (e.g. format! bound to a
            # user-defined `fn format`). Skip them entirely.
            if nt == "call_expression":
                receiver_type = self._infer_receiver_type(
                    node, scope_stack, struct_fields or {}, impl_returns or {}
                )
                results.append((node, caller_name, node.start_point[0] + 1, receiver_type))

            for child in node.children:
                walk(child, scope_stack, caller_name, current_impl)

        walk(root, [], "", None)
        return results

    def _extract_impl_type(self, impl_node: "Node") -> Optional[str]:
        """Return the type implemented by an ``impl_item`` (``impl Foo`` → ``"Foo"``).

        For ``impl Trait for Foo``, tree-sitter-rust still exposes ``Foo`` via
        the ``type`` field. Generic impls like ``impl<T> Foo<T>`` collapse to
        ``"Foo"`` (the generic args are dropped for resolution purposes).
        """
        type_node = impl_node.child_by_field_name("type")
        if type_node is None:
            return None
        return self._extract_type_str(type_node)

    def _populate_parameter_bindings(self, fn_node: "Node", scope: dict) -> None:
        """Add ``param_name -> type`` bindings (and ``self`` if applicable)."""
        params_node = fn_node.child_by_field_name("parameters")
        if params_node is None:
            return
        for param in params_node.children:
            if param.type == "self_parameter":
                if scope.get("impl_type"):
                    scope["bindings"]["self"] = scope["impl_type"]
                continue
            if param.type != "parameter":
                continue
            pattern = param.child_by_field_name("pattern")
            type_node = param.child_by_field_name("type")
            if pattern is None or type_node is None:
                continue
            if pattern.type != "identifier":
                continue
            name = self._get_node_text(pattern)
            type_str = self._extract_type_str(type_node)
            if name and type_str:
                scope["bindings"][name] = type_str

    def _extract_let_binding(self, let_node: "Node") -> Optional[tuple[str, str]]:
        """For ``let x: T = ...`` or ``let x = T::new()``, return ``(x, T)``."""
        pattern = let_node.child_by_field_name("pattern")
        if pattern is None or pattern.type != "identifier":
            return None
        name = self._get_node_text(pattern)
        if not name:
            return None

        type_annot = let_node.child_by_field_name("type")
        if type_annot is not None:
            type_str = self._extract_type_str(type_annot)
            if type_str:
                return (name, type_str)

        value = let_node.child_by_field_name("value")
        if value is not None and value.type == "call_expression":
            fn_child = value.child_by_field_name("function")
            if fn_child is not None and fn_child.type == "scoped_identifier":
                type_str = self._extract_constructor_type(fn_child)
                if type_str:
                    return (name, type_str)
        return None

    def _extract_type_str(self, type_node: "Node") -> Optional[str]:
        """Flatten a type AST node to a bare type name (``&Foo`` → ``"Foo"``)."""
        nt = type_node.type
        if nt == "type_identifier":
            return self._get_node_text(type_node)
        if nt == "reference_type":
            inner = type_node.child_by_field_name("type")
            if inner is not None:
                return self._extract_type_str(inner)
            for child in type_node.children:
                if child.type in ("type_identifier", "generic_type", "reference_type"):
                    return self._extract_type_str(child)
            return None
        if nt == "generic_type":
            base = type_node.child_by_field_name("type")
            if base is not None:
                return self._extract_type_str(base)
            return None
        if nt == "scoped_type_identifier":
            # std::path::PathBuf → "PathBuf"
            last = None
            for child in type_node.children:
                if child.type == "type_identifier":
                    last = self._get_node_text(child)
            return last
        # Last-resort: find any type_identifier descendant.
        for child in type_node.children:
            if child.type == "type_identifier":
                return self._get_node_text(child)
        return None

    def _extract_constructor_type(self, scoped_id: "Node") -> Optional[str]:
        """For ``Foo::new`` or ``a::b::Foo::default``, return ``"Foo"``."""
        parts: List[str] = []

        def walk_path(node: "Node") -> None:
            for child in node.children:
                if child.type in ("identifier", "type_identifier"):
                    parts.append(self._get_node_text(child) or "")
                elif child.type == "scoped_identifier":
                    walk_path(child)

        walk_path(scoped_id)
        if len(parts) >= 2:
            return parts[-2] or None
        return None

    def _infer_receiver_type(
        self,
        call_node: "Node",
        scope_stack: List[dict],
        struct_fields: dict,
        impl_returns: dict,
    ) -> Optional[str]:
        """For ``obj.method()`` return the inferred type of ``obj``.

        Delegates to ``_infer_value_type`` which handles direct receivers
        (``self``, identifier), nested field access (``self.a.b.method()``),
        method-chain return types (``a.foo().bar()``), constructor chains
        (``Foo::new().method()``), and stops gracefully on unsupported
        expression shapes by returning ``None`` so the downstream resolver
        can fall back to name-only matching.
        """
        if call_node.type == "macro_invocation":
            return None
        fn_child = call_node.child_by_field_name("function")
        if fn_child is None:
            return None
        if fn_child.type == "field_expression":
            value = fn_child.child_by_field_name("value")
            if value is None:
                return None
            return self._infer_value_type(value, scope_stack, struct_fields, impl_returns)
        if fn_child.type == "scoped_identifier":
            # `Type::method()` — bind to the path prefix when it's `Self` (the
            # enclosing impl type) or, conservatively, the first identifier in
            # the path. Both let the downstream resolver pick the right impl
            # instead of fanning out across every same-named function.
            path_root = self._extract_scoped_path_root(fn_child)
            if path_root == "Self":
                for scope in reversed(scope_stack):
                    if scope.get("impl_type"):
                        return scope["impl_type"]
                return None
            return path_root
        return None

    def _is_method_syntax_call(self, call_node: "Node") -> bool:
        """Return True iff this is `obj.method()` dot-dispatch syntax.

        Distinct from plain `func()` and path calls `Foo::bar()`. The
        resolver uses this to pick its fallback policy: a method-syntax
        call with no inferable receiver type is dropped (name-only would
        bind to unrelated user-defined methods with the same leaf name),
        while plain function calls keep the name-only fallback.
        """
        fn_child = call_node.child_by_field_name("function")
        return fn_child is not None and fn_child.type == "field_expression"

    def _extract_scoped_path_root(self, scoped_id: "Node") -> Optional[str]:
        """For ``a::b::c::d`` return ``"c"`` — the type-bearing segment.

        ``scoped_identifier`` nests left-recursively, so the outermost node's
        ``path`` field is the parent path and ``name`` is the final segment.
        ``Foo::new`` -> path="Foo", name="new" -> return "Foo".
        ``a::b::c::d`` -> path="a::b::c" (scoped_identifier), name="d";
        recursively the path-side resolves to "c" (the segment just before
        the called function, which is the impl-bearing type in Rust's
        ``Module::Type::method`` convention).
        """
        path_node = scoped_id.child_by_field_name("path")
        if path_node is None:
            return None
        if path_node.type == "identifier" or path_node.type == "type_identifier":
            return self._get_node_text(path_node)
        if path_node.type == "scoped_identifier":
            inner_name = path_node.child_by_field_name("name")
            if inner_name is not None:
                return self._get_node_text(inner_name)
        return None

    def _infer_value_type(
        self,
        value_node: "Node",
        scope_stack: List[dict],
        struct_fields: dict,
        impl_returns: dict,
    ) -> Optional[str]:
        """Return the inferred static type of any value-position expression.

        Handles:
        * ``self`` -> enclosing ``impl T`` type.
        * ``identifier`` -> scope-stack lookup (parameter, let binding).
        * ``field_expression`` -> recurse on the value side, look up the
          field name in the resulting type's ``struct`` declaration.
        * ``call_expression`` -> if it's a method call, recurse on the
          receiver and look up the return type in ``impl_returns``; if
          it's a path call ``Foo::new()``, return ``Foo``.

        Returns ``None`` for shapes we don't model (index expressions,
        deref, complex patterns, methods with no `->` return) — callers
        fall back to name-only resolution downstream.
        """
        nt = value_node.type
        if nt == "self":
            for scope in reversed(scope_stack):
                if scope.get("impl_type"):
                    return scope["impl_type"]
            return None
        if nt == "identifier":
            var = self._get_node_text(value_node)
            if var is None:
                return None
            for scope in reversed(scope_stack):
                if var in scope["bindings"]:
                    return scope["bindings"][var]
            return None
        if nt == "field_expression":
            inner_value = value_node.child_by_field_name("value")
            field_node = value_node.child_by_field_name("field")
            if inner_value is None or field_node is None:
                return None
            inner_type = self._infer_value_type(
                inner_value, scope_stack, struct_fields, impl_returns
            )
            if inner_type is None:
                return None
            field_name = self._get_node_text(field_node)
            if not field_name:
                return None
            type_fields = struct_fields.get(inner_type)
            if type_fields is None:
                return None
            return type_fields.get(field_name)
        if nt == "call_expression":
            inner_fn = value_node.child_by_field_name("function")
            if inner_fn is None:
                return None
            if inner_fn.type == "field_expression":
                # Method call: receiver.method() — look up method's return.
                inner_receiver = inner_fn.child_by_field_name("value")
                inner_method = inner_fn.child_by_field_name("field")
                if inner_receiver is None or inner_method is None:
                    return None
                receiver_type = self._infer_value_type(
                    inner_receiver, scope_stack, struct_fields, impl_returns
                )
                if receiver_type is None:
                    return None
                method_name = self._get_node_text(inner_method)
                if not method_name:
                    return None
                return impl_returns.get((receiver_type, method_name))
            if inner_fn.type == "scoped_identifier":
                # Path call: Foo::new(), a::b::Type::default() — assume the
                # call returns the type to the left of the final segment,
                # which is the constructor convention for ::new/::default/etc.
                return self._extract_constructor_type(inner_fn)
            return None
        return None

    def _extract_callee_name(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Field access calls: obj.method()
        - Scoped calls: std::mem::drop()
        - Macro calls: println!, vec!
        """
        # For macro invocations
        if call_node.type == "macro_invocation":
            for child in call_node.children:
                if child.type == "identifier":
                    return self._get_node_text(child)
                elif child.type == "scoped_identifier":
                    # std::println -> extract "println"
                    return self._extract_scoped_name(child)
            return None

        # For call expressions
        for child in call_node.children:
            if child.type == "field_expression":
                # obj.method() -> extract "method"
                return self._extract_field_name(child)
            elif child.type == "scoped_identifier":
                # std::mem::drop() -> extract "drop"
                return self._extract_scoped_name(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_field_name(self, field_node: "Node") -> Optional[str]:
        """Extract field name from a field_expression node.

        For obj.method, extracts "method".
        For obj.field1.field2, extracts "field2" (the final field).
        """
        # field_expression: value (identifier or field_expression) . field_ref (field_identifier)
        for child in reversed(field_node.children):
            if child.type == "field_identifier":
                return self._get_node_text(child)
            elif child.type == "field_expression":
                result = self._extract_field_name(child)
                if result:
                    return result

        return None

    def _extract_scoped_name(self, scoped_node: "Node") -> Optional[str]:
        """Extract the final identifier from a scoped identifier.

        For std::mem::drop, extracts "drop".
        For std::collections::HashMap, extracts "HashMap".
        """
        # scoped_identifier: [identifier::] identifier
        # We want the last identifier
        for child in reversed(scoped_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "scoped_identifier":
                result = self._extract_scoped_name(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node."""
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text
