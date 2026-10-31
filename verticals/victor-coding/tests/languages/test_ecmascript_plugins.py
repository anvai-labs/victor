# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Characterization tests for the JavaScript and TypeScript language plugins.

The two plugins share an ECMAScript base. These golden tests pin, for both
plugins, the observable output of every method that base touches -- call
node discovery, function/callee/member name extraction, call edges, test
runner, build system, formatter, linter, config and capabilities -- so the
shared code cannot silently change either plugin. Values were captured from
the pre-refactor per-plugin implementations and document current behaviour
(including quirks), not ideal behaviour.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from victor_coding.languages.plugins.javascript import JavaScriptPlugin
from victor_coding.languages.plugins.typescript import TypeScriptPlugin

FUNCTION_NODE_TYPES = {
    "function_declaration",
    "function_expression",
    "method_definition",
    "method_signature",
    "arrow_function",
}
CALL_NODE_TYPES = {"call_expression", "new_expression"}

JS_SOURCE = """\
init();

function helper(x) {
  return Math.max(x, 1);
}

const arrow = (a) => helper(a);
const fexpr = function named(b) { return arrow(b); };
let anon = function () { return obj["dyn"](1); };

class Service extends Base {
  constructor() {
    this.client = new Client();
  }
  run(items) {
    items.forEach((i) => this.process(i));
    return a.b.c.deep(helper(outer(inner())));
  }
}

handler = () => fetchData().then(parse);

class Mixed extends mixin(Base) {
  go() { run(); }
}
"""

TS_SOURCE = """\
init();

interface Greeter {
  greet(name: string): string;
}

function helper(x: number): number {
  return Math.max(x, 1);
}

const arrow = (a: number) => helper(a);
const fexpr = function named(b: number) { return arrow(b); };

class Service extends Base implements Greeter {
  private client: Client;
  constructor() {
    this.client = new Client();
  }
  greet(name: string): string {
    return obj["dyn"](name);
  }
  run(items: number[]) {
    items.forEach((i) => this.process(i));
    return a.b.c.deep(helper(outer(inner())));
  }
}

class Mixed extends mixin(Base) {
  go() { run(); }
}

interface Keyed extends Mixin<typeof make> {
  load(cb = fallback()): void;
  [Symbol.for("k")](): void;
}

function outer() {
  interface Local {
    [make()]: string;
  }
  let shape: { m(a = f()): void };
  inner();
}
"""


def _parse(language: str, source: str) -> Any:
    """Parse with the production grammar, else the language pack, else skip."""
    try:
        from victor_coding.codebase.tree_sitter_manager import get_parser

        return get_parser(language).parse(source.encode("utf-8"))
    except Exception:
        pass
    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language

        return Parser(get_language(language)).parse(source.encode("utf-8"))
    except Exception as exc:  # pragma: no cover - depends on installed grammars
        pytest.skip(f"no tree-sitter grammar available for {language}: {exc}")


def _walk(node: Any):
    yield node
    for child in node.children:
        yield from _walk(child)


def _span(node: Any) -> tuple:
    return (node.type, node.start_point[0] + 1, node.start_point[1])


def snapshot_ast(plugin: Any, language: str, source: str) -> Dict[str, Any]:
    """Observable AST-level outputs of every shared extraction method."""
    root = _parse(language, source).root_node
    nodes = list(_walk(root))
    return {
        "call_nodes": [
            (_span(node), enclosing, line)
            for node, enclosing, line in plugin._find_call_nodes(root)
        ],
        "function_names": [
            (_span(n), plugin._extract_function_name(n))
            for n in nodes
            if n.type in FUNCTION_NODE_TYPES
        ],
        "callee_names": [
            (_span(n), plugin._extract_callee_name(n)) for n in nodes if n.type in CALL_NODE_TYPES
        ],
        "member_names": [
            (_span(n), plugin._extract_member_name(n))
            for n in nodes
            if n.type == "member_expression"
        ],
        "subscript_names": [
            (_span(n), plugin._extract_subscript_name(n))
            for n in nodes
            if n.type == "subscript_expression"
        ],
        "edges": sorted(
            (c.caller_name, c.callee_name, c.caller_line, c.receiver_type, c.is_method_call)
            for c in plugin.detect_calls_edges(
                _parse(language, source), source, Path("sample")
            ).calls
        ),
        "metadata": plugin.detect_calls_edges(
            _parse(language, source), source, Path("sample")
        ).metadata,
    }


# Project layouts for runner/build-system detection: name -> {file: content}.
PROJECTS: Dict[str, Dict[str, str]] = {
    "empty": {},
    "pkg_plain": {"package.json": '{"name": "x", "scripts": {"build": "tsc"}}'},
    "pkg_vitest": {"package.json": '{"devDependencies": {"vitest": "^1.0.0"}}'},
    "pkg_jest": {"package.json": '{"devDependencies": {"jest": "^29.0.0"}}'},
    "pkg_mocha": {"package.json": '{"devDependencies": {"mocha": "^10.0.0"}}'},
    "pkg_jest_script": {"package.json": '{"scripts": {"test": "jest --ci"}}'},
    "pkg_vitest_and_jest": {
        "package.json": '{"devDependencies": {"jest": "1", "vitest": "1", "mocha": "1"}}'
    },
    "pkg_jest_and_mocha": {"package.json": '{"devDependencies": {"jest": "1", "mocha": "1"}}'},
    "jest_config_only": {"jest.config.js": "module.exports = {};"},
    "vitest_config_only": {"vitest.config.ts": "export default {};"},
    "mocharc_only": {".mocharc.json": "{}"},
    "pkg_yarn": {"package.json": "{}", "yarn.lock": ""},
    "pkg_pnpm": {"package.json": "{}", "pnpm-lock.yaml": ""},
    "pkg_pnpm_yarn": {"package.json": "{}", "pnpm-lock.yaml": "", "yarn.lock": ""},
    "tsconfig_only": {"tsconfig.json": "{}"},
    "tsconfig_and_pkg": {"tsconfig.json": "{}", "package.json": '{"jest": {}}', "yarn.lock": ""},
}


def _make_project(root: Path, files: Dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        (root / rel).write_text(content)
    return root


def _asdict(value: Optional[Any]) -> Optional[Dict[str, Any]]:
    return None if value is None else dataclasses.asdict(value)


def snapshot_tooling(plugin: Any, base: Path) -> Dict[str, Any]:
    """Tooling outputs (runner, build system, formatter, linter) per project layout."""
    out: Dict[str, Any] = {}
    for name, files in PROJECTS.items():
        root = _make_project(base / name, files)
        out[name] = {
            "test_runner": _asdict(plugin.get_test_runner(root)),
            "build_system": _asdict(plugin.get_build_system(root)),
            "formatter": _asdict(plugin.get_formatter(root)),
            "linter": _asdict(plugin.get_linter(root)),
        }
    return out


def snapshot_static(plugin: Any) -> Dict[str, Any]:
    return {
        "config": dataclasses.asdict(plugin.config),
        "capabilities": dataclasses.asdict(plugin.capabilities),
        "queries": dataclasses.asdict(plugin.tree_sitter_queries),
        "call_edge_language": plugin._call_edge_language,
    }


PLUGINS: List[Any] = [
    pytest.param(JavaScriptPlugin, "javascript", JS_SOURCE, id="javascript"),
    pytest.param(TypeScriptPlugin, "typescript", TS_SOURCE, id="typescript"),
]

GOLDEN_PATH = Path(__file__).parent / "golden" / "ecmascript_plugins.json"


def _normalize(value: Any) -> Any:
    """JSON round-trip so tuples/lists compare like the stored golden file."""
    return json.loads(
        json.dumps(value, sort_keys=True, default=lambda o: getattr(o, "value", repr(o)))
    )


def snapshot_plugin(plugin_cls: Any, language: str, source: str, base: Path) -> Dict[str, Any]:
    plugin = plugin_cls()
    return _normalize(
        {
            "ast": snapshot_ast(plugin, language, source),
            "tooling": snapshot_tooling(plugin, base),
            "static": snapshot_static(plugin),
        }
    )


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("plugin_cls,language,source", PLUGINS)
def test_plugin_outputs_match_golden(
    plugin_cls: Any, language: str, source: str, tmp_path: Path, golden: Dict[str, Any]
) -> None:
    snap = snapshot_plugin(plugin_cls, language, source, tmp_path)
    expected = golden[language]
    for section in ("ast", "tooling", "static"):
        for key in expected[section]:
            assert snap[section][key] == expected[section][key], f"{language}.{section}.{key}"
    assert snap == expected


def test_call_edge_language_strings_are_preserved() -> None:
    assert JavaScriptPlugin._call_edge_language == "javascript/typescript"
    assert TypeScriptPlugin._call_edge_language is None
    assert JavaScriptPlugin().config.name == "javascript"
    assert TypeScriptPlugin().config.name == "typescript"


@pytest.mark.parametrize(
    "project,js_runner,ts_runner",
    [
        ("empty", "jest", "jest"),
        ("pkg_vitest", "vitest", "vitest"),
        ("pkg_jest", "jest", "jest"),
        ("pkg_mocha", "mocha", "jest"),
        ("pkg_vitest_and_jest", "vitest", "vitest"),
        ("pkg_jest_and_mocha", "jest", "jest"),
    ],
)
def test_test_runner_selection(
    project: str, js_runner: str, ts_runner: str, tmp_path: Path
) -> None:
    root = _make_project(tmp_path / project, PROJECTS[project])
    js = JavaScriptPlugin().get_test_runner(root)
    ts = TypeScriptPlugin().get_test_runner(root)
    assert (js.name, js.file_pattern) == (js_runner, "*.test.js")
    assert (ts.name, ts.file_pattern) == (ts_runner, "*.test.ts")


def test_parallel_args_differ_between_plugins(tmp_path: Path) -> None:
    root = _make_project(tmp_path / "jest", PROJECTS["pkg_jest"])
    assert JavaScriptPlugin().get_test_runner(root).parallel_args == ["--maxWorkers=auto"]
    assert TypeScriptPlugin().get_test_runner(root).parallel_args == []
    root = _make_project(tmp_path / "vitest", PROJECTS["pkg_vitest"])
    assert JavaScriptPlugin().get_test_runner(root).parallel_args == ["--pool", "threads"]
    assert TypeScriptPlugin().get_test_runner(root).parallel_args == []


@pytest.mark.parametrize(
    "project,js_build,ts_build",
    [
        ("empty", None, None),
        ("pkg_plain", "npm", None),
        ("pkg_yarn", "yarn", None),
        ("pkg_pnpm", "pnpm", None),
        ("pkg_pnpm_yarn", "pnpm", None),
        ("tsconfig_only", None, "tsc"),
        ("tsconfig_and_pkg", "yarn", "tsc"),
    ],
)
def test_build_system_selection(
    project: str, js_build: Optional[str], ts_build: Optional[str], tmp_path: Path
) -> None:
    root = _make_project(tmp_path / project, PROJECTS[project])
    js = JavaScriptPlugin().get_build_system(root)
    ts = TypeScriptPlugin().get_build_system(root)
    assert (js.name if js else None) == js_build
    assert (ts.name if ts else None) == ts_build


def test_method_signature_names_only_in_typescript() -> None:
    root = _parse("typescript", TS_SOURCE).root_node
    sig = next(n for n in _walk(root) if n.type == "method_signature")
    assert TypeScriptPlugin()._extract_function_name(sig) == "greet"
    assert JavaScriptPlugin()._extract_function_name(sig) is None
