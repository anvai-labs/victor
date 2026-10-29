# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Characterization tests for ``detect_calls_edges`` across language plugins.

Each plugin gets one realistic snippet with free-function calls, method
calls, qualified/namespaced calls where the language has them, nested
functions/classes and a call outside any function. The exact sorted edge
tuples ``(caller_name, callee_name, caller_line, receiver_type,
is_method_call)`` and the result ``metadata`` were captured from the
pre-refactor per-plugin implementations and are pinned verbatim: they
document current behaviour (including its quirks), not ideal behaviour.

Parsing prefers the production grammar wheels (``tree_sitter_<lang>`` via
``victor_coding.codebase.tree_sitter_manager``) and falls back to
``tree_sitter_language_pack``; the two produce identical edges for every
language here except Kotlin, whose language-pack grammar is a different
grammar (``simple_identifier`` nodes) that the plugin does not target, so
Kotlin requires the production wheel.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from victor_coding.languages.base import BaseLanguagePlugin
from victor_coding.languages.plugins import additional, cpp, go, java, javascript
from victor_coding.languages.plugins import python as python_plugin
from victor_coding.languages.plugins import rust, typescript

# Languages whose language-pack grammar differs from the production wheel in
# a way the plugin's node-type names depend on.
_PRODUCTION_GRAMMAR_ONLY = {"kotlin"}


def _parse(language: str, source: str) -> Any:
    """Parse with the production grammar, else the language pack, else skip."""
    try:
        from victor_coding.codebase.tree_sitter_manager import get_parser

        return get_parser(language).parse(source.encode("utf-8"))
    except Exception:
        pass
    if language in _PRODUCTION_GRAMMAR_ONLY:
        pytest.skip(f"production tree_sitter grammar for {language} not installed")
    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language

        return Parser(get_language(language)).parse(source.encode("utf-8"))
    except Exception as exc:  # pragma: no cover - depends on installed grammars
        pytest.skip(f"no tree-sitter grammar available for {language}: {exc}")


def _detect(plugin: BaseLanguagePlugin, language: str, source: str, file_name: str):
    tree = _parse(language, source)
    result = plugin.detect_calls_edges(tree, source, Path(file_name))
    edges = sorted(
        (c.caller_name, c.callee_name, c.caller_line, c.receiver_type, c.is_method_call)
        for c in result.calls
    )
    return edges, result.metadata


PYTHON_SOURCE = """\
import os

setup_logging()


def helper(x):
    return len(x)


class Service:
    def run(self, items):
        self.prepare()
        result = helper(items)
        os.path.join("a", "b")

        def inner():
            return compute(result)

        return inner()

    async def fetch(self):
        await self.client.get("/x")


def main():
    svc = Service()
    svc.run([1, 2])
"""
PYTHON_EDGES = [
    ("fetch", "get", 22, None, False),
    ("helper", "len", 7, None, False),
    ("inner", "compute", 17, None, False),
    ("main", "Service", 26, None, False),
    ("main", "run", 27, None, False),
    ("run", "helper", 13, None, False),
    ("run", "inner", 19, None, False),
    ("run", "join", 14, None, False),
    ("run", "prepare", 12, None, False),
]
GO_SOURCE = """\
package main

import "fmt"

var cfg = loadConfig()

type Server struct{}

func (s *Server) Start() {
\ts.listen()
\tfmt.Println("started")
\tgo worker()
}

func helper(n int) int {
\treturn compute(n)
}

func main() {
\ts := &Server{}
\ts.Start()
\tf := func() { cleanup() }
\tf()
\thelper(1)
}
"""
GO_EDGES = [
    ("helper", "compute", 16, None, False),
    ("main", "Start", 21, None, False),
    ("main", "cleanup", 22, None, False),
    ("main", "f", 23, None, False),
    ("main", "helper", 24, None, False),
]
JAVA_SOURCE = """\
package demo;

import java.util.List;

public class App {
    private static final Logger LOG = Logger.getLogger("x");

    public App() {
        init();
    }

    public void run(List<String> items) {
        process(items);
        this.helper();
        System.out.println("done");
        Helper h = new Helper();
        h.assist();
    }

    private void helper() {
        Utils.compute(1);
    }

    static class Inner {
        void go() {
            launch();
        }
    }
}
"""
JAVA_EDGES = [
    ("App", "getLogger", 6, None, False),
    ("App", "init", 9, None, False),
    ("go", "launch", 26, None, False),
    ("helper", "compute", 21, None, False),
    ("run", "Helper", 16, None, False),
    ("run", "assist", 17, None, False),
    ("run", "helper", 14, None, False),
    ("run", "println", 15, None, False),
    ("run", "process", 13, None, False),
]
JAVASCRIPT_SOURCE = """\
import fs from "fs";

setup();

function helper(x) {
  return format(x);
}

class Service {
  run(items) {
    this.prepare();
    const result = helper(items);
    console.log(result);
    return items.map((i) => transform(i));
  }
}

const main = () => {
  const svc = new Service();
  svc.run([1, 2]);
  fs.readFileSync("a");
};

function outer() {
  function inner() {
    return compute();
  }
  return inner();
}
"""
JAVASCRIPT_EDGES = [
    ("helper", "format", 6, None, False),
    ("inner", "compute", 26, None, False),
    ("outer", "inner", 28, None, False),
]
TYPESCRIPT_SOURCE = """\
import { readFileSync } from "fs";

bootstrap();

interface Runner {
  run(items: string[]): void;
}

function helper(x: number): string {
  return format(x);
}

class Service implements Runner {
  run(items: string[]): void {
    this.prepare();
    const result: string = helper(items.length);
    console.log(result);
  }

  private prepare(): void {
    Utils.compute<number>(1);
  }
}

const main = async (): Promise<void> => {
  const svc = new Service();
  svc.run(["a"]);
  await fetchData();
};

function outer() {
  function inner() {
    return compute();
  }
  return inner();
}
"""
TYPESCRIPT_EDGES = [
    ("helper", "format", 10, None, False),
    ("inner", "compute", 33, None, False),
    ("outer", "inner", 35, None, False),
]
CPP_SOURCE = """\
#include <iostream>

int global_value = compute_global();

namespace util {
int helper(int x) {
    return square(x);
}
}

class Widget {
public:
    void draw() {
        this->prepare();
        render();
        util::helper(2);
    }
    void prepare();
};

void Widget::prepare() {
    std::cout << "x";
    std::sort(a, b);
}

int main() {
    Widget w;
    w.draw();
    auto p = new Widget();
    p->draw();
    auto f = [](int v) { return util::helper(v); };
    return f(1);
}
"""
CPP_EDGES = [
    ("draw", "helper", 16, None, False),
    ("draw", "prepare", 14, None, False),
    ("draw", "render", 15, None, False),
    ("helper", "square", 7, None, False),
    ("main", "draw", 28, None, False),
    ("main", "draw", 30, None, False),
    ("main", "f", 32, None, False),
    ("main", "helper", 31, None, False),
]
C_SOURCE = """\
#include <stdio.h>

static int counter = 0;

static int helper(int x) {
    return square(x);
}

struct ops {
    void (*fn)(int);
};

void run(struct ops *o) {
    printf("run\\n");
    o->fn(1);
    helper(2);
}

int main(void) {
    struct ops o;
    run(&o);
    (*o.fn)(3);
    return helper(4);
}
"""
C_EDGES = [
    ("helper", "square", 6, None, False),
    ("main", "helper", 23, None, False),
    ("main", "run", 21, None, False),
    ("run", "fn", 15, None, False),
    ("run", "helper", 16, None, False),
    ("run", "printf", 14, None, False),
]
KOTLIN_SOURCE = """\
package demo

val logger = createLogger()

fun helper(x: Int): Int {
    return square(x)
}

class Service {
    fun run(items: List<String>) {
        prepare()
        this.cleanup()
        val r = helper(items.size)
        println(r)
        items.forEach { process(it) }
    }

    companion object {
        fun create(): Service = Service()
    }
}

fun main() {
    val s = Service()
    s.run(listOf("a"))
    fun inner() = compute()
    inner()
}
"""
KOTLIN_EDGES = [
    ("create", "Service", 19, None, False),
    ("helper", "square", 6, None, False),
    ("inner", "compute", 26, None, False),
    ("main", "Service", 24, None, False),
    ("main", "inner", 27, None, False),
    ("main", "listOf", 25, None, False),
    ("main", "run", 25, None, False),
    ("run", "cleanup", 12, None, False),
    ("run", "forEach", 15, None, False),
    ("run", "helper", 13, None, False),
    ("run", "prepare", 11, None, False),
    ("run", "println", 14, None, False),
    ("run", "process", 15, None, False),
]
CSHARP_SOURCE = """\
using System;

namespace Demo
{
    public class App
    {
        private static readonly Logger Log = Logger.Create();

        public App()
        {
            Init();
        }

        public void Run(string[] items)
        {
            Process(items);
            this.Helper();
            Console.WriteLine("done");
            var h = new Helper();
            h.Assist();
            System.IO.File.ReadAllText("x");
        }

        private void Helper()
        {
            Utils.Compute(1);
            int Local() => Square(2);
            Local();
        }

        public int Value => Compute();
    }
}
"""
CSHARP_EDGES = [
    ("App", "Compute", 31, None, False),
    ("App", "Create", 7, None, False),
    ("App", "Init", 11, None, False),
    ("Helper", "Compute", 26, None, False),
    ("Helper", "Local", 28, None, False),
    ("Helper", "Square", 27, None, False),
    ("Run", "Assist", 20, None, False),
    ("Run", "Helper", 17, None, False),
    ("Run", "Helper", 19, None, False),
    ("Run", "Process", 16, None, False),
    ("Run", "ReadAllText", 21, None, False),
    ("Run", "WriteLine", 18, None, False),
]
RUBY_SOURCE = """\
require "json"

setup_logging

module Util
  def self.helper(x)
    square(x)
  end
end

class Service
  def run(items)
    prepare
    result = Util.helper(items)
    puts result
    items.each { |i| process(i) }
    JSON.parse("{}")
  end

  def prepare
    self.cleanup()
  end
end

def main
  svc = Service.new
  svc.run([1, 2])
end
"""
RUBY_EDGES = [
    ("helper", "square", 7, None, False),
    ("main", "new", 26, None, False),
    ("main", "run", 27, None, False),
    ("prepare", "cleanup", 21, None, False),
    ("run", "each", 16, None, False),
    ("run", "helper", 14, None, False),
    ("run", "parse", 17, None, False),
    ("run", "prepare", 13, None, False),
    ("run", "process", 16, None, False),
    ("run", "puts", 15, None, False),
]
PHP_SOURCE = """\
<?php

namespace App;

bootstrap();

function helper($x) {
    return square($x);
}

class Service {
    public function run($items) {
        $this->prepare();
        $r = helper($items);
        self::log($r);
        Util::compute(1);
        \\App\\Lib\\fmt($r);
        $obj = new Helper();
        $obj->assist();
    }

    private function prepare() {
        strlen("x");
    }
}

$f = function() { return cleanup(); };
"""
PHP_EDGES = [
    ("helper", "square", 8, None, False),
    ("prepare", "strlen", 23, None, False),
    ("run", "Helper", 18, None, False),
    ("run", "assist", 19, None, False),
    ("run", "compute", 16, None, False),
    ("run", "helper", 14, None, False),
    ("run", "log", 15, None, False),
    ("run", "prepare", 13, None, False),
]
SWIFT_SOURCE = """\
import Foundation

let shared = makeShared()

func helper(_ x: Int) -> Int {
    return square(x)
}

class Service {
    func run(items: [Int]) {
        prepare()
        self.cleanup()
        let r = helper(items.count)
        print(r)
        items.forEach { process($0) }
    }

    init() {
        configure()
    }
}

struct Point {
    func distance() -> Double {
        return sqrt(2.0)
    }
}

func main() {
    let s = Service()
    s.run(items: [1])
    func inner() -> Int { return compute() }
    _ = inner()
}
"""
SWIFT_EDGES = [
    ("Service", "configure", 19, None, False),
    ("distance", "sqrt", 25, None, False),
    ("helper", "square", 6, None, False),
    ("inner", "compute", 32, None, False),
    ("main", "Service", 30, None, False),
    ("main", "inner", 33, None, False),
    ("main", "run", 31, None, False),
    ("run", "cleanup", 12, None, False),
    ("run", "forEach", 15, None, False),
    ("run", "helper", 13, None, False),
    ("run", "prepare", 11, None, False),
    ("run", "print", 14, None, False),
    ("run", "process", 15, None, False),
]
RUST_SOURCE = """\
use std::collections::HashMap;

static INIT: i32 = 0;

fn helper(x: i32) -> i32 {
    square(x)
}

struct Server { map: HashMap<String, i32> }

impl Server {
    fn new() -> Self {
        Server { map: HashMap::new() }
    }

    fn start(&self) {
        self.listen();
        let n = helper(1);
        println!("{}", n);
    }
}

fn main() {
    let s = Server::new();
    s.start();
    let v: Vec<i32> = Vec::new();
    v.len();
    let f = |x: i32| compute(x);
    f(2);
}
"""
RUST_EDGES = [
    ("helper", "square", 6, None, False),
    ("main", "compute", 28, None, False),
    ("main", "f", 29, None, False),
    ("main", "len", 27, "Vec", True),
    ("main", "new", 24, "Server", False),
    ("main", "new", 26, "Vec", False),
    ("main", "start", 25, "Server", True),
    ("new", "new", 13, "HashMap", False),
    ("start", "helper", 18, None, False),
    ("start", "listen", 17, "Server", True),
]

CASES = [
    pytest.param(
        python_plugin.PythonPlugin,
        "python",
        PYTHON_SOURCE,
        "sample.py",
        PYTHON_EDGES,
        {"language": "python", "file": "sample.py"},
        id="python",
    ),
    pytest.param(
        go.GoPlugin,
        "go",
        GO_SOURCE,
        "sample.go",
        GO_EDGES,
        {"language": "go", "file": "sample.go"},
        id="go",
    ),
    pytest.param(
        java.JavaPlugin,
        "java",
        JAVA_SOURCE,
        "sample.java",
        JAVA_EDGES,
        {"language": "java", "file": "sample.java"},
        id="java",
    ),
    pytest.param(
        javascript.JavaScriptPlugin,
        "javascript",
        JAVASCRIPT_SOURCE,
        "sample.js",
        JAVASCRIPT_EDGES,
        {"language": "javascript/typescript", "file": "sample.js"},
        id="javascript",
    ),
    pytest.param(
        typescript.TypeScriptPlugin,
        "typescript",
        TYPESCRIPT_SOURCE,
        "sample.ts",
        TYPESCRIPT_EDGES,
        {"language": "typescript", "file": "sample.ts"},
        id="typescript",
    ),
    pytest.param(
        cpp.CppPlugin,
        "cpp",
        CPP_SOURCE,
        "sample.cpp",
        CPP_EDGES,
        {"language": "cpp", "file": "sample.cpp"},
        id="cpp",
    ),
    pytest.param(
        additional.CPlugin,
        "c",
        C_SOURCE,
        "sample.c",
        C_EDGES,
        {"language": "c", "file": "sample.c"},
        id="c",
    ),
    pytest.param(
        additional.KotlinPlugin,
        "kotlin",
        KOTLIN_SOURCE,
        "sample.kt",
        KOTLIN_EDGES,
        {"language": "kotlin", "file": "sample.kt"},
        id="kotlin",
    ),
    pytest.param(
        additional.CSharpPlugin,
        "csharp",
        CSHARP_SOURCE,
        "sample.cs",
        CSHARP_EDGES,
        {"language": "csharp", "file": "sample.cs"},
        id="csharp",
    ),
    pytest.param(
        additional.RubyPlugin,
        "ruby",
        RUBY_SOURCE,
        "sample.rb",
        RUBY_EDGES,
        {"language": "ruby", "file": "sample.rb"},
        id="ruby",
    ),
    pytest.param(
        additional.PhpPlugin,
        "php",
        PHP_SOURCE,
        "sample.php",
        PHP_EDGES,
        {"language": "php", "file": "sample.php"},
        id="php",
    ),
    pytest.param(
        additional.SwiftPlugin,
        "swift",
        SWIFT_SOURCE,
        "sample.swift",
        SWIFT_EDGES,
        {"language": "swift", "file": "sample.swift"},
        id="swift",
    ),
    pytest.param(
        rust.RustPlugin,
        "rust",
        RUST_SOURCE,
        "sample.rs",
        RUST_EDGES,
        {"language": "rust", "file": "sample.rs"},
        id="rust",
    ),
]


@pytest.mark.parametrize("plugin_cls,language,source,file_name,expected_edges,expected_meta", CASES)
def test_call_edges_are_pinned(
    plugin_cls, language, source, file_name, expected_edges, expected_meta
) -> None:
    edges, metadata = _detect(plugin_cls(), language, source, file_name)
    assert edges == expected_edges
    assert metadata == expected_meta


def test_every_call_edge_plugin_is_characterized() -> None:
    """The 12 template plugins plus Rust's override are all covered."""
    covered = {p.values[0] for p in CASES}
    assert covered == {
        python_plugin.PythonPlugin,
        go.GoPlugin,
        java.JavaPlugin,
        javascript.JavaScriptPlugin,
        typescript.TypeScriptPlugin,
        cpp.CppPlugin,
        additional.CPlugin,
        additional.KotlinPlugin,
        additional.CSharpPlugin,
        additional.RubyPlugin,
        additional.PhpPlugin,
        additional.SwiftPlugin,
        rust.RustPlugin,
    }


def test_plugin_without_call_support_returns_no_edges() -> None:
    """Plugins that do not implement call detection report nothing."""
    plugin = additional.ScalaPlugin()
    tree = SimpleNamespace(root_node=SimpleNamespace(children=[], type="compilation_unit"))
    result = plugin.detect_calls_edges(tree, "object A", Path("a.scala"))
    assert result.calls == []
    assert result.metadata["language"] == "scala"
