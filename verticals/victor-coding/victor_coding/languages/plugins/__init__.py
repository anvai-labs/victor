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

"""Built-in language plugins.

Provides language support for:
- Python (pytest, black, ruff)
- JavaScript (jest, prettier, eslint)
- TypeScript (jest, prettier, eslint)
- Rust (cargo test, rustfmt, clippy)
- Go (go test, gofmt, golint)
- Java (junit, google-java-format, checkstyle)
- C/C++ (gtest, clang-format, clang-tidy)
- Config files (JSON, YAML, TOML, INI, HOCON)
- Additional: Kotlin, C#, Ruby, PHP, Swift, Scala, Bash, SQL, HTML, CSS, Lua, Elixir, Haskell, R, Markdown, XML
"""

from victor_coding.languages.plugins.python import PythonPlugin
from victor_coding.languages.plugins.javascript import JavaScriptPlugin
from victor_coding.languages.plugins.typescript import TypeScriptPlugin
from victor_coding.languages.plugins.rust import RustPlugin
from victor_coding.languages.plugins.go import GoPlugin
from victor_coding.languages.plugins.java import JavaPlugin
from victor_coding.languages.plugins.cpp import CppPlugin
from victor_coding.languages.plugins.config import (
    JsonPlugin,
    YamlPlugin,
    TomlPlugin,
    IniPlugin,
    HoconPlugin,
)
from victor_coding.languages.plugins.additional import (
    CPlugin,
    KotlinPlugin,
    CSharpPlugin,
    RubyPlugin,
    PhpPlugin,
    SwiftPlugin,
    ScalaPlugin,
    BashPlugin,
    SqlPlugin,
    HtmlPlugin,
    CssPlugin,
    LuaPlugin,
    ElixirPlugin,
    HaskellPlugin,
    RPlugin,
    MarkdownPlugin,
    XmlPlugin,
    # Tier-2 coding languages (post-TSA plugin sweep)
    ZigPlugin,
    JuliaPlugin,
    OcamlPlugin,
    SolidityPlugin,
    PerlPlugin,
    ObjcPlugin,
    # Tier-3 build / scripting / schema languages
    MakePlugin,
    CmakePlugin,
    GraphqlPlugin,
    GroovyPlugin,
    HclPlugin,
    # Tier-4 hardware-description / shader languages
    VhdlPlugin,
    VerilogPlugin,
    GlslPlugin,
)

__all__ = [
    # Core languages
    "PythonPlugin",
    "JavaScriptPlugin",
    "TypeScriptPlugin",
    "RustPlugin",
    "GoPlugin",
    "JavaPlugin",
    "CppPlugin",
    # Config files
    "JsonPlugin",
    "YamlPlugin",
    "TomlPlugin",
    "IniPlugin",
    "HoconPlugin",
    # Additional languages
    "CPlugin",
    "KotlinPlugin",
    "CSharpPlugin",
    "RubyPlugin",
    "PhpPlugin",
    "SwiftPlugin",
    "ScalaPlugin",
    "BashPlugin",
    "SqlPlugin",
    "HtmlPlugin",
    "CssPlugin",
    "LuaPlugin",
    "ElixirPlugin",
    "HaskellPlugin",
    "RPlugin",
    "MarkdownPlugin",
    "XmlPlugin",
    # Tier-2 coding languages
    "ZigPlugin",
    "JuliaPlugin",
    "OcamlPlugin",
    "SolidityPlugin",
    "PerlPlugin",
    "ObjcPlugin",
    # Tier-3 build / scripting / schema languages
    "MakePlugin",
    "CmakePlugin",
    "GraphqlPlugin",
    "GroovyPlugin",
    "HclPlugin",
    # Tier-4 hardware-description / shader languages
    "VhdlPlugin",
    "VerilogPlugin",
    "GlslPlugin",
]
