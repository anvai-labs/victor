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

"""JavaScript language plugin."""

from pathlib import Path
from typing import Optional

from victor_coding.languages.base import (
    BuildSystem,
    CommentStyle,
    DocCommentPattern,
    LanguageConfig,
    Linter,
    QueryPattern,
    TreeSitterQueries,
)
from victor_coding.languages.plugins._ecmascript import EcmaScriptPluginBase


class JavaScriptPlugin(EcmaScriptPluginBase):
    """JavaScript language plugin.

    Supports:
    - Testing: jest, mocha, vitest
    - Formatting: prettier
    - Linting: eslint
    - Bundling: webpack, vite, esbuild
    """

    # Historically shared by the JavaScript and TypeScript call-edge walkers.
    _call_edge_language = "javascript/typescript"

    _vitest_parallel_args = ("--pool", "threads")
    _jest_parallel_args = ("--maxWorkers=auto",)
    _detects_mocha = True

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="javascript",
            display_name="JavaScript",
            aliases=["js", "jsx", "mjs", "cjs"],
            extensions=[".js", ".jsx", ".mjs", ".cjs"],
            filenames=["package.json", ".eslintrc", ".prettierrc"],
            shebangs=["node"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'", "`"],
            indent_size=2,
            use_tabs=False,
            package_managers=["npm", "yarn", "pnpm", "bun"],
            build_systems=["webpack", "vite", "esbuild", "rollup"],
            test_frameworks=["jest", "mocha", "vitest", "ava"],
            language_server="typescript-language-server",
            language_server_name="TypeScript Language Server",
            tree_sitter_language="javascript",
            doc_comment_pattern=DocCommentPattern(
                line_prefixes=["///"],
                block_start="/**",
                block_end="*/",
            ),
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Create tree-sitter queries for JavaScript symbol/call extraction."""
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (identifier) @name)"),
                QueryPattern("function", "(function_declaration name: (identifier) @name)"),
                QueryPattern("function", "(method_definition name: (property_identifier) @name)"),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))",
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))",
                ),
                QueryPattern(
                    "function",
                    "(assignment_expression left: (identifier) @name right: (arrow_function))",
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (member_expression property: (property_identifier) @callee))
                (new_expression constructor: (identifier) @callee)
            """,
            references="""
                (call_expression function: (identifier) @name)
                (call_expression function: (member_expression property: (property_identifier) @name))
                (member_expression property: (property_identifier) @name)
                (new_expression constructor: (identifier) @name)
                (identifier) @name
            """,
            inheritance="""
                (class_declaration
                    name: (identifier) @child
                    (class_heritage (identifier) @base))
            """,
            composition="""
                (class_declaration
                    name: (identifier) @owner
                    body: (class_body
                        (method_definition
                            body: (statement_block
                                (expression_statement
                                    (assignment_expression
                                        left: (member_expression object: (this) property: (property_identifier))
                                        right: (new_expression constructor: (identifier) @type)))))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("method_definition", "name"),
                ("class_declaration", "name"),
            ],
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get eslint linter."""
        return Linter(
            name="eslint",
            command=["npx", "eslint", "."],
            fix_args=["--fix"],
            config_file=".eslintrc",
            output_format="json",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get npm/yarn/pnpm build system."""
        package_json = project_root / "package.json"

        if not package_json.exists():
            return None

        # Detect package manager
        if (project_root / "pnpm-lock.yaml").exists():
            pm = "pnpm"
        elif (project_root / "yarn.lock").exists():
            pm = "yarn"
        else:
            pm = "npm"

        return BuildSystem(
            name=pm,
            build_command=[pm, "run", "build"],
            run_command=[pm, "run", "start"],
            install_command=[pm, "install"],
            manifest_file="package.json",
        )
