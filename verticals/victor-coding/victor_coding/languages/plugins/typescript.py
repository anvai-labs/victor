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

"""TypeScript language plugin."""

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


class TypeScriptPlugin(EcmaScriptPluginBase):
    """TypeScript language plugin.

    Supports:
    - Testing: jest, vitest
    - Formatting: prettier
    - Linting: eslint, typescript-eslint
    - Type checking: tsc
    """

    _supports_type_checking = True
    _test_file_pattern = "*.test.ts"
    _method_node_types = ("method_definition", "method_signature")
    _extra_function_types = ("method_signature",)  # TypeScript specific
    _class_types = ("class_declaration", "interface_declaration")
    _scope_body_types = ("class_body", "interface_body")

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="typescript",
            display_name="TypeScript",
            aliases=["ts", "tsx"],
            extensions=[".ts", ".tsx", ".mts", ".cts"],
            filenames=["tsconfig.json"],
            shebangs=["ts-node", "tsx"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'", "`"],
            indent_size=2,
            use_tabs=False,
            package_managers=["npm", "yarn", "pnpm", "bun"],
            build_systems=["tsc", "webpack", "vite", "esbuild"],
            test_frameworks=["jest", "vitest", "mocha"],
            language_server="typescript-language-server",
            language_server_name="TypeScript Language Server",
            tree_sitter_language="typescript",
            doc_comment_pattern=DocCommentPattern(
                line_prefixes=["///"],
                block_start="/**",
                block_end="*/",
            ),
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Create tree-sitter queries for TypeScript symbol/call extraction.

        The @def capture is used for end_line (function body boundaries).
        The @name capture is used for the symbol name.
        """
        return TreeSitterQueries(
            symbols=[
                # tree-sitter-typescript: class_declaration name is type_identifier,
                # NOT identifier. function_declaration still uses identifier.
                QueryPattern("class", "(class_declaration name: (type_identifier) @name) @def"),
                QueryPattern("function", "(function_declaration name: (identifier) @name) @def"),
                QueryPattern(
                    "function",
                    "(method_signature name: (property_identifier) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(method_definition name: (property_identifier) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def",
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression))) @def",
                ),
                QueryPattern(
                    "function",
                    "(assignment_expression left: (identifier) @name right: (arrow_function)) @def",
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
                    name: (type_identifier) @child
                    (class_heritage
                        (extends_clause (identifier) @base)))
            """,
            implements="""
                (class_declaration
                    name: (type_identifier) @child
                    (class_heritage
                        (implements_clause (type_identifier) @interface)))
            """,
            composition="""
                (class_declaration
                    name: (type_identifier) @owner
                    body: (class_body
                        (public_field_definition
                            type: (type_annotation (type_identifier) @type))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("method_definition", "name"),
                ("method_signature", "name"),
                ("class_declaration", "name"),
            ],
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get eslint linter with TypeScript support."""
        return Linter(
            name="eslint",
            command=["npx", "eslint", "--ext", ".ts,.tsx", "."],
            fix_args=["--fix"],
            config_file=".eslintrc",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get TypeScript build system."""
        tsconfig = project_root / "tsconfig.json"

        if not tsconfig.exists():
            return None

        return BuildSystem(
            name="tsc",
            build_command=["npx", "tsc"],
            run_command=["npx", "ts-node"],
            clean_command=["rm", "-rf", "dist"],
            manifest_file="tsconfig.json",
        )
