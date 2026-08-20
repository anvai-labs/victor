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

"""Configuration file language plugins (JSON, YAML, TOML, INI).

These plugins support indexing of configuration files for:
- Key extraction for symbol search
- No tree-sitter queries (config files don't have AST-based relationships)
"""

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    CommentStyle,
    LanguageCapabilities,
    LanguageConfig,
    TreeSitterQueries,
)


class JsonPlugin(BaseLanguagePlugin):
    """JSON configuration file plugin.

    Supports:
    - Key extraction via regex
    - No tree-sitter (not available for JSON)
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="json",
            display_name="JSON",
            aliases=["config-json"],
            extensions=[".json", ".jsonc", ".json5"],
            filenames=["package.json", "tsconfig.json", "composer.json"],
            shebangs=[],
            comment_style=CommentStyle.NONE,  # JSON has no comments
            line_comment="",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"'],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server=None,
            language_server_name=None,
            tree_sitter_language=None,  # No tree-sitter for JSON
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,  # No tree-sitter
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,  # jq, prettier
            supports_linting=True,  # jsonlint
            supports_completion=False,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Config files use regex-based key extraction, not tree-sitter."""
        return TreeSitterQueries()


class YamlPlugin(BaseLanguagePlugin):
    """YAML configuration file plugin.

    Supports:
    - Key extraction via regex
    - No tree-sitter (not available for YAML)
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="yaml",
            display_name="YAML",
            aliases=["config-yaml", "yml"],
            extensions=[".yaml", ".yml"],
            filenames=[
                ".github/workflows/*.yml",
                "docker-compose.yml",
                "docker-compose.yaml",
                "kubernetes.yaml",
                ".gitlab-ci.yml",
            ],
            shebangs=[],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="yaml-language-server",
            language_server_name="YAML Language Server",
            tree_sitter_language=None,  # No tree-sitter for YAML
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,  # yamlfmt, prettier
            supports_linting=True,  # yamllint
            supports_completion=True,  # via yaml-language-server
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Config files use regex-based key extraction, not tree-sitter."""
        return TreeSitterQueries()


class TomlPlugin(BaseLanguagePlugin):
    """TOML configuration file plugin.

    Supports:
    - Key extraction via regex
    - No tree-sitter (limited availability)
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="toml",
            display_name="TOML",
            aliases=["config-toml"],
            extensions=[".toml"],
            filenames=["pyproject.toml", "Cargo.toml", "poetry.toml"],
            shebangs=[],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', "'", '"""', "'''"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="taplo",
            language_server_name="Taplo TOML Language Server",
            tree_sitter_language=None,  # Limited tree-sitter support
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,  # taplo, prettier-plugin-toml
            supports_linting=True,  # taplo
            supports_completion=True,  # via taplo
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Config files use regex-based key extraction, not tree-sitter."""
        return TreeSitterQueries()


class IniPlugin(BaseLanguagePlugin):
    """INI/Properties configuration file plugin.

    Supports:
    - Key extraction via regex
    - No tree-sitter
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="ini",
            display_name="INI",
            aliases=["config-ini", "properties", "config-properties"],
            extensions=[".ini", ".cfg", ".conf", ".properties"],
            filenames=["setup.cfg", "tox.ini", ".editorconfig", ".gitconfig"],
            shebangs=[],
            comment_style=CommentStyle.SEMICOLON,
            line_comment=";",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', "'"],
            indent_size=4,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server=None,
            language_server_name=None,
            tree_sitter_language=None,
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=False,
            supports_linting=False,
            supports_completion=False,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Config files use regex-based key extraction, not tree-sitter."""
        return TreeSitterQueries()


class HoconPlugin(BaseLanguagePlugin):
    """HOCON configuration file plugin.

    Supports:
    - Key extraction via regex
    - No tree-sitter
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="hocon",
            display_name="HOCON",
            aliases=["config-hocon"],
            extensions=[".conf", ".hocon"],
            filenames=["application.conf", "reference.conf"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '"""'],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server=None,
            language_server_name=None,
            tree_sitter_language=None,
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=False,
            supports_linting=False,
            supports_completion=False,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Config files use regex-based key extraction, not tree-sitter."""
        return TreeSitterQueries()
