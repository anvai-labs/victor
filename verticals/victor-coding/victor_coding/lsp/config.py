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

"""LSP server configuration and defaults.

Defines configurations for common language servers that can be used with Victor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LSPServerConfig:
    """Configuration for a language server."""

    name: str  # Human-readable name
    language_id: str  # LSP language identifier
    file_extensions: List[str]  # File extensions this server handles
    command: List[str]  # Command to start the server
    args: List[str] = field(default_factory=list)  # Additional arguments
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    root_patterns: List[str] = field(default_factory=list)  # Files indicating project root
    install_command: Optional[str] = None  # How to install the server
    install_check: Optional[str] = None  # Command to check if installed


# Pre-configured language servers
LANGUAGE_SERVERS: Dict[str, LSPServerConfig] = {
    "python": LSPServerConfig(
        name="Pyright",
        language_id="python",
        file_extensions=[".py", ".pyi"],
        command=["pyright-langserver", "--stdio"],
        root_patterns=["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
        settings={
            "python": {
                "analysis": {
                    "autoSearchPaths": True,
                    "useLibraryCodeForTypes": True,
                    "diagnosticMode": "workspace",
                }
            }
        },
        install_command="pip install pyright",
        install_check="pyright-langserver --version",
    ),
    "python-pylsp": LSPServerConfig(
        name="Python LSP Server",
        language_id="python",
        file_extensions=[".py", ".pyi"],
        command=["pylsp"],
        root_patterns=["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"],
        settings={
            "pylsp": {
                "plugins": {
                    "pycodestyle": {"enabled": True},
                    "pyflakes": {"enabled": True},
                    "pylint": {"enabled": False},
                }
            }
        },
        install_command="pip install python-lsp-server",
        install_check="pylsp --version",
    ),
    "typescript": LSPServerConfig(
        name="TypeScript Language Server",
        language_id="typescript",
        file_extensions=[".ts", ".tsx", ".js", ".jsx"],
        command=["typescript-language-server", "--stdio"],
        root_patterns=["package.json", "tsconfig.json", "jsconfig.json"],
        settings={
            "typescript": {
                "inlayHints": {
                    "includeInlayParameterNameHints": "all",
                    "includeInlayFunctionParameterTypeHints": True,
                }
            }
        },
        install_command="npm install -g typescript-language-server typescript",
        install_check="typescript-language-server --version",
    ),
    "rust": LSPServerConfig(
        name="rust-analyzer",
        language_id="rust",
        file_extensions=[".rs"],
        command=["rust-analyzer"],
        root_patterns=["Cargo.toml"],
        settings={
            "rust-analyzer": {
                "checkOnSave": {"command": "clippy"},
                "inlayHints": {"enable": True},
            }
        },
        install_command="rustup component add rust-analyzer",
        install_check="rust-analyzer --version",
    ),
    "go": LSPServerConfig(
        name="gopls",
        language_id="go",
        file_extensions=[".go"],
        command=["gopls"],
        root_patterns=["go.mod", "go.sum"],
        settings={
            "gopls": {
                "staticcheck": True,
                "usePlaceholders": True,
            }
        },
        install_command="go install golang.org/x/tools/gopls@latest",
        install_check="gopls version",
    ),
    "java": LSPServerConfig(
        name="Eclipse JDT Language Server",
        language_id="java",
        file_extensions=[".java"],
        command=["jdtls"],
        root_patterns=["pom.xml", "build.gradle", ".project"],
        install_command="brew install jdtls",
        install_check="jdtls --version",
    ),
    "c": LSPServerConfig(
        name="clangd",
        language_id="c",
        file_extensions=[".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"],
        command=["clangd"],
        root_patterns=["compile_commands.json", "CMakeLists.txt", "Makefile"],
        settings={
            "clangd": {
                "arguments": ["--background-index", "--clang-tidy"],
            }
        },
        install_command="brew install llvm",
        install_check="clangd --version",
    ),
    "lua": LSPServerConfig(
        name="lua-language-server",
        language_id="lua",
        file_extensions=[".lua"],
        command=["lua-language-server"],
        root_patterns=[".luarc.json", ".luarc.jsonc"],
        install_command="brew install lua-language-server",
        install_check="lua-language-server --version",
    ),
    "yaml": LSPServerConfig(
        name="YAML Language Server",
        language_id="yaml",
        file_extensions=[".yaml", ".yml"],
        command=["yaml-language-server", "--stdio"],
        root_patterns=[],
        settings={
            "yaml": {
                "validate": True,
                "format": {"enable": True},
            }
        },
        install_command="npm install -g yaml-language-server",
        install_check="yaml-language-server --version",
    ),
    "json": LSPServerConfig(
        name="VSCode JSON Language Server",
        language_id="json",
        file_extensions=[".json", ".jsonc"],
        command=["vscode-json-language-server", "--stdio"],
        root_patterns=[],
        install_command="npm install -g vscode-langservers-extracted",
        install_check="vscode-json-language-server --version",
    ),
    "html": LSPServerConfig(
        name="VSCode HTML Language Server",
        language_id="html",
        file_extensions=[".html", ".htm"],
        command=["vscode-html-language-server", "--stdio"],
        root_patterns=[],
        install_command="npm install -g vscode-langservers-extracted",
        install_check="vscode-html-language-server --version",
    ),
    "css": LSPServerConfig(
        name="VSCode CSS Language Server",
        language_id="css",
        file_extensions=[".css", ".scss", ".less"],
        command=["vscode-css-language-server", "--stdio"],
        root_patterns=[],
        install_command="npm install -g vscode-langservers-extracted",
        install_check="vscode-css-language-server --version",
    ),
    "bash": LSPServerConfig(
        name="Bash Language Server",
        language_id="shellscript",
        file_extensions=[".sh", ".bash", ".zsh"],
        command=["bash-language-server", "start"],
        root_patterns=[],
        install_command="npm install -g bash-language-server",
        install_check="bash-language-server --version",
    ),
    "dockerfile": LSPServerConfig(
        name="Docker Language Server",
        language_id="dockerfile",
        file_extensions=["Dockerfile"],
        command=["docker-langserver", "--stdio"],
        root_patterns=["Dockerfile", "docker-compose.yml"],
        install_command="npm install -g dockerfile-language-server-nodejs",
        install_check="docker-langserver --version",
    ),
    "sql": LSPServerConfig(
        name="SQL Language Server",
        language_id="sql",
        file_extensions=[".sql"],
        command=["sql-language-server", "up", "--method", "stdio"],
        root_patterns=[],
        install_command="npm install -g sql-language-server",
        install_check="sql-language-server --version",
    ),
    "markdown": LSPServerConfig(
        name="Marksman",
        language_id="markdown",
        file_extensions=[".md", ".markdown"],
        command=["marksman", "server"],
        root_patterns=[],
        install_command="brew install marksman",
        install_check="marksman --version",
    ),
    "toml": LSPServerConfig(
        name="Taplo",
        language_id="toml",
        file_extensions=[".toml"],
        command=["taplo", "lsp", "stdio"],
        root_patterns=["Cargo.toml", "pyproject.toml"],
        install_command="cargo install taplo-cli",
        install_check="taplo --version",
    ),
}


def get_server_for_file(file_path: str) -> Optional[LSPServerConfig]:
    """Get the appropriate language server config for a file.

    Args:
        file_path: Path to the file

    Returns:
        LSPServerConfig if a matching server is found
    """
    from pathlib import Path

    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name

    # Check by extension
    for config in LANGUAGE_SERVERS.values():
        if ext in config.file_extensions:
            return config
        # Check for exact filename matches (like Dockerfile)
        if name in config.file_extensions:
            return config

    return None


def get_language_id(file_path: str) -> Optional[str]:
    """Get the LSP language ID for a file.

    Args:
        file_path: Path to the file

    Returns:
        Language ID string or None
    """
    config = get_server_for_file(file_path)
    return config.language_id if config else None
