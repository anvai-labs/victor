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

"""Language plugin registry.

Manages registration and discovery of language plugins,
providing a central point for language detection and routing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

from victor_coding.languages.base import LanguagePlugin

logger = logging.getLogger(__name__)

# Type alias for plugin factory
PluginFactory = Callable[[], LanguagePlugin]


class LanguageRegistry:
    """Registry for language plugins.

    Provides:
    - Plugin registration by name/extension
    - Language detection from files and content
    - Plugin discovery and lazy loading
    """

    def __init__(self):
        """Initialize empty registry."""
        self._plugins: Dict[str, PluginFactory] = {}
        self._instances: Dict[str, LanguagePlugin] = {}
        self._extension_map: Dict[str, str] = {}  # .py -> python
        self._alias_map: Dict[str, str] = {}  # py -> python

    def register(
        self,
        name: str,
        plugin: Union[Type[LanguagePlugin], PluginFactory],
        extensions: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a language plugin.

        Args:
            name: Canonical language name
            plugin: Plugin class or factory function
            extensions: File extensions (auto-detected from config if None)
            aliases: Alternative names for the language
        """
        name = name.lower()

        # Convert class to factory if needed
        if isinstance(plugin, type):
            factory: PluginFactory = plugin  # type: ignore
        else:
            factory = plugin

        self._plugins[name] = factory
        logger.debug(f"Registered language plugin: {name}")

        # Create instance to get config
        instance = self._get_or_create_instance(name)

        # Register extensions
        exts = extensions or instance.config.extensions
        for ext in exts:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = "." + ext
            self._extension_map[ext] = name

        # Register aliases
        all_aliases = (aliases or []) + instance.config.aliases
        for alias in all_aliases:
            self._alias_map[alias.lower()] = name

    def unregister(self, name: str) -> None:
        """Unregister a language plugin.

        Args:
            name: Language name to unregister
        """
        name = self._resolve_name(name)

        if name in self._plugins:
            # Get config for cleanup
            if name in self._instances:
                instance = self._instances[name]
                # Remove extensions
                for ext in instance.config.extensions:
                    self._extension_map.pop(ext.lower(), None)
                # Remove aliases
                for alias in instance.config.aliases:
                    self._alias_map.pop(alias.lower(), None)
                del self._instances[name]

            del self._plugins[name]
            logger.info(f"Unregistered language plugin: {name}")

    def get(self, name: str) -> LanguagePlugin:
        """Get a language plugin by name.

        Args:
            name: Language name or alias

        Returns:
            Language plugin instance

        Raises:
            KeyError: If language not registered
        """
        name = self._resolve_name(name)

        if name not in self._plugins:
            available = ", ".join(sorted(self._plugins.keys()))
            raise KeyError(f"Language '{name}' not registered. Available: {available}")

        return self._get_or_create_instance(name)

    def detect_language(self, path: Path) -> Optional[str]:
        """Detect language from a file path.

        Args:
            path: File path to check

        Returns:
            Language name or None if not detected
        """
        # Check extension
        ext = path.suffix.lower()
        if ext in self._extension_map:
            return self._extension_map[ext]

        # Check with each plugin
        for name in self._plugins:
            plugin = self._get_or_create_instance(name)
            if plugin.detect_from_file(path):
                return name

        return None

    def detect_from_content(self, content: str, filename: Optional[str] = None) -> Optional[str]:
        """Detect language from content.

        Args:
            content: File content
            filename: Optional filename hint

        Returns:
            Language name or None if not detected
        """
        # Check filename hint first
        if filename:
            path = Path(filename)
            ext = path.suffix.lower()
            if ext in self._extension_map:
                return self._extension_map[ext]

        # Ask each plugin for confidence
        best_match: Optional[str] = None
        best_score = 0.0

        for name in self._plugins:
            plugin = self._get_or_create_instance(name)
            score = plugin.detect_from_content(content, filename)
            if score > best_score:
                best_score = score
                best_match = name

        # Require minimum confidence
        if best_score >= 0.3:
            return best_match

        return None

    def has(self, name: str) -> bool:
        """Check if a language is registered.

        Args:
            name: Language name or alias

        Returns:
            True if registered
        """
        try:
            self._resolve_name(name)
            return True
        except KeyError:
            return False

    def list_languages(self) -> List[str]:
        """List all registered language names.

        Returns:
            Sorted list of language names
        """
        return sorted(self._plugins.keys())

    def get_extensions(self, name: str) -> List[str]:
        """Get file extensions for a language.

        Args:
            name: Language name

        Returns:
            List of extensions (with dots)
        """
        name = self._resolve_name(name)
        plugin = self._get_or_create_instance(name)
        return plugin.config.extensions

    def _resolve_name(self, name: str) -> str:
        """Resolve alias to canonical name."""
        name = name.lower()

        # Check alias map
        if name in self._alias_map:
            return self._alias_map[name]

        # Check if it's already canonical
        if name in self._plugins:
            return name

        raise KeyError(f"Unknown language: {name}")

    def _get_or_create_instance(self, name: str) -> LanguagePlugin:
        """Get or create plugin instance."""
        if name not in self._instances:
            factory = self._plugins[name]
            self._instances[name] = factory()
        return self._instances[name]

    def discover_plugins(self) -> int:
        """Auto-discover and register built-in plugins.

        Returns:
            Number of plugins registered
        """
        from victor_coding.languages.plugins import (
            # Core languages
            PythonPlugin,
            JavaScriptPlugin,
            TypeScriptPlugin,
            RustPlugin,
            GoPlugin,
            JavaPlugin,
            CppPlugin,
            # Config files
            JsonPlugin,
            YamlPlugin,
            TomlPlugin,
            IniPlugin,
            HoconPlugin,
            # Additional languages
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

        plugins = [
            # Core languages
            ("python", PythonPlugin),
            ("javascript", JavaScriptPlugin),
            ("typescript", TypeScriptPlugin),
            ("rust", RustPlugin),
            ("go", GoPlugin),
            ("java", JavaPlugin),
            ("cpp", CppPlugin),
            # Config files
            ("json", JsonPlugin),
            ("yaml", YamlPlugin),
            ("toml", TomlPlugin),
            ("ini", IniPlugin),
            ("hocon", HoconPlugin),
            # Additional languages
            ("c", CPlugin),
            ("kotlin", KotlinPlugin),
            ("csharp", CSharpPlugin),
            ("ruby", RubyPlugin),
            ("php", PhpPlugin),
            ("swift", SwiftPlugin),
            ("scala", ScalaPlugin),
            ("bash", BashPlugin),
            ("sql", SqlPlugin),
            ("html", HtmlPlugin),
            ("css", CssPlugin),
            ("lua", LuaPlugin),
            ("elixir", ElixirPlugin),
            ("haskell", HaskellPlugin),
            ("r", RPlugin),
            ("markdown", MarkdownPlugin),
            ("xml", XmlPlugin),
            # Tier-2 coding languages (post-TSA plugin sweep)
            ("zig", ZigPlugin),
            ("julia", JuliaPlugin),
            ("ocaml", OcamlPlugin),
            ("solidity", SolidityPlugin),
            ("perl", PerlPlugin),
            ("objc", ObjcPlugin),
            # Tier-3 build / scripting / schema languages
            ("make", MakePlugin),
            ("cmake", CmakePlugin),
            ("graphql", GraphqlPlugin),
            ("groovy", GroovyPlugin),
            ("hcl", HclPlugin),
            # Tier-4 hardware-description / shader languages
            ("vhdl", VhdlPlugin),
            ("verilog", VerilogPlugin),
            ("glsl", GlslPlugin),
        ]

        count = 0
        for name, plugin_class in plugins:
            try:
                self.register(name, plugin_class)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register {name} plugin: {e}")

        logger.info(f"Discovered {count} language plugins")
        return count


# Global registry instance
_global_registry: Optional[LanguageRegistry] = None


def get_language_registry() -> LanguageRegistry:
    """Get the global language registry.

    Returns:
        Global registry instance (creates if needed)
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = LanguageRegistry()
    return _global_registry
