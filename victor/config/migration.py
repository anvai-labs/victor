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

"""Configuration migration utility.

This module handles migration from the old configuration format to the new
unified configuration format:

Old format:
  ~/.victor/profiles.yaml - Profile definitions
  ~/.victor/api_keys.yaml - API keys storage

New format:
  ~/.victor/config.yaml - Unified configuration

The migration is reversible and creates backups before making any changes.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Migration Result
# =============================================================================


@dataclass
class MigrationResult:
    """Result of a configuration migration."""

    success: bool
    migrated_accounts: int
    migrated_keys: int
    backup_path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


# =============================================================================
# Config Migrator
# =============================================================================


class ConfigMigrator:
    """Migrate old configuration to new unified format.

    Usage:
        migrator = ConfigMigrator()

        # Check if migration is needed
        if migrator.detect_old_config():
            result = migrator.migrate(prompt=True)

        # Migrate without prompt
        result = migrator.migrate(prompt=False)
    """

    # Old config file paths
    OLD_PROFILES_FILE = Path.home() / ".victor" / "profiles.yaml"
    OLD_API_KEYS_FILE = Path.home() / ".victor" / "api_keys.yaml"

    # New config file path
    NEW_CONFIG_FILE = Path.home() / ".victor" / "config.yaml"

    # Providers with OAuth support
    OAUTH_PROVIDERS: Set[str] = {"openai", "qwen"}

    # Local providers (no API key)
    LOCAL_PROVIDERS: Set[str] = {"ollama", "lmstudio", "vllm"}

    def __init__(
        self,
        victor_dir: Optional[Path] = None,
        dry_run: bool = False,
    ):
        """Initialize migrator.

        Args:
            victor_dir: Victor directory (default: ~/.victor)
            dry_run: If True, don't actually modify files
        """
        if victor_dir:
            self._victor_dir = victor_dir
        else:
            self._victor_dir = Path.home() / ".victor"

        self._dry_run = dry_run
        self._backup_dir = self._victor_dir / "backups"

    @property
    def victor_dir(self) -> Path:
        """Get Victor directory."""
        return self._victor_dir

    def detect_old_config(self) -> bool:
        """Check if old configuration files exist.

        Returns:
            True if old config files exist
        """
        return self.old_profiles_file.exists() or self.old_api_keys_file.exists()

    def detect_new_config(self) -> bool:
        """Check if new configuration file exists.

        Returns:
            True if new config.yaml exists
        """
        return self.new_config_file.exists()

    @property
    def old_profiles_file(self) -> Path:
        """Get old profiles.yaml path."""
        return self._victor_dir / "profiles.yaml"

    @property
    def old_api_keys_file(self) -> Path:
        """Get old api_keys.yaml path."""
        return self._victor_dir / "api_keys.yaml"

    @property
    def new_config_file(self) -> Path:
        """Get new config.yaml path."""
        return self._victor_dir / "config.yaml"

    def load_old_profiles(self) -> Dict[str, Any]:
        """Load old profiles.yaml file.

        Returns:
            Profiles dict, or empty dict if file doesn't exist
        """
        if not self.old_profiles_file.exists():
            return {}

        try:
            with open(self.old_profiles_file, "r") as f:
                data = yaml.safe_load(f) or {}
                logger.debug(f"Loaded old profiles from {self.old_profiles_file}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load old profiles: {e}")
            return {}

    def load_old_api_keys(self) -> Dict[str, Any]:
        """Load old api_keys.yaml file.

        Returns:
            API keys dict, or empty dict if file doesn't exist
        """
        if not self.old_api_keys_file.exists():
            return {}

        try:
            with open(self.old_api_keys_file, "r") as f:
                data = yaml.safe_load(f) or {}
                logger.debug(f"Loaded old API keys from {self.old_api_keys_file}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load old API keys: {e}")
            return {}

    def create_backup(self) -> Path:
        """Create backup of old configuration files.

        Returns:
            Path to backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self._backup_dir / f"migration_{timestamp}"

        if self._dry_run:
            logger.info(f"[DRY RUN] Would create backup at: {backup_path}")
            return backup_path

        backup_path.mkdir(parents=True, exist_ok=True)

        # Backup profiles.yaml
        if self.old_profiles_file.exists():
            shutil.copy2(self.old_profiles_file, backup_path / "profiles.yaml")
            logger.info("Backed up profiles.yaml")

        # Backup api_keys.yaml
        if self.old_api_keys_file.exists():
            shutil.copy2(self.old_api_keys_file, backup_path / "api_keys.yaml")
            logger.info("Backed up api_keys.yaml")

        return backup_path

    def migrate(
        self,
        prompt: bool = True,
        force: bool = False,
    ) -> MigrationResult:
        """Migrate old configuration to new format.

        Args:
            prompt: If True, prompt user before migration (requires UI)
            force: If True, overwrite existing new config

        Returns:
            MigrationResult with details
        """
        result = MigrationResult(
            success=False,
            migrated_accounts=0,
            migrated_keys=0,
        )

        # Check if migration is needed
        if not self.detect_old_config():
            result.add_warning("No old configuration files found")
            result.success = True
            return result

        # Check if new config already exists
        if self.new_config_file.exists() and not force:
            result.add_error(
                f"New config file already exists: {self.new_config_file}. "
                "Use --force to overwrite."
            )
            return result

        # Load old configs
        old_profiles = self.load_old_profiles()
        old_api_keys = self.load_old_api_keys()

        # Create new config
        new_config = self._convert_to_new_format(old_profiles, old_api_keys, result)

        if not new_config:
            result.add_error("Failed to convert configuration")
            return result

        # Create backup
        try:
            backup_path = self.create_backup()
            result.backup_path = backup_path
        except Exception as e:
            result.add_error(f"Failed to create backup: {e}")
            return result

        # Write new config
        if not self._dry_run:
            try:
                self.new_config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.new_config_file, "w") as f:
                    yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

                # Set secure permissions
                self.new_config_file.chmod(0o600)

                logger.info(f"Created new config: {self.new_config_file}")
            except Exception as e:
                result.add_error(f"Failed to write new config: {e}")
                return result
        else:
            logger.info(f"[DRY RUN] Would create: {self.new_config_file}")

        # Update result
        result.success = True
        result.migrated_accounts = len(new_config.get("accounts", {}))
        result.migrated_keys = len(old_api_keys.get("api_keys", {}))

        return result

    def rollback(self, backup_path: Optional[Path] = None) -> bool:
        """Rollback migration from backup.

        Args:
            backup_path: Backup directory (default: latest backup)

        Returns:
            True if rollback successful
        """
        if backup_path is None:
            # Find latest backup
            backups = sorted(self._backup_dir.glob("migration_*"))
            if not backups:
                logger.error("No backup found for rollback")
                return False
            backup_path = backups[-1]

        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False

        try:
            # Restore profiles.yaml
            profiles_backup = backup_path / "profiles.yaml"
            if profiles_backup.exists():
                shutil.copy2(profiles_backup, self.old_profiles_file)
                logger.info("Restored profiles.yaml")

            # Restore api_keys.yaml
            keys_backup = backup_path / "api_keys.yaml"
            if keys_backup.exists():
                shutil.copy2(keys_backup, self.old_api_keys_file)
                logger.info("Restored api_keys.yaml")

            # Remove new config
            if self.new_config_file.exists():
                self.new_config_file.unlink()
                logger.info("Removed new config.yaml")

            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _convert_to_new_format(
        self,
        old_profiles: Dict[str, Any],
        old_api_keys: Dict[str, Any],
        result: MigrationResult,
    ) -> Optional[Dict[str, Any]]:
        """Convert old format to new format.

        Args:
            old_profiles: Old profiles.yaml content
            old_api_keys: Old api_keys.yaml content
            result: MigrationResult to record errors/warnings

        Returns:
            New config dict, or None if conversion failed
        """
        from victor.config.accounts import ProviderAccount, AuthConfig, ConfigDefaults

        new_config: Dict[str, Any] = {"accounts": {}, "defaults": {}}

        # Get profiles and default profile
        profiles = old_profiles.get("profiles", {})
        default_profile = old_profiles.get("default_profile", "default")

        new_config["defaults"]["account"] = default_profile

        # Get API keys
        api_keys = old_api_keys.get("api_keys", {})

        # Track migrated providers
        migrated_providers: Set[str] = set()

        # Migrate each profile as an account
        for profile_name, profile_config in profiles.items():
            try:
                provider = profile_config.get("provider")
                model = profile_config.get("model")

                if not provider:
                    result.add_warning(f"Profile '{profile_name}' missing provider, skipping")
                    continue

                # Determine auth method
                auth_method = "api_key"
                if provider in self.LOCAL_PROVIDERS:
                    auth_method = "none"
                elif provider in self.OAUTH_PROVIDERS:
                    # Check if OAuth is configured
                    if profile_config.get("auth_mode") == "oauth":
                        auth_method = "oauth"

                # Determine auth source
                # If key is in file, use "file", otherwise "keyring"
                auth_source = "keyring"
                if provider in api_keys:
                    auth_source = "file"

                # Create auth config
                auth_config = AuthConfig(method=auth_method, source=auth_source)

                # Create account
                account = ProviderAccount(
                    name=profile_name,
                    provider=provider,
                    model=model or "default",
                    auth=auth_config,
                    endpoint=profile_config.get("base_url"),
                    temperature=profile_config.get("temperature"),
                    max_tokens=profile_config.get("max_tokens"),
                    tags=["migrated"],
                )

                # Add to config
                new_config["accounts"][profile_name] = account.to_dict()
                migrated_providers.add(provider)

            except Exception as e:
                result.add_error(f"Failed to migrate profile '{profile_name}': {e}")

        # Migrate any API keys that don't have profiles
        for provider, api_key in api_keys.items():
            if provider not in migrated_providers:
                try:
                    # Determine auth method
                    auth_method = "none" if provider in self.LOCAL_PROVIDERS else "api_key"

                    # Create account for this provider
                    account = ProviderAccount(
                        name=provider,
                        provider=provider,
                        model="default",
                        auth=AuthConfig(method=auth_method, source="file", value=api_key),
                        tags=["migrated", "api-key-only"],
                    )

                    new_config["accounts"][provider] = account.to_dict()

                except Exception as e:
                    result.add_error(f"Failed to migrate API key for '{provider}': {e}")

        return new_config


# =============================================================================
# Convenience Functions
# =============================================================================


def check_migration_needed() -> bool:
    """Check if migration is needed.

    Returns:
        True if old config exists and new config doesn't
    """
    migrator = ConfigMigrator()
    return migrator.detect_old_config() and not migrator.detect_new_config()


def run_migration(
    prompt: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> MigrationResult:
    """Run configuration migration.

    Args:
        prompt: If True, prompt user before migration
        force: If True, overwrite existing new config
        dry_run: If True, don't actually modify files

    Returns:
        MigrationResult with details
    """
    migrator = ConfigMigrator(dry_run=dry_run)
    return migrator.migrate(prompt=prompt, force=force)


def rollback_migration(backup_path: Optional[Path] = None) -> bool:
    """Rollback migration from backup.

    Args:
        backup_path: Backup directory (default: latest backup)

    Returns:
        True if rollback successful
    """
    migrator = ConfigMigrator()
    return migrator.rollback(backup_path)
