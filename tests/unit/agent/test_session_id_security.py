from __future__ import annotations

"""Tests for Phase 1 security improvements: session secret auto-generation
and session ID entropy."""

from pathlib import Path

from pydantic import SecretStr

from victor.agent.session_id import (
    generate_session_id,
    parse_session_id,
    validate_session_id,
)

# ---------------------------------------------------------------------------
# Phase 1A: server_session_secret auto-generation
# ---------------------------------------------------------------------------


class TestServerSessionSecretAutoGeneration:
    """Verify that Settings auto-generates a session secret when not provided."""

    def test_settings_creates_non_none_secret(self):
        """Settings() should always produce a non-None server_session_secret."""
        from victor.config.settings import Settings

        settings = Settings()
        assert settings.server.server_session_secret is not None

    def test_explicit_secret_is_preserved(self):
        """An explicitly supplied SecretStr must not be overwritten."""
        from victor.config.groups.server_config import ServerSettings

        server = ServerSettings(server_session_secret=SecretStr("custom"))
        assert server.server_session_secret.get_secret_value() == "custom"

    def test_generated_secret_has_sufficient_length(self):
        """Auto-generated secret should be at least 32 characters."""
        from victor.config.settings import Settings

        settings = Settings()
        secret_value = settings.server.server_session_secret.get_secret_value()
        assert len(secret_value) >= 32


# ---------------------------------------------------------------------------
# Phase 1B: session ID entropy
# ---------------------------------------------------------------------------


class TestSessionIdEntropy:
    """Verify that generated session IDs contain a random entropy suffix."""

    def test_new_session_ids_have_three_components(self):
        """Generated session IDs must have 3 dash-separated parts."""
        sid = generate_session_id(project_root=Path("/tmp/testproj"))
        parts = sid.split("-", maxsplit=2)
        assert len(parts) == 3, f"Expected 3 components, got {len(parts)}: {sid}"

    def test_thousand_generated_ids_are_unique(self):
        """1000 generated IDs must all be distinct (entropy prevents collisions)."""
        ids = set()
        root = Path("/tmp/testproj")
        for _ in range(1000):
            ids.add(generate_session_id(project_root=root))
        assert len(ids) == 1000

    def test_legacy_two_part_ids_still_parse(self):
        """Old 2-part session IDs (no entropy) must remain parseable."""
        legacy_id = "myproj-8M0KX"
        parsed = parse_session_id(legacy_id)
        assert parsed["project_root"] == "myproj"
        assert parsed["base62_timestamp"] == "8M0KX"
        assert parsed["entropy"] == ""

    def test_new_three_part_ids_parse_correctly(self):
        """New 3-part session IDs must parse with entropy field populated."""
        sid = generate_session_id(project_root=Path("/tmp/testproj"))
        parsed = parse_session_id(sid)
        assert parsed["project_root"]
        assert parsed["base62_timestamp"]
        assert parsed["entropy"] != ""
        assert isinstance(parsed["timestamp_ms"], int)

    def test_validate_session_id_accepts_new_format(self):
        """validate_session_id must accept the new 3-part format."""
        sid = generate_session_id(project_root=Path("/tmp/testproj"))
        assert validate_session_id(sid) is True

    def test_validate_session_id_accepts_legacy_format(self):
        """validate_session_id must still accept legacy 2-part format."""
        assert validate_session_id("myproj-8M0KX") is True

    def test_validate_session_id_rejects_invalid(self):
        """validate_session_id must reject single-segment strings."""
        assert validate_session_id("nope") is False
