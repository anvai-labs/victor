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

"""Unified credential management for Victor services.

Provides a layered credential resolution system with multiple backends:
1. Environment variables (highest priority)
2. OS Keyring (macOS Keychain, Windows Credential Manager, Linux libsecret)
3. Config file ~/.victor/credentials (encrypted)
4. Cloud IAM roles (EC2, Lambda, EKS pod identity)
5. Cloud Secret Managers (for app secrets, not bootstrap creds)

CLI Usage:
    # Store credentials
    victor keys set aws --profile default
    victor keys set docker --registry docker.io
    victor keys set postgres --alias mydb

    # List stored credentials
    victor keys list

    # Get credentials
    victor keys get aws --profile default

    # Delete credentials
    victor keys delete aws --profile default

    # Sync to cloud secret manager
    victor keys sync-to aws-secretsmanager --secret-name victor-creds

Example:
    from victor.workflows.services.credentials import CredentialManager

    creds = CredentialManager()

    # Get AWS credentials
    aws = creds.get_aws("default")
    session = boto3.Session(
        aws_access_key_id=aws.access_key_id,
        aws_secret_access_key=aws.secret_access_key,
    )

    # Get database credentials
    db = creds.get_database("mydb")
    conn = psycopg2.connect(db.connection_string)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Type, Union

logger = logging.getLogger(__name__)

# Optional imports
try:
    import keyring

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    keyring = None  # type: ignore

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# TOTP for authenticator apps
try:
    import pyotp

    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    pyotp = None  # type: ignore

# Touch ID / biometric auth on macOS
try:
    import LocalAuthentication  # pyobjc on macOS

    BIOMETRIC_AVAILABLE = platform.system() == "Darwin"
except ImportError:
    BIOMETRIC_AVAILABLE = False
    LocalAuthentication = None  # type: ignore


# =============================================================================
# MFA Types and Configuration
# =============================================================================


class MFAMethod(Enum):
    """Multi-factor authentication methods."""

    NONE = "none"  # No MFA required
    TOTP = "totp"  # Time-based OTP (Google Authenticator, Authy)
    BIOMETRIC = "biometric"  # Touch ID, Windows Hello, fingerprint
    PASSKEY = "passkey"  # FIDO2/WebAuthn hardware keys
    DEVICE_PIN = "device_pin"  # Device-specific PIN


@dataclass
class MFAConfig:
    """MFA configuration for credential access.

    Attributes:
        method: MFA method to use
        totp_secret: Base32-encoded TOTP secret (for TOTP method)
        required_for_read: Require MFA to read credential
        required_for_write: Require MFA to write credential
        cache_duration: How long to cache MFA verification (seconds)
        fallback_method: Fallback if primary method unavailable
    """

    method: MFAMethod = MFAMethod.NONE
    totp_secret: Optional[str] = None
    required_for_read: bool = False
    required_for_write: bool = True
    cache_duration: int = 300  # 5 minutes
    fallback_method: Optional[MFAMethod] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "totp_secret": self.totp_secret,
            "required_for_read": self.required_for_read,
            "required_for_write": self.required_for_write,
            "cache_duration": self.cache_duration,
            "fallback_method": (self.fallback_method.value if self.fallback_method else None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MFAConfig":
        fallback = data.get("fallback_method")
        return cls(
            method=MFAMethod(data.get("method", "none")),
            totp_secret=data.get("totp_secret"),
            required_for_read=data.get("required_for_read", False),
            required_for_write=data.get("required_for_write", True),
            cache_duration=data.get("cache_duration", 300),
            fallback_method=MFAMethod(fallback) if fallback else None,
        )


class MFAVerifier:
    """Multi-factor authentication verifier.

    Supports:
    - TOTP (Google Authenticator, Authy, etc.)
    - Biometric (Touch ID on macOS, Windows Hello)
    - Passkeys (FIDO2/WebAuthn)
    - Device PIN fallback

    Example:
        verifier = MFAVerifier()

        # Setup TOTP
        secret = verifier.setup_totp("my-credential")
        print(f"Add to authenticator: {verifier.get_totp_uri(secret, 'victor')}")

        # Verify TOTP
        if verifier.verify_totp(secret, "123456"):
            print("Verified!")

        # Verify biometric
        if verifier.verify_biometric("Access Victor credentials"):
            print("Touch ID verified!")
    """

    def __init__(self):
        self._verification_cache: Dict[str, datetime] = {}

    def is_cached(self, key: str, cache_duration: int = 300) -> bool:
        """Check if MFA verification is cached."""
        if key not in self._verification_cache:
            return False

        cached_at = self._verification_cache[key]
        if datetime.now(timezone.utc) - cached_at > timedelta(seconds=cache_duration):
            del self._verification_cache[key]
            return False

        return True

    def cache_verification(self, key: str) -> None:
        """Cache successful MFA verification."""
        self._verification_cache[key] = datetime.now(timezone.utc)

    # TOTP (Authenticator apps)
    def setup_totp(self, credential_name: str) -> str:
        """Generate new TOTP secret.

        Returns:
            Base32-encoded secret to store and show to user
        """
        if not TOTP_AVAILABLE:
            raise ImportError("pyotp not available. Install with: pip install pyotp")

        return pyotp.random_base32()

    def get_totp_uri(
        self,
        secret: str,
        issuer: str = "Victor",
        account: Optional[str] = None,
    ) -> str:
        """Get TOTP provisioning URI for QR code.

        Args:
            secret: TOTP secret
            issuer: App name shown in authenticator
            account: Account identifier

        Returns:
            otpauth:// URI for QR code
        """
        if not TOTP_AVAILABLE:
            raise ImportError("pyotp not available")

        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=account, issuer_name=issuer)

    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code from authenticator.

        Args:
            secret: TOTP secret
            code: 6-digit code from authenticator

        Returns:
            True if valid
        """
        if not TOTP_AVAILABLE:
            raise ImportError("pyotp not available")

        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)  # Allow 30s clock skew

    # Biometric (Touch ID, Windows Hello)
    def is_biometric_available(self) -> bool:
        """Check if biometric auth is available."""
        if platform.system() == "Darwin" and BIOMETRIC_AVAILABLE:
            return True
        # Windows Hello — platform-specific, not yet implemented
        return False

    def verify_biometric(self, reason: str = "Authenticate to Victor") -> bool:
        """Verify using biometric (Touch ID, Windows Hello).

        Args:
            reason: Reason shown to user

        Returns:
            True if verified
        """
        if platform.system() == "Darwin" and BIOMETRIC_AVAILABLE:
            return self._verify_touch_id(reason)

        # Windows Hello — platform-specific, not yet implemented
        logger.warning("Biometric auth not available on this platform")
        return False

    def _verify_touch_id(self, reason: str) -> bool:
        """Verify using macOS Touch ID."""
        try:
            context = LocalAuthentication.LAContext.alloc().init()
            error = None

            # Check if Touch ID is available
            can_evaluate, error = context.canEvaluatePolicy_error_(
                LocalAuthentication.LAPolicyDeviceOwnerAuthenticationWithBiometrics,
                None,
            )

            if not can_evaluate:
                logger.debug(f"Touch ID not available: {error}")
                return False

            # Perform authentication
            import threading

            result = {"success": False}
            event = threading.Event()

            def callback(success, error):
                result["success"] = success
                event.set()

            context.evaluatePolicy_localizedReason_reply_(
                LocalAuthentication.LAPolicyDeviceOwnerAuthenticationWithBiometrics,
                reason,
                callback,
            )

            # Wait for callback (with timeout)
            event.wait(timeout=30)
            return result["success"]

        except Exception as e:
            logger.error(f"Touch ID verification failed: {e}")
            return False

    # Passkeys (FIDO2/WebAuthn)
    def is_passkey_available(self) -> bool:
        """Check if passkey/FIDO2 is available."""
        # FIDO2/WebAuthn — requires python-fido2 package, not yet implemented
        return False

    def verify_passkey(self, credential_id: str) -> bool:
        """Verify using passkey/FIDO2 hardware key."""
        # FIDO2 verification — not yet implemented
        logger.warning("Passkey auth not yet implemented")
        return False

    # Device PIN fallback
    def verify_device_pin(self, expected_hash: str, pin: str) -> bool:
        """Verify device PIN.

        Args:
            expected_hash: SHA-256 hash of correct PIN
            pin: User-entered PIN

        Returns:
            True if PIN matches
        """
        pin_hash = hashlib.sha256(pin.encode()).hexdigest()
        return pin_hash == expected_hash

    def hash_pin(self, pin: str) -> str:
        """Hash a PIN for storage."""
        return hashlib.sha256(pin.encode()).hexdigest()

    # Unified verification
    def verify(
        self,
        config: MFAConfig,
        credential_key: str,
        prompt_callback: Optional[callable] = None,
    ) -> bool:
        """Verify MFA based on configuration.

        Args:
            config: MFA configuration
            credential_key: Key for caching
            prompt_callback: Function to prompt user for code/pin
                            Signature: (method: str, message: str) -> str

        Returns:
            True if verified (or no MFA required)
        """
        if config.method == MFAMethod.NONE:
            return True

        # Check cache first
        if self.is_cached(credential_key, config.cache_duration):
            logger.debug(f"MFA cached for {credential_key}")
            return True

        method = config.method
        verified = False

        # Try primary method
        if method == MFAMethod.TOTP:
            if not config.totp_secret:
                logger.error("TOTP secret not configured")
                return False

            if prompt_callback:
                code = prompt_callback("totp", "Enter authenticator code: ")
                verified = self.verify_totp(config.totp_secret, code)
            else:
                logger.error("No prompt callback for TOTP verification")
                return False

        elif method == MFAMethod.BIOMETRIC:
            if self.is_biometric_available():
                verified = self.verify_biometric(f"Access credential: {credential_key}")
            else:
                # Fall through to fallback
                method = config.fallback_method

        elif method == MFAMethod.PASSKEY:
            if self.is_passkey_available():
                verified = self.verify_passkey(credential_key)
            else:
                method = config.fallback_method

        # Try fallback if primary failed and fallback configured
        if not verified and config.fallback_method and method != config.fallback_method:
            logger.debug(f"Trying fallback method: {config.fallback_method}")
            fallback_config = MFAConfig(
                method=config.fallback_method,
                totp_secret=config.totp_secret,
                cache_duration=config.cache_duration,
            )
            verified = self.verify(fallback_config, credential_key, prompt_callback)

        if verified:
            self.cache_verification(credential_key)

        return verified


# Global MFA verifier instance
_mfa_verifier: Optional[MFAVerifier] = None


def get_mfa_verifier() -> MFAVerifier:
    """Get the global MFA verifier instance."""
    global _mfa_verifier
    if _mfa_verifier is None:
        _mfa_verifier = MFAVerifier()
    return _mfa_verifier


# =============================================================================
# Smart Card / PIV / CAC Support
# =============================================================================


class SmartCardType(Enum):
    """Types of smart cards."""

    PIV = "piv"  # Personal Identity Verification (FIPS 201)
    CAC = "cac"  # Common Access Card (DoD)
    X509 = "x509"  # Generic X.509 certificate


@dataclass
class SmartCardCredentials:
    """Smart card / PIV / CAC credentials.

    Supports FedRAMP-compliant certificate-based authentication
    using hardware tokens (PIV cards, CAC cards, YubiKey).

    Attributes:
        card_type: Type of smart card
        certificate_subject: X.509 certificate subject (CN)
        certificate_issuer: Certificate issuer
        serial_number: Card/certificate serial number
        slot: PKCS#11 slot number
        pin_required: Whether PIN is required
        middleware: PKCS#11 middleware path
    """

    card_type: SmartCardType
    certificate_subject: Optional[str] = None
    certificate_issuer: Optional[str] = None
    serial_number: Optional[str] = None
    slot: int = 0
    pin_required: bool = True
    middleware: Optional[str] = None  # Path to PKCS#11 library

    @classmethod
    def detect_cards(cls) -> List["SmartCardCredentials"]:
        """Detect available smart cards.

        Returns:
            List of detected smart card credentials
        """
        cards = []

        # Try to detect via pkcs11-tool or system APIs
        try:
            import subprocess

            # Use pkcs11-tool to list tokens
            result = subprocess.run(
                ["pkcs11-tool", "--list-slots"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Parse output for slots with tokens
                for line in result.stdout.splitlines():
                    if "token present" in line.lower():
                        slot_num = 0  # Parse slot number
                        cards.append(
                            cls(
                                card_type=SmartCardType.PIV,
                                slot=slot_num,
                            )
                        )
        except FileNotFoundError:
            logger.debug("pkcs11-tool not found")
        except Exception as e:
            logger.debug(f"Smart card detection failed: {e}")

        # macOS: Check for CryptoTokenKit
        if platform.system() == "Darwin":
            try:
                import subprocess

                result = subprocess.run(
                    ["security", "list-smartcards"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    cards.append(
                        cls(
                            card_type=SmartCardType.PIV,
                            middleware=cls._get_pkcs11_middleware(),
                        )
                    )
            except Exception:
                pass

        return cards

    @classmethod
    def _get_pkcs11_middleware(cls) -> Optional[str]:
        """Get platform-specific PKCS#11 middleware path.

        Returns:
            Path to PKCS#11 library or None if not found
        """
        system = platform.system()

        # Platform-specific default paths
        if system == "Darwin":
            candidates = [
                "/usr/local/lib/pkcs11/opensc-pkcs11.so",
                "/opt/homebrew/lib/pkcs11/opensc-pkcs11.so",
                "/Library/OpenSC/lib/opensc-pkcs11.so",
            ]
        elif system == "Linux":
            candidates = [
                "/usr/lib/x86_64-linux-gnu/opensc-pkcs11.so",
                "/usr/lib64/opensc-pkcs11.so",
                "/usr/lib/opensc-pkcs11.so",
            ]
        elif system == "Windows":
            # Windows uses DLLs
            candidates = [
                str(
                    Path(os.environ.get("PROGRAMFILES", "C:\\Program Files"))
                    / "OpenSC Project"
                    / "OpenSC"
                    / "pkcs11"
                    / "opensc-pkcs11.dll"
                ),
                str(
                    Path(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"))
                    / "OpenSC Project"
                    / "OpenSC"
                    / "pkcs11"
                    / "opensc-pkcs11.dll"
                ),
            ]
        else:
            candidates = []

        # Find first existing library
        for path in candidates:
            if Path(path).exists():
                return path

        return None

    def get_certificate(self, pin: Optional[str] = None) -> Optional[bytes]:
        """Extract certificate from smart card.

        Args:
            pin: Card PIN (if required)

        Returns:
            DER-encoded certificate bytes
        """
        try:
            import subprocess

            cmd = [
                "pkcs11-tool",
                "--read-object",
                "--type",
                "cert",
                "--slot",
                str(self.slot),
            ]

            if pin:
                cmd.extend(["--pin", pin])

            if self.middleware:
                cmd.extend(["--module", self.middleware])

            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0:
                return result.stdout

        except Exception as e:
            logger.error(f"Failed to read certificate: {e}")

        return None

    def sign_challenge(self, challenge: bytes, pin: str) -> Optional[bytes]:
        """Sign a challenge using the smart card private key.

        Used for challenge-response authentication.

        Args:
            challenge: Challenge bytes to sign
            pin: Card PIN

        Returns:
            Signature bytes
        """
        try:
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(challenge)
                challenge_file = f.name

            cmd = [
                "pkcs11-tool",
                "--sign",
                "--slot",
                str(self.slot),
                "--pin",
                pin,
                "--input-file",
                challenge_file,
                "--mechanism",
                "SHA256-RSA-PKCS",
            ]

            if self.middleware:
                cmd.extend(["--module", self.middleware])

            result = subprocess.run(cmd, capture_output=True)

            os.unlink(challenge_file)

            if result.returncode == 0:
                return result.stdout

        except Exception as e:
            logger.error(f"Failed to sign challenge: {e}")

        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "smartcard",
            "card_type": self.card_type.value,
            "certificate_subject": self.certificate_subject,
            "certificate_issuer": self.certificate_issuer,
            "serial_number": self.serial_number,
            "slot": self.slot,
            "pin_required": self.pin_required,
            "middleware": self.middleware,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmartCardCredentials":
        return cls(
            card_type=SmartCardType(data.get("card_type", "piv")),
            certificate_subject=data.get("certificate_subject"),
            certificate_issuer=data.get("certificate_issuer"),
            serial_number=data.get("serial_number"),
            slot=data.get("slot", 0),
            pin_required=data.get("pin_required", True),
            middleware=data.get("middleware"),
        )


# =============================================================================
# SSO / OAuth / OIDC Support
# =============================================================================


class SSOProvider(Enum):
    """Supported SSO providers."""

    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    ONELOGIN = "onelogin"
    PING = "ping"
    KEYCLOAK = "keycloak"
    GENERIC_OIDC = "oidc"
    GENERIC_SAML = "saml"
    # LLM provider subscription OAuth (FEP-0004)
    OPENAI_CODEX = "openai_codex"
    QWEN = "qwen"
    GOOGLE_GEMINI = "google_gemini"
    GITHUB_COPILOT = "github_copilot"


@dataclass
class SSOConfig:
    """SSO / OIDC configuration.

    Supports enterprise identity providers for Victor CLI authentication.

    Attributes:
        provider: SSO provider type
        issuer_url: OIDC issuer URL
        client_id: OAuth client ID
        client_secret: OAuth client secret (optional for PKCE)
        redirect_uri: OAuth redirect URI
        scopes: OAuth scopes to request
        audience: API audience (for JWT validation)
        organization_id: Okta organization ID
        use_pkce: Use PKCE flow (recommended)
    """

    provider: SSOProvider
    issuer_url: str
    client_id: str
    client_secret: Optional[str] = None
    redirect_uri: str = "http://localhost:8400/callback"
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    audience: Optional[str] = None
    organization_id: Optional[str] = None  # Okta org ID
    use_pkce: bool = True
    # Config-driven overrides (eliminate per-provider if/elif in SSOAuthenticator)
    authorize_path: Optional[str] = None  # e.g. "/o/oauth2/v2/auth" for Google
    token_url: Optional[str] = None  # Full token URL when host differs from issuer
    extra_auth_params: Optional[Dict[str, str]] = None  # e.g. {"access_type": "offline"}
    # Device code flow (for GitHub Copilot, headless environments)
    use_device_flow: bool = False  # Use device code flow instead of browser redirect
    device_code_url: Optional[str] = None  # URL for device code requests

    @classmethod
    def for_okta(
        cls,
        domain: str,
        client_id: str,
        client_secret: Optional[str] = None,
    ) -> "SSOConfig":
        """Create Okta SSO configuration.

        Args:
            domain: Okta domain (e.g., 'mycompany.okta.com')
            client_id: OAuth client ID
            client_secret: OAuth client secret (optional for PKCE)
        """
        return cls(
            provider=SSOProvider.OKTA,
            issuer_url=f"https://{domain}",
            client_id=client_id,
            client_secret=client_secret,
            scopes=["openid", "profile", "email", "groups"],
        )

    @classmethod
    def for_azure_ad(
        cls,
        tenant_id: str,
        client_id: str,
        client_secret: Optional[str] = None,
    ) -> "SSOConfig":
        """Create Azure AD SSO configuration."""
        return cls(
            provider=SSOProvider.AZURE_AD,
            issuer_url=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            client_id=client_id,
            client_secret=client_secret,
        )

    @classmethod
    def for_google(cls, client_id: str, client_secret: str) -> "SSOConfig":
        """Create Google SSO configuration."""
        return cls(
            provider=SSOProvider.GOOGLE,
            issuer_url="https://accounts.google.com",
            client_id=client_id,
            client_secret=client_secret,
        )

    @classmethod
    def for_openai_codex(cls) -> "SSOConfig":
        """Create OpenAI Codex subscription OAuth configuration.

        Uses OpenAI's public OAuth client for ChatGPT Plus/Pro/Enterprise
        subscription-based access. See FEP-0004.

        Scopes and endpoints sourced from openai/codex (codex-rs/login/src/server.rs).
        """
        return cls(
            provider=SSOProvider.OPENAI_CODEX,
            issuer_url="https://auth.openai.com",
            client_id="app_EMoamEEZ73f0CkXaXp7hrann",
            scopes=[
                "openid",
                "profile",
                "email",
                "offline_access",
                "api.connectors.read",
                "api.connectors.invoke",
            ],
            redirect_uri="http://localhost:1455/auth/callback",
            use_pkce=True,
        )

    @classmethod
    def for_qwen(cls) -> "SSOConfig":
        """Create Qwen (Alibaba) subscription OAuth configuration.

        Uses Qwen's OAuth for Coding Plan subscription access. See FEP-0004.
        """
        return cls(
            provider=SSOProvider.QWEN,
            issuer_url="https://chat.qwen.ai",
            client_id="qwen-code",
            scopes=["openid", "profile", "email", "offline_access"],
            use_pkce=True,
        )

    @classmethod
    def for_google_gemini(cls) -> "SSOConfig":
        """Create Google Gemini subscription OAuth configuration.

        Uses Google's public OAuth client (same as Gemini CLI).
        Google AI Pro/Ultra subscribers can use this to access Gemini models
        via their subscription instead of per-token API billing.

        Client ID and secret are public (standard for Google installed apps).
        Source: gemini-cli/packages/core/src/code_assist/oauth2.ts
        """
        return cls(
            provider=SSOProvider.GOOGLE_GEMINI,
            issuer_url="https://accounts.google.com",
            client_id=(
                "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j" ".apps.googleusercontent.com"
            ),
            client_secret="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",
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ],
            authorize_path="/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            extra_auth_params={"access_type": "offline"},
            redirect_uri="http://127.0.0.1:8401/oauth2callback",
            use_pkce=True,
        )

    @classmethod
    def for_github_copilot(cls) -> "SSOConfig":
        """Create GitHub Copilot subscription OAuth configuration.

        Uses OAuth device code flow (no browser redirect needed).
        Requires a GitHub Copilot subscription (Individual, Business, or Enterprise).
        Third-party access is officially supported via Copilot SDK (April 2026).

        Source: github/copilot-sdk/docs/setup/github-oauth.md
        """
        return cls(
            provider=SSOProvider.GITHUB_COPILOT,
            issuer_url="https://github.com",
            client_id="",  # Must be configured per-app via GitHub OAuth App
            scopes=["copilot"],
            use_pkce=False,
            use_device_flow=True,
            device_code_url="https://github.com/login/device/code",
            token_url="https://github.com/login/oauth/access_token",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "issuer_url": self.issuer_url,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "scopes": self.scopes,
            "audience": self.audience,
            "organization_id": self.organization_id,
            "use_pkce": self.use_pkce,
            "authorize_path": self.authorize_path,
            "token_url": self.token_url,
            "extra_auth_params": self.extra_auth_params,
            "use_device_flow": self.use_device_flow,
            "device_code_url": self.device_code_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSOConfig":
        return cls(
            provider=SSOProvider(data["provider"]),
            issuer_url=data["issuer_url"],
            client_id=data["client_id"],
            client_secret=data.get("client_secret"),
            redirect_uri=data.get("redirect_uri", "http://localhost:8400/callback"),
            scopes=data.get("scopes", ["openid", "profile", "email"]),
            audience=data.get("audience"),
            organization_id=data.get("organization_id"),
            use_pkce=data.get("use_pkce", True),
        )


@dataclass
class SSOTokens:
    """OAuth tokens from SSO authentication."""

    access_token: str
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    scopes: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": self.scopes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSOTokens":
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            id_token=data.get("id_token"),
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scopes=data.get("scopes", []),
        )


class SSOAuthenticator:
    """SSO authentication handler.

    Supports OAuth 2.0 / OIDC flows with enterprise IdPs.

    Example:
        config = SSOConfig.for_okta("mycompany.okta.com", "client_id")
        auth = SSOAuthenticator(config)

        # Start browser-based login
        tokens = await auth.login()

        # Refresh tokens
        new_tokens = await auth.refresh(tokens.refresh_token)
    """

    def __init__(self, config: SSOConfig):
        self.config = config
        self._server: Optional[Any] = None

    async def login(self, timeout: int = 120) -> SSOTokens:
        """Perform SSO login via browser or device code flow.

        Uses device code flow if config.use_device_flow is True,
        otherwise uses browser-based PKCE flow with local callback server.

        Args:
            timeout: Maximum time to wait for login

        Returns:
            SSOTokens with access/refresh tokens
        """
        # Device code flow (GitHub Copilot, headless environments)
        if self.config.use_device_flow:
            return await self._device_code_login(timeout)

        import secrets
        import urllib.parse
        import webbrowser

        # Generate state and PKCE verifier
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)

        if self.config.use_pkce:
            import hashlib

            code_challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
                .decode()
                .rstrip("=")
            )
            code_challenge_method = "S256"
        else:
            code_challenge = None
            code_challenge_method = None

        # Build authorization URL
        auth_params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
        }

        if code_challenge:
            auth_params["code_challenge"] = code_challenge
            auth_params["code_challenge_method"] = code_challenge_method

        if self.config.audience:
            auth_params["audience"] = self.config.audience

        # OpenAI Codex-specific params (from openai/codex server.rs)
        if self.config.provider == SSOProvider.OPENAI_CODEX:
            auth_params["codex_cli_simplified_flow"] = "true"
            auth_params["originator"] = "victor"

        # Merge provider-specific extra params (e.g. access_type=offline for Google)
        if self.config.extra_auth_params:
            auth_params.update(self.config.extra_auth_params)

        # Config-driven authorization path (new providers just set authorize_path)
        if self.config.authorize_path:
            authorize_path = self.config.authorize_path
        elif self.config.provider == SSOProvider.OPENAI_CODEX:
            authorize_path = "/oauth/authorize"
        elif self.config.provider == SSOProvider.OKTA:
            authorize_path = "/oauth2/v1/authorize"
        else:
            authorize_path = "/authorize"

        auth_url = f"{self.config.issuer_url}{authorize_path}?{urllib.parse.urlencode(auth_params)}"

        # Start local callback server
        callback_result: Dict[str, Any] = {}
        callback_event = asyncio.Event()

        async def handle_callback(request):
            from aiohttp import web

            code = request.query.get("code")
            returned_state = request.query.get("state")
            error = request.query.get("error")

            if error:
                callback_result["error"] = error
            elif returned_state != state:
                callback_result["error"] = "State mismatch"
            else:
                callback_result["code"] = code

            callback_event.set()

            return web.Response(
                text="<html><body><h1>Login successful!</h1>"
                "<p>You can close this window.</p></body></html>",
                content_type="text/html",
            )

        # Start server
        from aiohttp import web

        app = web.Application()
        parsed_redirect = urllib.parse.urlparse(self.config.redirect_uri)
        callback_path = parsed_redirect.path or "/callback"
        app.router.add_get(callback_path, handle_callback)

        port = parsed_redirect.port or 8400

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", port)
        await site.start()

        try:
            # Open browser
            logger.info("Opening browser for SSO login...")
            webbrowser.open(auth_url)

            # Wait for callback
            try:
                await asyncio.wait_for(callback_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError("SSO login timed out")

            if "error" in callback_result:
                raise ValueError(f"SSO error: {callback_result['error']}")

            # Exchange code for tokens
            return await self._exchange_code(
                callback_result["code"],
                code_verifier if self.config.use_pkce else None,
            )

        finally:
            await runner.cleanup()

    @staticmethod
    def _get_ssl_context():
        """Create SSL context using certifi certs (fixes macOS venv SSL errors)."""
        import ssl

        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return None  # Fall back to system default

    async def _device_code_login(self, timeout: int = 120) -> SSOTokens:
        """Perform OAuth device code flow (RFC 8628).

        Used by GitHub Copilot and other providers that support headless auth.
        User visits a URL, enters a code, and the CLI polls for completion.

        Args:
            timeout: Maximum time to wait for user to complete auth

        Returns:
            SSOTokens with access token
        """
        import aiohttp

        device_code_url = self.config.device_code_url
        if not device_code_url:
            device_code_url = f"{self.config.issuer_url}/login/device/code"

        # Resolve token URL
        if self.config.token_url:
            token_url = self.config.token_url
        else:
            token_url = f"{self.config.issuer_url}/login/oauth/access_token"

        # Step 1: Request device code
        ssl_ctx = self._get_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None

        headers = {"Accept": "application/json"}
        data = {
            "client_id": self.config.client_id,
            "scope": " ".join(self.config.scopes),
        }

        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(device_code_url, data=data, headers=headers) as response:
                if response.status != 200:
                    error = await response.text()
                    raise ValueError(f"Device code request failed: {error}")
                result = await response.json()

        device_code = result["device_code"]
        user_code = result["user_code"]
        verification_uri = result.get("verification_uri", result.get("verification_url", ""))
        interval = result.get("interval", 5)
        expires_in = result.get("expires_in", timeout)

        # Step 2: Show user the code and URL
        import webbrowser

        print(f"\n  Visit: {verification_uri}")
        print(f"  Enter code: {user_code}\n")

        try:
            webbrowser.open(verification_uri)
        except Exception:
            pass  # Browser open is best-effort

        # Step 3: Poll for token
        poll_data = {
            "client_id": self.config.client_id,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
        if self.config.client_secret:
            poll_data["client_secret"] = self.config.client_secret

        import time

        deadline = time.time() + min(timeout, expires_in)
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None

        async with aiohttp.ClientSession(connector=connector) as session:
            while time.time() < deadline:
                await asyncio.sleep(interval)

                async with session.post(token_url, data=poll_data, headers=headers) as response:
                    token_result = await response.json()

                error = token_result.get("error")
                if error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    interval = min(interval + 5, 30)
                    continue
                elif error:
                    raise ValueError(
                        f"Device code auth failed: {error} - "
                        f"{token_result.get('error_description', '')}"
                    )

                # Success
                access_token = token_result["access_token"]
                expires_in_secs = token_result.get("expires_in", 28800)

                return SSOTokens(
                    access_token=access_token,
                    refresh_token=token_result.get("refresh_token"),
                    token_type=token_result.get("token_type", "Bearer"),
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in_secs),
                    scopes=self.config.scopes,
                )

        raise TimeoutError("Device code authentication timed out")

    async def _exchange_code(
        self,
        code: str,
        code_verifier: Optional[str],
    ) -> SSOTokens:
        """Exchange authorization code for tokens."""
        import aiohttp

        # Config-driven token URL (new providers just set token_url)
        if self.config.token_url:
            token_url = self.config.token_url
        elif self.config.provider == SSOProvider.OKTA:
            token_url = f"{self.config.issuer_url}/oauth2/v1/token"
        else:
            token_url = f"{self.config.issuer_url}/oauth/token"

        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "code": code,
            "redirect_uri": self.config.redirect_uri,
        }

        if code_verifier:
            data["code_verifier"] = code_verifier

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        ssl_ctx = self._get_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise ValueError(f"Token exchange failed: {error}")

                result = await response.json()

                expires_at = None
                if "expires_in" in result:
                    expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=result["expires_in"]
                    )

                return SSOTokens(
                    access_token=result["access_token"],
                    refresh_token=result.get("refresh_token"),
                    id_token=result.get("id_token"),
                    token_type=result.get("token_type", "Bearer"),
                    expires_at=expires_at,
                    scopes=result.get("scope", "").split(),
                )

    async def refresh(self, refresh_token: str) -> SSOTokens:
        """Refresh access token using refresh token."""
        import aiohttp

        # Config-driven token URL
        if self.config.token_url:
            token_url = self.config.token_url
        elif self.config.provider == SSOProvider.OKTA:
            token_url = f"{self.config.issuer_url}/oauth2/v1/token"
        else:
            token_url = f"{self.config.issuer_url}/oauth/token"

        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "refresh_token": refresh_token,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        ssl_ctx = self._get_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error = await response.text()
                    raise ValueError(f"Token refresh failed: {error}")

                result = await response.json()

                expires_at = None
                if "expires_in" in result:
                    expires_at = datetime.now(timezone.utc) + timedelta(
                        seconds=result["expires_in"]
                    )

                return SSOTokens(
                    access_token=result["access_token"],
                    refresh_token=result.get("refresh_token", refresh_token),
                    id_token=result.get("id_token"),
                    token_type=result.get("token_type", "Bearer"),
                    expires_at=expires_at,
                )

    async def logout(self, id_token: Optional[str] = None) -> None:
        """Logout from SSO provider."""
        import webbrowser

        logout_url = f"{self.config.issuer_url}/logout"

        params = {"client_id": self.config.client_id}
        if id_token:
            params["id_token_hint"] = id_token

        if self.config.provider == SSOProvider.OKTA:
            logout_url = f"{self.config.issuer_url}/oauth2/v1/logout"

        import urllib.parse

        full_url = f"{logout_url}?{urllib.parse.urlencode(params)}"
        webbrowser.open(full_url)


# =============================================================================
# System Authentication (PAM, NTLM, Kerberos, AD)
# =============================================================================


class SystemAuthType(Enum):
    """System authentication types."""

    PAM = "pam"  # Linux PAM
    KERBEROS = "kerberos"  # Kerberos/GSSAPI
    NTLM = "ntlm"  # Windows NTLM
    ACTIVE_DIRECTORY = "ad"  # Windows Active Directory


@dataclass
class SystemAuthConfig:
    """System authentication configuration.

    Leverages existing OS authentication mechanisms:
    - Linux: PAM (pam.d), Kerberos (krb5)
    - macOS: PAM, Kerberos, Security Framework
    - Windows: NTLM, Kerberos, Active Directory

    This allows Victor to use existing authenticated sessions
    without requiring users to re-enter credentials.

    Attributes:
        auth_type: Type of system authentication
        service_name: PAM service name (default: 'login')
        realm: Kerberos realm (e.g., 'CORP.EXAMPLE.COM')
        domain: AD domain (e.g., 'corp.example.com')
        use_current_user: Use current logged-in user
        principal: Kerberos principal or AD username
    """

    auth_type: SystemAuthType
    service_name: str = "login"  # PAM service
    realm: Optional[str] = None  # Kerberos realm
    domain: Optional[str] = None  # AD domain
    use_current_user: bool = True
    principal: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auth_type": self.auth_type.value,
            "service_name": self.service_name,
            "realm": self.realm,
            "domain": self.domain,
            "use_current_user": self.use_current_user,
            "principal": self.principal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemAuthConfig":
        return cls(
            auth_type=SystemAuthType(data["auth_type"]),
            service_name=data.get("service_name", "login"),
            realm=data.get("realm"),
            domain=data.get("domain"),
            use_current_user=data.get("use_current_user", True),
            principal=data.get("principal"),
        )


class SystemAuthenticator:
    """System-level authentication handler.

    Integrates with OS authentication mechanisms for seamless SSO
    with enterprise environments.

    Supported platforms:
    - Linux: PAM (/etc/pam.d), Kerberos (kinit, klist)
    - macOS: PAM, Security Framework, Kerberos
    - Windows: NTLM, Kerberos (SSPI), Active Directory

    Example:
        auth = SystemAuthenticator()

        # Check for valid Kerberos ticket
        if auth.has_valid_kerberos_ticket():
            principal = auth.get_kerberos_principal()
            print(f"Authenticated as: {principal}")

        # Verify PAM credentials
        if auth.verify_pam_credentials("username", "password"):
            print("PAM authentication successful")
    """

    def __init__(self, config: Optional[SystemAuthConfig] = None):
        self.config = config or SystemAuthConfig(auth_type=SystemAuthType.PAM)

    # =========================================================================
    # PAM Authentication (Linux/macOS)
    # =========================================================================

    def verify_pam_credentials(
        self,
        username: str,
        password: str,
        service: str = "login",
    ) -> bool:
        """Verify credentials using PAM.

        Args:
            username: Username
            password: Password
            service: PAM service name (from /etc/pam.d/)

        Returns:
            True if authentication successful
        """
        if platform.system() == "Windows":
            logger.warning("PAM not available on Windows")
            return False

        try:
            import pam

            p = pam.pam()
            return p.authenticate(username, password, service=service)
        except ImportError:
            # Try python-pam
            try:
                import PAM

                def pam_conv(auth, query_list, userData):
                    resp = []
                    for query, qtype in query_list:
                        if qtype == PAM.PAM_PROMPT_ECHO_OFF:
                            resp.append((password, 0))
                        elif qtype == PAM.PAM_PROMPT_ECHO_ON:
                            resp.append((username, 0))
                        else:
                            resp.append(("", 0))
                    return resp

                auth = PAM.pam()
                auth.start(service)
                auth.set_item(PAM.PAM_USER, username)
                auth.set_item(PAM.PAM_CONV, pam_conv)

                try:
                    auth.authenticate()
                    auth.acct_mgmt()
                    return True
                except PAM.error:
                    return False
            except ImportError:
                logger.error("PAM module not available. Install: pip install python-pam")
                return False

    def get_current_pam_user(self) -> Optional[str]:
        """Get current PAM-authenticated user.

        Returns:
            Username or None
        """
        return os.environ.get("USER") or os.environ.get("LOGNAME")

    # =========================================================================
    # Kerberos Authentication (Linux/macOS/Windows)
    # =========================================================================

    def has_valid_kerberos_ticket(self) -> bool:
        """Check if user has a valid Kerberos ticket.

        Returns:
            True if valid ticket exists
        """
        try:
            import subprocess

            # Use klist to check for valid tickets
            result = subprocess.run(
                ["klist", "-s"],
                capture_output=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            # Try Windows
            if platform.system() == "Windows":
                try:
                    result = subprocess.run(
                        ["klist"],
                        capture_output=True,
                        text=True,
                    )
                    return "Cached Tickets" in result.stdout
                except FileNotFoundError:
                    pass
            return False

    def get_kerberos_principal(self) -> Optional[str]:
        """Get current Kerberos principal.

        Returns:
            Principal name (e.g., 'user@REALM.COM') or None
        """
        try:
            import subprocess

            result = subprocess.run(
                ["klist"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Parse output for principal
                for line in result.stdout.splitlines():
                    if "Principal:" in line or "Default principal:" in line:
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            return parts[1].strip()
            return None
        except Exception:
            return None

    def kinit(
        self,
        principal: str,
        password: Optional[str] = None,
        keytab: Optional[str] = None,
    ) -> bool:
        """Obtain Kerberos ticket.

        Args:
            principal: Kerberos principal
            password: Password (if not using keytab)
            keytab: Path to keytab file

        Returns:
            True if successful
        """
        try:
            import subprocess

            cmd = ["kinit"]

            if keytab:
                cmd.extend(["-k", "-t", keytab])

            cmd.append(principal)

            if password and not keytab:
                # Use stdin for password
                result = subprocess.run(
                    cmd,
                    input=password.encode(),
                    capture_output=True,
                )
            else:
                result = subprocess.run(cmd, capture_output=True)

            return result.returncode == 0
        except FileNotFoundError:
            logger.error("kinit not found. Install Kerberos tools.")
            return False

    def get_kerberos_service_ticket(self, service: str) -> Optional[bytes]:
        """Get Kerberos service ticket for delegation.

        Args:
            service: Service principal (e.g., 'HTTP/server.example.com')

        Returns:
            Ticket bytes or None
        """
        try:
            # Try GSSAPI (Unix/macOS)
            import gssapi

            server_name = gssapi.Name(service, gssapi.NameType.hostbased_service)

            ctx = gssapi.SecurityContext(
                name=server_name,
                usage="initiate",
            )

            token = ctx.step()
            return bytes(token) if token else None

        except ImportError:
            logger.debug("gssapi not available")

        # Try pyspnego (cross-platform)
        try:
            import spnego

            ctx = spnego.client(service)
            token = ctx.step()
            return token

        except ImportError:
            logger.debug("spnego not available")

        return None

    # =========================================================================
    # NTLM Authentication (Windows)
    # =========================================================================

    def verify_ntlm_credentials(
        self,
        username: str,
        password: str,
        domain: Optional[str] = None,
    ) -> bool:
        """Verify credentials using NTLM (Windows).

        Args:
            username: Username
            password: Password
            domain: Domain (optional)

        Returns:
            True if authentication successful
        """
        if platform.system() != "Windows":
            logger.warning("NTLM native auth only available on Windows")
            return self._verify_ntlm_pyspnego(username, password, domain)

        try:
            import win32security

            if domain:
                pass
            else:
                pass

            # LogonUser validates credentials
            handle = win32security.LogonUser(
                username,
                domain or "",
                password,
                win32security.LOGON32_LOGON_NETWORK,
                win32security.LOGON32_PROVIDER_DEFAULT,
            )

            handle.Close()
            return True

        except ImportError:
            logger.error("pywin32 not available. Install: pip install pywin32")
            return False
        except Exception as e:
            logger.debug(f"NTLM auth failed: {e}")
            return False

    def _verify_ntlm_pyspnego(
        self,
        username: str,
        password: str,
        domain: Optional[str],
    ) -> bool:
        """Verify NTLM credentials using pyspnego (cross-platform)."""
        try:
            import spnego

            # Create NTLM context for validation
            ctx = spnego.client(
                username=username,
                password=password,
                hostname=domain or "localhost",
                service="host",
                protocol="ntlm",
            )

            # Complete authentication exchange
            token = ctx.step()
            return token is not None

        except ImportError:
            logger.error("spnego not available. Install: pip install pyspnego")
            return False
        except Exception as e:
            logger.debug(f"NTLM validation failed: {e}")
            return False

    def get_ntlm_token(self) -> Optional[bytes]:
        """Get NTLM token for current user (Windows SSPI).

        Returns:
            NTLM token bytes or None
        """
        if platform.system() == "Windows":
            try:
                import sspi
                import sspicon

                sspi.QuerySecurityPackageInfo("NTLM")

                cred, _ = sspi.AcquireCredentialsHandle(
                    None, "NTLM", sspicon.SECPKG_CRED_OUTBOUND, None, None
                )

                ctx = sspi.InitializeSecurityContext(
                    cred,
                    None,
                    "localhost",
                    sspicon.ISC_REQ_CONFIDENTIALITY | sspicon.ISC_REQ_REPLAY_DETECT,
                    None,
                    sspicon.SECURITY_NATIVE_DREP,
                    None,
                    None,
                )

                if ctx[0]:
                    return ctx[1][0].Buffer

            except ImportError:
                logger.debug("sspi not available")

        # Try pyspnego
        try:
            import spnego

            ctx = spnego.client(service="host", hostname="localhost", protocol="ntlm")
            return ctx.step()

        except ImportError:
            pass

        return None

    # =========================================================================
    # Active Directory (Windows)
    # =========================================================================

    def verify_ad_credentials(
        self,
        username: str,
        password: str,
        domain: str,
    ) -> bool:
        """Verify credentials against Active Directory.

        Args:
            username: AD username
            password: Password
            domain: AD domain

        Returns:
            True if authentication successful
        """
        # Try LDAP bind
        try:
            import ldap3

            server = ldap3.Server(
                domain,
                get_info=ldap3.ALL,
                use_ssl=True,
            )

            # Try different UPN formats
            upn = f"{username}@{domain}"

            conn = ldap3.Connection(
                server,
                user=upn,
                password=password,
                auto_bind=True,
            )

            conn.unbind()
            return True

        except ImportError:
            logger.debug("ldap3 not available")
        except Exception as e:
            logger.debug(f"LDAP bind failed: {e}")

        # Fall back to Kerberos
        realm = domain.upper()
        principal = f"{username}@{realm}"
        return self.kinit(principal, password)

    def get_ad_user_info(self, username: str, domain: str) -> Optional[Dict[str, Any]]:
        """Get Active Directory user information.

        Args:
            username: AD username
            domain: AD domain

        Returns:
            User info dict or None
        """
        try:
            import ldap3

            server = ldap3.Server(domain, use_ssl=True)
            conn = ldap3.Connection(server, auto_bind=True)

            # Search for user
            search_base = ",".join(f"DC={p}" for p in domain.split("."))
            search_filter = f"(sAMAccountName={username})"

            conn.search(
                search_base,
                search_filter,
                attributes=["cn", "mail", "memberOf", "department"],
            )

            if conn.entries:
                entry = conn.entries[0]
                return {
                    "cn": str(entry.cn) if hasattr(entry, "cn") else None,
                    "email": str(entry.mail) if hasattr(entry, "mail") else None,
                    "groups": (list(entry.memberOf) if hasattr(entry, "memberOf") else []),
                    "department": (str(entry.department) if hasattr(entry, "department") else None),
                }

        except Exception as e:
            logger.debug(f"AD lookup failed: {e}")

        return None

    def get_ad_groups(self) -> List[str]:
        """Get AD groups for current user.

        Returns:
            List of group names
        """
        if platform.system() == "Windows":
            try:
                import win32net
                import win32api

                username = win32api.GetUserName()
                win32api.GetDomainName()

                groups = []
                level = 0
                resume = 0

                while True:
                    info, total, resume = win32net.NetUserGetLocalGroups(None, username, level)
                    groups.extend([g["name"] for g in info])
                    if not resume:
                        break

                return groups

            except ImportError:
                logger.debug("pywin32 not available")

        # Try whoami (Unix)
        try:
            import subprocess

            result = subprocess.run(["groups"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split()
        except FileNotFoundError:
            pass

        return []

    # =========================================================================
    # Unified Authentication
    # =========================================================================

    def authenticate(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Authenticate using configured method.

        Args:
            username: Username (optional if use_current_user)
            password: Password (optional for Kerberos with ticket)

        Returns:
            Tuple of (success, authenticated_user)
        """
        auth_type = self.config.auth_type

        if auth_type == SystemAuthType.PAM:
            if not username or not password:
                return False, None
            if self.verify_pam_credentials(username, password, self.config.service_name):
                return True, username
            return False, None

        elif auth_type == SystemAuthType.KERBEROS:
            if self.has_valid_kerberos_ticket():
                principal = self.get_kerberos_principal()
                return True, principal

            if username and password:
                principal = self.config.principal or username
                if self.config.realm:
                    principal = f"{principal}@{self.config.realm}"
                if self.kinit(principal, password):
                    return True, principal

            return False, None

        elif auth_type == SystemAuthType.NTLM:
            if not username or not password:
                return False, None
            domain = self.config.domain
            if self.verify_ntlm_credentials(username, password, domain):
                return True, f"{domain}\\{username}" if domain else username
            return False, None

        elif auth_type == SystemAuthType.ACTIVE_DIRECTORY:
            if not self.config.domain:
                return False, None
            if not username or not password:
                return False, None
            if self.verify_ad_credentials(username, password, self.config.domain):
                return True, f"{username}@{self.config.domain}"
            return False, None

        return False, None

    def is_authenticated(self) -> Tuple[bool, Optional[str]]:
        """Check if current session is authenticated.

        Returns:
            Tuple of (is_authenticated, authenticated_user)
        """
        # Check for valid Kerberos ticket
        if self.has_valid_kerberos_ticket():
            principal = self.get_kerberos_principal()
            return True, principal

        # Check for PAM user
        pam_user = self.get_current_pam_user()
        if pam_user:
            return True, pam_user

        # Windows: Check for logged-in user
        if platform.system() == "Windows":
            try:
                import win32api

                user = win32api.GetUserName()
                domain = win32api.GetDomainName()
                return True, f"{domain}\\{user}"
            except ImportError:
                pass

        return False, None


# Global system authenticator
_system_auth: Optional[SystemAuthenticator] = None


def get_system_authenticator(
    config: Optional[SystemAuthConfig] = None,
) -> SystemAuthenticator:
    """Get the global system authenticator instance."""
    global _system_auth
    if _system_auth is None or config is not None:
        _system_auth = SystemAuthenticator(config)
    return _system_auth


# =============================================================================
# Credential Types
# =============================================================================


class CredentialType(Enum):
    """Types of credentials."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DATABASE = "database"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SECRET_MANAGER = "secret_manager"


@dataclass
class AWSCredentials:
    """AWS credentials."""

    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    region: str = "us-east-1"
    profile: str = "default"
    expires_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "aws",
            "access_key_id": self.access_key_id,
            "secret_access_key": self.secret_access_key,
            "session_token": self.session_token,
            "region": self.region,
            "profile": self.profile,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AWSCredentials":
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            access_key_id=data["access_key_id"],
            secret_access_key=data["secret_access_key"],
            session_token=data.get("session_token"),
            region=data.get("region", "us-east-1"),
            profile=data.get("profile", "default"),
            expires_at=expires_at,
        )

    @classmethod
    def from_environment(cls, profile: str = "default") -> Optional["AWSCredentials"]:
        """Load from environment variables."""
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not access_key or not secret_key:
            return None

        return cls(
            access_key_id=access_key,
            secret_access_key=secret_key,
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            profile=profile,
        )

    def to_boto3_session(self) -> Any:
        """Create boto3 session from these credentials."""
        try:
            import boto3

            return boto3.Session(
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                aws_session_token=self.session_token,
                region_name=self.region,
            )
        except ImportError:
            raise ImportError("boto3 not installed. Run: pip install boto3")


@dataclass
class AzureCredentials:
    """Azure credentials."""

    client_id: str
    client_secret: str
    tenant_id: str
    subscription_id: Optional[str] = None
    profile: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "azure",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "tenant_id": self.tenant_id,
            "subscription_id": self.subscription_id,
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureCredentials":
        return cls(
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            tenant_id=data["tenant_id"],
            subscription_id=data.get("subscription_id"),
            profile=data.get("profile", "default"),
        )

    @classmethod
    def from_environment(cls, profile: str = "default") -> Optional["AzureCredentials"]:
        """Load from environment variables."""
        client_id = os.environ.get("AZURE_CLIENT_ID")
        client_secret = os.environ.get("AZURE_CLIENT_SECRET")
        tenant_id = os.environ.get("AZURE_TENANT_ID")

        if not all([client_id, client_secret, tenant_id]):
            return None

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID"),
            profile=profile,
        )


@dataclass
class GCPCredentials:
    """Google Cloud credentials."""

    service_account_json: str  # JSON string or path
    project_id: Optional[str] = None
    profile: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "gcp",
            "service_account_json": self.service_account_json,
            "project_id": self.project_id,
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GCPCredentials":
        return cls(
            service_account_json=data["service_account_json"],
            project_id=data.get("project_id"),
            profile=data.get("profile", "default"),
        )

    @classmethod
    def from_environment(cls, profile: str = "default") -> Optional["GCPCredentials"]:
        """Load from environment variables."""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path:
            return None

        try:
            with open(creds_path) as f:
                service_account_json = f.read()

            return cls(
                service_account_json=service_account_json,
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                profile=profile,
            )
        except Exception:
            return None


@dataclass
class DockerCredentials:
    """Docker registry credentials."""

    registry: str  # e.g., docker.io, ghcr.io, ECR URL
    username: str
    password: str
    email: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "docker",
            "registry": self.registry,
            "username": self.username,
            "password": self.password,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DockerCredentials":
        return cls(
            registry=data["registry"],
            username=data["username"],
            password=data["password"],
            email=data.get("email"),
        )

    def to_auth_config(self) -> Dict[str, str]:
        """Get Docker SDK auth config."""
        return {
            "username": self.username,
            "password": self.password,
            "email": self.email or "",
            "serveraddress": self.registry,
        }


@dataclass
class KubernetesCredentials:
    """Kubernetes credentials."""

    context: str
    kubeconfig_path: Optional[str] = None
    kubeconfig_content: Optional[str] = None  # Base64 encoded
    token: Optional[str] = None  # Service account token
    server: Optional[str] = None  # API server URL
    certificate_authority: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "kubernetes",
            "context": self.context,
            "kubeconfig_path": self.kubeconfig_path,
            "kubeconfig_content": self.kubeconfig_content,
            "token": self.token,
            "server": self.server,
            "certificate_authority": self.certificate_authority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KubernetesCredentials":
        return cls(
            context=data["context"],
            kubeconfig_path=data.get("kubeconfig_path"),
            kubeconfig_content=data.get("kubeconfig_content"),
            token=data.get("token"),
            server=data.get("server"),
            certificate_authority=data.get("certificate_authority"),
        )

    @classmethod
    def from_default(cls) -> Optional["KubernetesCredentials"]:
        """Load from default kubeconfig."""
        kubeconfig_path = os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))

        if os.path.exists(kubeconfig_path):
            return cls(
                context=os.environ.get("KUBE_CONTEXT", ""),
                kubeconfig_path=kubeconfig_path,
            )
        return None


@dataclass
class DatabaseCredentials:
    """Database connection credentials."""

    alias: str
    driver: Literal["postgresql", "mysql", "mongodb", "redis", "sqlite"]
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: Optional[str] = None
    extra_params: Dict[str, str] = field(default_factory=dict)

    @property
    def connection_string(self) -> str:
        """Generate connection string."""
        if self.driver == "postgresql":
            base = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            if self.ssl_mode:
                base += f"?sslmode={self.ssl_mode}"
            return base

        elif self.driver == "mysql":
            return (
                f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            )

        elif self.driver == "mongodb":
            return (
                f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            )

        elif self.driver == "redis":
            if self.password:
                return f"redis://:{self.password}@{self.host}:{self.port}/0"
            return f"redis://{self.host}:{self.port}/0"

        elif self.driver == "sqlite":
            return f"sqlite:///{self.database}"

        raise ValueError(f"Unknown driver: {self.driver}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "database",
            "alias": self.alias,
            "driver": self.driver,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
            "ssl_mode": self.ssl_mode,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseCredentials":
        return cls(
            alias=data["alias"],
            driver=data["driver"],
            host=data["host"],
            port=data["port"],
            database=data["database"],
            username=data["username"],
            password=data["password"],
            ssl_mode=data.get("ssl_mode"),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class APIKeyCredentials:
    """Generic API key credentials."""

    name: str
    key: str
    secret: Optional[str] = None
    endpoint: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "api_key",
            "name": self.name,
            "key": self.key,
            "secret": self.secret,
            "endpoint": self.endpoint,
            "headers": self.headers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKeyCredentials":
        return cls(
            name=data["name"],
            key=data["key"],
            secret=data.get("secret"),
            endpoint=data.get("endpoint"),
            headers=data.get("headers", {}),
        )


# Type alias for all credential types
Credential = Union[
    AWSCredentials,
    AzureCredentials,
    GCPCredentials,
    DockerCredentials,
    KubernetesCredentials,
    DatabaseCredentials,
    APIKeyCredentials,
]


# =============================================================================
# Credential Backends
# =============================================================================


class CredentialBackend(ABC):
    """Abstract backend for credential storage."""

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get credential by key."""
        ...

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store credential."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete credential."""
        ...

    @abstractmethod
    def list_keys(self) -> List[str]:
        """List all stored keys."""
        ...


class EnvironmentBackend(CredentialBackend):
    """Environment variable backend (read-only)."""

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        # Parse key format: aws/default, docker/docker.io, etc.
        parts = key.split("/", 1)
        cred_type = parts[0]

        if cred_type == "aws":
            creds = AWSCredentials.from_environment(parts[1] if len(parts) > 1 else "default")
            return creds.to_dict() if creds else None

        elif cred_type == "azure":
            creds = AzureCredentials.from_environment()
            return creds.to_dict() if creds else None

        elif cred_type == "gcp":
            creds = GCPCredentials.from_environment()
            return creds.to_dict() if creds else None

        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        raise NotImplementedError("Environment backend is read-only")

    def delete(self, key: str) -> bool:
        raise NotImplementedError("Environment backend is read-only")

    def list_keys(self) -> List[str]:
        keys = []
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            keys.append("aws/default")
        if os.environ.get("AZURE_CLIENT_ID"):
            keys.append("azure/default")
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            keys.append("gcp/default")
        return keys


class KeyringBackend(CredentialBackend):
    """OS keyring backend (macOS Keychain, Windows Credential Manager)."""

    SERVICE_NAME = "victor-credentials"

    def __init__(self):
        if not KEYRING_AVAILABLE:
            raise ImportError("keyring not available. Install with: pip install keyring")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            value = keyring.get_password(self.SERVICE_NAME, key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.debug(f"Keyring get failed for {key}: {e}")
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        try:
            keyring.set_password(self.SERVICE_NAME, key, json.dumps(value))
        except Exception as e:
            logger.error(f"Keyring set failed for {key}: {e}")
            raise

    def delete(self, key: str) -> bool:
        try:
            keyring.delete_password(self.SERVICE_NAME, key)
            return True
        except Exception:
            return False

    def list_keys(self) -> List[str]:
        # Keyring doesn't support listing, so we need to track keys separately
        index_key = "__victor_key_index__"
        try:
            index = keyring.get_password(self.SERVICE_NAME, index_key)
            if index:
                return json.loads(index)
        except Exception:
            pass
        return []

    def _update_index(self, keys: List[str]) -> None:
        index_key = "__victor_key_index__"
        try:
            keyring.set_password(self.SERVICE_NAME, index_key, json.dumps(keys))
        except Exception as e:
            logger.warning(f"Failed to update key index: {e}")


class FileBackend(CredentialBackend):
    """Encrypted file backend (~/.victor/credentials)."""

    def __init__(self, path: Optional[Path] = None, master_key: Optional[str] = None):
        self.path = path or Path.home() / ".victor" / "credentials"
        self._master_key = master_key
        self._cipher: Optional[Any] = None

        if CRYPTO_AVAILABLE and master_key:
            self._init_cipher(master_key)

    def _init_cipher(self, master_key: str) -> None:
        """Initialize Fernet cipher from master key."""
        # Derive key using PBKDF2
        salt = b"victor-credentials-salt"  # Fixed salt for deterministic key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self._cipher = Fernet(key)

    def _load(self) -> Dict[str, Any]:
        """Load credentials from file."""
        if not self.path.exists():
            return {}

        try:
            content = self.path.read_bytes()

            if self._cipher:
                content = self._cipher.decrypt(content)

            return json.loads(content.decode())
        except Exception as e:
            logger.error(f"Failed to load credentials file: {e}")
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        """Save credentials to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(data, indent=2).encode()

        if self._cipher:
            content = self._cipher.encrypt(content)

        self.path.write_bytes(content)
        # Set restrictive permissions
        self.path.chmod(0o600)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        return data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        data = self._load()
        data[key] = value
        self._save(data)

    def delete(self, key: str) -> bool:
        data = self._load()
        if key in data:
            del data[key]
            self._save(data)
            return True
        return False

    def list_keys(self) -> List[str]:
        return list(self._load().keys())


# =============================================================================
# Credential Manager
# =============================================================================


class CredentialManager:
    """Unified credential manager with layered resolution.

    Resolution order:
    1. Environment variables
    2. OS Keyring
    3. Config file
    4. Cloud IAM (for running on cloud)

    Example:
        creds = CredentialManager()

        # Get AWS credentials
        aws = creds.get_aws("production")

        # Get database credentials
        db = creds.get_database("mydb")

        # Set credentials
        creds.set_aws(AWSCredentials(...))
    """

    def __init__(
        self,
        use_keyring: bool = True,
        config_path: Optional[Path] = None,
        master_key: Optional[str] = None,
    ):
        """Initialize credential manager.

        Args:
            use_keyring: Whether to use OS keyring
            config_path: Path to config file
            master_key: Master key for file encryption
        """
        self._backends: List[CredentialBackend] = []

        # Add backends in priority order
        self._backends.append(EnvironmentBackend())

        if use_keyring and KEYRING_AVAILABLE:
            try:
                self._backends.append(KeyringBackend())
            except Exception as e:
                logger.debug(f"Keyring not available: {e}")

        self._backends.append(FileBackend(config_path, master_key))

        # Primary storage backend (for writes)
        self._storage_backend: CredentialBackend
        if use_keyring and KEYRING_AVAILABLE:
            try:
                self._storage_backend = KeyringBackend()
            except Exception:
                self._storage_backend = FileBackend(config_path, master_key)
        else:
            self._storage_backend = FileBackend(config_path, master_key)

    def _get_raw(self, key: str) -> Optional[Dict[str, Any]]:
        """Get raw credential data from any backend."""
        for backend in self._backends:
            try:
                value = backend.get(key)
                if value:
                    return value
            except Exception as e:
                logger.debug(f"Backend {type(backend).__name__} failed for {key}: {e}")
        return None

    def _parse_credential(self, data: Dict[str, Any]) -> Optional[Credential]:
        """Parse credential data into typed object."""
        cred_type = data.get("type")

        if cred_type == "aws":
            return AWSCredentials.from_dict(data)
        elif cred_type == "azure":
            return AzureCredentials.from_dict(data)
        elif cred_type == "gcp":
            return GCPCredentials.from_dict(data)
        elif cred_type == "docker":
            return DockerCredentials.from_dict(data)
        elif cred_type == "kubernetes":
            return KubernetesCredentials.from_dict(data)
        elif cred_type == "database":
            return DatabaseCredentials.from_dict(data)
        elif cred_type == "api_key":
            return APIKeyCredentials.from_dict(data)

        return None

    # AWS
    def get_aws(self, profile: str = "default") -> Optional[AWSCredentials]:
        """Get AWS credentials."""
        data = self._get_raw(f"aws/{profile}")
        if data:
            creds = AWSCredentials.from_dict(data)
            if not creds.is_expired:
                return creds

        # Try IAM role (EC2, Lambda, EKS)
        try:
            import boto3

            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                frozen = credentials.get_frozen_credentials()
                return AWSCredentials(
                    access_key_id=frozen.access_key,
                    secret_access_key=frozen.secret_key,
                    session_token=frozen.token,
                    region=session.region_name or "us-east-1",
                    profile=profile,
                )
        except Exception:
            pass

        return None

    def set_aws(self, creds: AWSCredentials) -> None:
        """Store AWS credentials."""
        self._storage_backend.set(f"aws/{creds.profile}", creds.to_dict())

    # Azure
    def get_azure(self, profile: str = "default") -> Optional[AzureCredentials]:
        """Get Azure credentials."""
        data = self._get_raw(f"azure/{profile}")
        if data:
            return AzureCredentials.from_dict(data)
        return None

    def set_azure(self, creds: AzureCredentials) -> None:
        """Store Azure credentials."""
        self._storage_backend.set(f"azure/{creds.profile}", creds.to_dict())

    # GCP
    def get_gcp(self, profile: str = "default") -> Optional[GCPCredentials]:
        """Get GCP credentials."""
        data = self._get_raw(f"gcp/{profile}")
        if data:
            return GCPCredentials.from_dict(data)
        return None

    def set_gcp(self, creds: GCPCredentials) -> None:
        """Store GCP credentials."""
        self._storage_backend.set(f"gcp/{creds.profile}", creds.to_dict())

    # Docker
    def get_docker(self, registry: str = "docker.io") -> Optional[DockerCredentials]:
        """Get Docker registry credentials."""
        data = self._get_raw(f"docker/{registry}")
        if data:
            return DockerCredentials.from_dict(data)
        return None

    def set_docker(self, creds: DockerCredentials) -> None:
        """Store Docker credentials."""
        self._storage_backend.set(f"docker/{creds.registry}", creds.to_dict())

    # Kubernetes
    def get_kubernetes(self, context: Optional[str] = None) -> Optional[KubernetesCredentials]:
        """Get Kubernetes credentials."""
        key = f"kubernetes/{context}" if context else "kubernetes/default"
        data = self._get_raw(key)
        if data:
            return KubernetesCredentials.from_dict(data)

        # Try default kubeconfig
        return KubernetesCredentials.from_default()

    def set_kubernetes(self, creds: KubernetesCredentials) -> None:
        """Store Kubernetes credentials."""
        self._storage_backend.set(f"kubernetes/{creds.context}", creds.to_dict())

    # Database
    def get_database(self, alias: str) -> Optional[DatabaseCredentials]:
        """Get database credentials."""
        data = self._get_raw(f"database/{alias}")
        if data:
            return DatabaseCredentials.from_dict(data)
        return None

    def set_database(self, creds: DatabaseCredentials) -> None:
        """Store database credentials."""
        self._storage_backend.set(f"database/{creds.alias}", creds.to_dict())

    # API Keys
    def get_api_key(self, name: str) -> Optional[APIKeyCredentials]:
        """Get API key credentials."""
        data = self._get_raw(f"api_key/{name}")
        if data:
            return APIKeyCredentials.from_dict(data)
        return None

    def set_api_key(self, creds: APIKeyCredentials) -> None:
        """Store API key credentials."""
        self._storage_backend.set(f"api_key/{creds.name}", creds.to_dict())

    # Generic
    def get(self, key: str) -> Optional[Credential]:
        """Get any credential by full key."""
        data = self._get_raw(key)
        if data:
            return self._parse_credential(data)
        return None

    def set(self, key: str, creds: Credential) -> None:
        """Store any credential."""
        self._storage_backend.set(key, creds.to_dict())

    def delete(self, key: str) -> bool:
        """Delete a credential."""
        return self._storage_backend.delete(key)

    def list(self) -> List[str]:
        """List all stored credential keys."""
        keys = set()
        for backend in self._backends:
            try:
                keys.update(backend.list_keys())
            except Exception:
                pass
        return sorted(keys)


# =============================================================================
# Global Instance
# =============================================================================

_default_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get the default credential manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CredentialManager()
    return _default_manager


__all__ = [
    # Types
    "CredentialType",
    "Credential",
    # Credential classes
    "AWSCredentials",
    "AzureCredentials",
    "GCPCredentials",
    "DockerCredentials",
    "KubernetesCredentials",
    "DatabaseCredentials",
    "APIKeyCredentials",
    # MFA
    "MFAMethod",
    "MFAConfig",
    "MFAVerifier",
    "get_mfa_verifier",
    # Smart Card / PIV / CAC
    "SmartCardType",
    "SmartCardCredentials",
    # SSO / OAuth / OIDC
    "SSOProvider",
    "SSOConfig",
    "SSOTokens",
    "SSOAuthenticator",
    # System Authentication (PAM, NTLM, Kerberos, AD)
    "SystemAuthType",
    "SystemAuthConfig",
    "SystemAuthenticator",
    "get_system_authenticator",
    # Backends
    "CredentialBackend",
    "EnvironmentBackend",
    "KeyringBackend",
    "FileBackend",
    # Manager
    "CredentialManager",
    "get_credential_manager",
]
