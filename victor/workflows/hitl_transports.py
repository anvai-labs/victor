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

"""HITL Transport Adapters.

Provides pluggable transport mechanisms for HITL approval requests.
Each transport handles sending notifications and polling/receiving responses
through different channels (email, Slack, GitHub PR, etc.).

Architecture:
    HITLRequest -> HITLTransport.send() -> External System
    External System -> HITLTransport.poll()/webhook -> HITLResponse

Transport Types:
    - Messaging: Email, SMS, Slack, Teams, Discord
    - SCM: GitHub PR, GitLab MR, Bitbucket PR
    - Project: Jira, Linear, Asana, ServiceNow
    - Incident: PagerDuty, OpsGenie, VictorOps
    - Infrastructure: ArgoCD, Terraform Cloud, Spacelift

Example:
    from victor.workflows.hitl_transports import (
        SlackTransport, SlackConfig, get_transport
    )

    # Configure Slack transport
    config = SlackConfig(
        webhook_url="https://hooks.slack.com/...",
        channel="#approvals",
    )
    transport = SlackTransport(config)

    # Send approval request
    await transport.send(request)

    # Poll for response
    response = await transport.poll(request.request_id, timeout=300)
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    runtime_checkable,
)

from victor.workflows.hitl import (
    HITLMode,
    HITLRequest,
    HITLResponse,
    HITLStatus,
)

if TYPE_CHECKING:
    from victor.core.identity import TokenCredential

logger = logging.getLogger(__name__)


# =============================================================================
# Transport Configuration Classes
# =============================================================================


@dataclass
class BaseTransportConfig:
    """Base configuration for all transports."""

    timeout: float = 300.0  # Default timeout in seconds
    poll_interval: float = 5.0  # Polling interval for async transports
    callback_url: Optional[str] = None  # URL for webhook callbacks
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmailConfig(BaseTransportConfig):
    """Configuration for email-based approvals."""

    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    from_address: str = "approvals@example.com"
    to_addresses: List[str] = field(default_factory=list)
    cc_addresses: List[str] = field(default_factory=list)
    subject_template: str = "[Approval Required] {workflow_name}: {prompt}"
    body_template: Optional[str] = None  # Uses default HTML template if None

    @classmethod
    def from_env(cls) -> "EmailConfig":
        """Create config from environment variables."""
        return cls(
            smtp_host=os.getenv("SMTP_HOST", "localhost"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            from_address=os.getenv("SMTP_FROM", "approvals@example.com"),
            to_addresses=os.getenv("APPROVAL_EMAILS", "").split(","),
        )


@dataclass
class SMSConfig(BaseTransportConfig):
    """Configuration for SMS-based approvals (Twilio/AWS SNS)."""

    provider: str = "twilio"  # twilio, sns, or custom
    account_sid: Optional[str] = None  # Twilio Account SID
    auth_token: Optional[str] = None  # Twilio Auth Token
    from_number: Optional[str] = None  # Twilio phone number
    to_numbers: List[str] = field(default_factory=list)
    message_template: str = (
        "{workflow_name}: {prompt}\nApprove: {approve_url}\nReject: {reject_url}"
    )

    @classmethod
    def from_env(cls) -> "SMSConfig":
        """Create config from environment variables."""
        return cls(
            provider=os.getenv("SMS_PROVIDER", "twilio"),
            account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
            from_number=os.getenv("TWILIO_FROM_NUMBER"),
            to_numbers=os.getenv("APPROVAL_PHONES", "").split(","),
        )


@dataclass
class SlackConfig(BaseTransportConfig):
    """Configuration for Slack-based approvals."""

    webhook_url: Optional[str] = None  # Incoming webhook URL
    bot_token: Optional[str] = None  # Bot token for interactive messages
    channel: str = "#approvals"
    mention_users: List[str] = field(default_factory=list)  # User IDs to mention
    mention_groups: List[str] = field(default_factory=list)  # Group IDs to mention
    thread_replies: bool = True  # Reply in thread for updates

    @classmethod
    def from_env(cls) -> "SlackConfig":
        """Create config from environment variables."""
        return cls(
            webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            bot_token=os.getenv("SLACK_BOT_TOKEN"),
            channel=os.getenv("SLACK_APPROVAL_CHANNEL", "#approvals"),
        )


@dataclass
class TeamsConfig(BaseTransportConfig):
    """Configuration for Microsoft Teams approvals."""

    # Entra (Azure AD) client-credentials — the authenticated delivery path.
    # tenant_id is NOT a credential; it only scopes the token request. The
    # client_id + client_secret are what authenticate the app.
    tenant_id: Optional[str] = None  # Azure AD / Entra tenant (directory) ID
    client_id: Optional[str] = None  # Entra app registration client ID
    client_secret: Optional[str] = None  # Entra app client secret
    # Microsoft Graph target for the approval message (required for Graph path).
    team_id: Optional[str] = None  # Graph team (group) ID
    channel_id: Optional[str] = None  # Graph channel ID
    # Legacy incoming-webhook fallback (a secret URL, NOT Entra-authenticated).
    webhook_url: Optional[str] = None  # Incoming webhook URL

    @classmethod
    def from_env(cls) -> "TeamsConfig":
        """Create config from environment variables."""
        return cls(
            tenant_id=os.getenv("AZURE_TENANT_ID"),
            client_id=os.getenv("AZURE_CLIENT_ID"),
            client_secret=os.getenv("AZURE_CLIENT_SECRET"),
            team_id=os.getenv("TEAMS_TEAM_ID"),
            channel_id=os.getenv("TEAMS_CHANNEL_ID"),
            webhook_url=os.getenv("TEAMS_WEBHOOK_URL"),
        )


@dataclass
class GitHubConfig(BaseTransportConfig):
    """Configuration for GitHub-based approvals (PRs, Checks, Deployments)."""

    token: Optional[str] = None  # GitHub token (PAT or GitHub App)
    app_id: Optional[str] = None  # GitHub App ID
    app_private_key: Optional[str] = None  # GitHub App private key
    owner: Optional[str] = None  # Repository owner
    repo: Optional[str] = None  # Repository name
    base_url: str = "https://api.github.com"  # For GitHub Enterprise

    # PR-specific
    pr_number: Optional[int] = None  # PR number for approval
    required_reviewers: List[str] = field(default_factory=list)

    # Check-specific
    check_name: str = "Victor Workflow Approval"

    # Deployment-specific
    environment: Optional[str] = None  # Deployment environment name

    @classmethod
    def from_env(cls) -> "GitHubConfig":
        """Create config from environment variables."""
        return cls(
            token=os.getenv("GITHUB_TOKEN"),
            app_id=os.getenv("GITHUB_APP_ID"),
            owner=os.getenv("GITHUB_OWNER"),
            repo=os.getenv("GITHUB_REPO"),
            base_url=os.getenv("GITHUB_API_URL", "https://api.github.com"),
        )


@dataclass
class GitLabConfig(BaseTransportConfig):
    """Configuration for GitLab-based approvals (MRs, Pipelines)."""

    token: Optional[str] = None  # GitLab token
    project_id: Optional[str] = None  # Project ID or path
    base_url: str = "https://gitlab.com"  # For self-hosted

    # MR-specific
    mr_iid: Optional[int] = None  # Merge request IID
    required_approvers: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "GitLabConfig":
        """Create config from environment variables."""
        return cls(
            token=os.getenv("GITLAB_TOKEN"),
            project_id=os.getenv("GITLAB_PROJECT_ID"),
            base_url=os.getenv("GITLAB_URL", "https://gitlab.com"),
        )


@dataclass
class JiraConfig(BaseTransportConfig):
    """Configuration for Jira-based approvals."""

    base_url: Optional[str] = None  # Jira instance URL
    email: Optional[str] = None  # Jira user email
    api_token: Optional[str] = None  # Jira API token
    project_key: Optional[str] = None  # Project key
    issue_type: str = "Task"  # Issue type for approval tickets
    approval_transition: str = "Approve"  # Transition name for approval
    rejection_transition: str = "Reject"  # Transition name for rejection

    @classmethod
    def from_env(cls) -> "JiraConfig":
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("JIRA_URL"),
            email=os.getenv("JIRA_EMAIL"),
            api_token=os.getenv("JIRA_API_TOKEN"),
            project_key=os.getenv("JIRA_PROJECT_KEY"),
        )


@dataclass
class PagerDutyConfig(BaseTransportConfig):
    """Configuration for PagerDuty-based approvals."""

    api_key: Optional[str] = None  # PagerDuty API key
    routing_key: Optional[str] = None  # Integration routing key
    service_id: Optional[str] = None  # Service ID
    escalation_policy_id: Optional[str] = None  # Escalation policy

    @classmethod
    def from_env(cls) -> "PagerDutyConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("PAGERDUTY_API_KEY"),
            routing_key=os.getenv("PAGERDUTY_ROUTING_KEY"),
            service_id=os.getenv("PAGERDUTY_SERVICE_ID"),
        )


@dataclass
class TerraformCloudConfig(BaseTransportConfig):
    """Configuration for Terraform Cloud run approvals."""

    token: Optional[str] = None  # TFC API token
    organization: Optional[str] = None  # TFC organization
    workspace: Optional[str] = None  # TFC workspace
    base_url: str = "https://app.terraform.io"

    @classmethod
    def from_env(cls) -> "TerraformCloudConfig":
        """Create config from environment variables."""
        return cls(
            token=os.getenv("TF_TOKEN"),
            organization=os.getenv("TFC_ORGANIZATION"),
            workspace=os.getenv("TFC_WORKSPACE"),
        )


@dataclass
class CustomHookConfig(BaseTransportConfig):
    """Configuration for custom hook-based approvals."""

    send_hook: Optional[Callable] = None  # async def send(request) -> str
    poll_hook: Optional[Callable] = None  # async def poll(request_id) -> Optional[Response]
    cancel_hook: Optional[Callable] = None  # async def cancel(request_id) -> bool


# =============================================================================
# Transport Protocol and Base Class
# =============================================================================


@runtime_checkable
class HITLTransportProtocol(Protocol):
    """Protocol for HITL transport implementations."""

    @property
    def mode(self) -> HITLMode:
        """Return the HITL mode this transport handles."""
        ...

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Send an approval request.

        Args:
            request: The HITL request
            workflow_id: Workflow identifier

        Returns:
            External reference ID (e.g., message ID, PR number)
        """
        ...

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Poll for a response.

        Args:
            request_id: HITL request ID
            external_ref: External reference from send()
            timeout: Optional timeout override

        Returns:
            Response if available, None if still pending
        """
        ...

    async def cancel(self, request_id: str, external_ref: str) -> bool:
        """Cancel a pending approval request.

        Args:
            request_id: HITL request ID
            external_ref: External reference from send()

        Returns:
            True if successfully cancelled
        """
        ...


class BaseTransport(ABC):
    """Base class for HITL transports."""

    def __init__(self, config: BaseTransportConfig):
        self.config = config

    @property
    @abstractmethod
    def mode(self) -> HITLMode:
        """Return the HITL mode this transport handles."""
        pass

    @abstractmethod
    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Send an approval request."""
        pass

    @abstractmethod
    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Poll for a response."""
        pass

    async def cancel(self, request_id: str, external_ref: str) -> bool:
        """Cancel a pending request (default: no-op)."""
        logger.debug(f"Cancel not implemented for {self.mode}")
        return False

    async def wait_for_response(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Wait for a response with polling.

        Args:
            request_id: HITL request ID
            external_ref: External reference from send()
            timeout: Total timeout (uses config default if not specified)

        Returns:
            Response when available, or None if timeout
        """
        timeout = timeout or self.config.timeout
        poll_interval = self.config.poll_interval
        elapsed = 0.0

        while elapsed < timeout:
            response = await self.poll(request_id, external_ref, timeout - elapsed)
            if response:
                return response

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return None

    def _build_callback_urls(self, request_id: str) -> Dict[str, str]:
        """Build approve/reject/details URLs for the approval message.

        The public base URL comes from ``config.callback_url`` or the
        ``VICTOR_HITL_CALLBACK_URL`` env var; without one the buttons are omitted
        rather than rendered as dead links. When a signing secret is configured,
        each approve/reject link carries an HMAC token binding (request_id,
        action, expiry) so the decision cannot be forged, flipped or replayed.
        """
        base_url = (self.config.callback_url or os.getenv("VICTOR_HITL_CALLBACK_URL") or "").rstrip(
            "/"
        )
        if not base_url:
            return {}

        from victor.workflows.hitl_signing import get_signing_secret, sign_action

        secret = get_signing_secret()
        if not secret:
            logger.warning(
                "HITL callback links are unsigned; set VICTOR_HITL_SIGNING_SECRET to "
                "make approve/reject links tamper-proof."
            )

        def _action_url(action: str) -> str:
            url = f"{base_url}/hitl/respond/{request_id}?action={action}"
            if secret:
                url += f"&token={sign_action(request_id, action, secret=secret)}"
            return url

        return {
            "approve_url": _action_url("approve"),
            "reject_url": _action_url("reject"),
            "details_url": f"{base_url}/hitl/requests/{request_id}",
        }


# =============================================================================
# Messaging Transports
# =============================================================================


class EmailTransport(BaseTransport):
    """Email-based HITL transport."""

    def __init__(self, config: EmailConfig):
        super().__init__(config)
        self.email_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.EMAIL

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Send approval email."""
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        urls = self._build_callback_urls(request.request_id)

        # Build email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = self.email_config.subject_template.format(
            workflow_name=workflow_id,
            prompt=request.prompt[:50],
        )
        msg["From"] = self.email_config.from_address
        msg["To"] = ", ".join(self.email_config.to_addresses)

        # HTML body
        html_body = self._build_html_email(request, workflow_id, urls)
        msg.attach(MIMEText(html_body, "html"))

        # Send
        await aiosmtplib.send(
            msg,
            hostname=self.email_config.smtp_host,
            port=self.email_config.smtp_port,
            username=self.email_config.smtp_user,
            password=self.email_config.smtp_password,
            use_tls=self.email_config.smtp_use_tls,
        )

        logger.info(f"Sent approval email for {request.request_id}")
        return f"email:{request.request_id}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Poll is handled via callback URL - return None."""
        return None

    def _build_html_email(
        self,
        request: HITLRequest,
        workflow_id: str,
        urls: Dict[str, str],
    ) -> str:
        """Build HTML email body."""
        # Escape all caller/request-controlled values to prevent HTML/script
        # injection in the rendered email body.
        context_html = ""
        if request.context:
            context_items = "".join(
                f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
                for k, v in request.context.items()
            )
            context_html = f"<ul>{context_items}</ul>"

        safe_workflow_id = html.escape(str(workflow_id))
        safe_prompt = html.escape(str(request.prompt))

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Approval Required</h2>
            <p><strong>Workflow:</strong> {safe_workflow_id}</p>
            <p><strong>Request:</strong> {safe_prompt}</p>
            {context_html}
            <div style="margin: 20px 0;">
                <a href="{urls.get('approve_url', '#')}"
                   style="background: #28a745; color: white; padding: 10px 20px;
                          text-decoration: none; border-radius: 5px; margin-right: 10px;">
                    ✓ Approve
                </a>
                <a href="{urls.get('reject_url', '#')}"
                   style="background: #dc3545; color: white; padding: 10px 20px;
                          text-decoration: none; border-radius: 5px;">
                    ✗ Reject
                </a>
            </div>
            <p style="color: #666; font-size: 12px;">
                This request will timeout in {request.timeout} seconds.
            </p>
        </body>
        </html>
        """


class SlackTransport(BaseTransport):
    """Slack-based HITL transport with interactive messages."""

    def __init__(self, config: SlackConfig):
        super().__init__(config)
        self.slack_config = config
        self._message_ts: Dict[str, str] = {}  # request_id -> message_ts

    @property
    def mode(self) -> HITLMode:
        return HITLMode.SLACK

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Send Slack message with interactive buttons."""
        import aiohttp

        urls = self._build_callback_urls(request.request_id)

        # Build Block Kit message
        blocks = self._build_blocks(request, workflow_id, urls)

        payload = {
            "channel": self.slack_config.channel,
            "blocks": blocks,
            "text": f"Approval required: {request.prompt}",
        }

        # Add mentions
        if self.slack_config.mention_users:
            mentions = " ".join(f"<@{u}>" for u in self.slack_config.mention_users)
            payload["text"] = f"{mentions} {payload['text']}"

        headers = {}
        if self.slack_config.bot_token:
            headers["Authorization"] = f"Bearer {self.slack_config.bot_token}"
            url = "https://slack.com/api/chat.postMessage"
        else:
            url = self.slack_config.webhook_url

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json()
                if not data.get("ok", True):  # Webhook returns no 'ok' field
                    logger.error(f"Slack API error: {data}")

                message_ts = data.get("ts", request.request_id)
                self._message_ts[request.request_id] = message_ts

        logger.info(f"Sent Slack message for {request.request_id}")
        return message_ts

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Poll is handled via Slack interactivity - return None."""
        return None

    def _build_blocks(
        self,
        request: HITLRequest,
        workflow_id: str,
        urls: Dict[str, str],
    ) -> List[Dict]:
        """Build Slack Block Kit blocks."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🔔 Approval Required"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Workflow:* `{workflow_id}`\n*Request:* {request.prompt}",
                },
            },
        ]

        # Add context
        if request.context:
            context_text = "\n".join(f"• *{k}:* {v}" for k, v in request.context.items())
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": context_text}})

        # Add action buttons
        blocks.append(
            {
                "type": "actions",
                "block_id": f"hitl_{request.request_id}",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✓ Approve"},
                        "style": "primary",
                        "action_id": "approve",
                        "value": request.request_id,
                        "url": urls.get("approve_url"),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✗ Reject"},
                        "style": "danger",
                        "action_id": "reject",
                        "value": request.request_id,
                        "url": urls.get("reject_url"),
                    },
                ],
            }
        )

        # Add timeout notice
        req_id_short = request.request_id[:12]
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏱️ Timeout: {request.timeout}s | Request ID: `{req_id_short}...`",
                    }
                ],
            }
        )

        return blocks


# Microsoft Graph base for authenticated Teams delivery. Token acquisition lives
# in victor.core.identity (a TokenCredential), injected into this transport.
_GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"


class TeamsTransport(BaseTransport):
    """Microsoft Teams HITL transport.

    Authenticated delivery uses the Entra (Azure AD) **client-credentials**
    flow: ``client_id`` + ``client_secret`` are exchanged at the tenant's token
    endpoint for a Microsoft Graph access token, which is then used as a Bearer
    token to post an Adaptive Card approval message to a channel. The
    ``tenant_id`` is only the directory scope in the token URL — it is not a
    credential and is never sent as one.

    Token acquisition is delegated to an injected
    :class:`~victor.core.identity.TokenCredential` (Dependency Inversion). When
    none is supplied, one is built lazily from the config's Entra
    client-credentials. If neither a credential nor a Graph target
    (``team_id``/``channel_id``) is available, the transport falls back to a
    legacy Teams *incoming webhook* — a secret URL (no Entra auth), logged as such.
    """

    def __init__(
        self,
        config: TeamsConfig,
        credential: Optional["TokenCredential"] = None,
    ):
        super().__init__(config)
        self.teams_config = config
        self._message_ids: Dict[str, str] = {}  # request_id -> Graph message id
        # Injected token source (DIP); built lazily from config when not provided.
        self._credential = credential

    @property
    def mode(self) -> HITLMode:
        return HITLMode.TEAMS

    def _resolve_credential(self) -> Optional["TokenCredential"]:
        """Return the injected credential, or build one from config (cached).

        Returns None when neither an explicit credential nor Entra
        client-credentials config is available, so the caller can fall back to
        the webhook path.
        """
        if self._credential is not None:
            return self._credential
        cfg = self.teams_config
        if cfg.tenant_id and cfg.client_id and cfg.client_secret:
            from victor.core.identity import build_entra_credential

            self._credential = build_entra_credential(
                tenant_id=cfg.tenant_id,
                client_id=cfg.client_id,
                client_secret=cfg.client_secret,
            )
        return self._credential

    def _can_use_graph(self) -> bool:
        """True when a token credential AND a Graph target are available."""
        cfg = self.teams_config
        return bool(self._resolve_credential() and cfg.team_id and cfg.channel_id)

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Send a Teams approval card; returns the external message reference."""
        import aiohttp

        cfg = self.teams_config
        urls = self._build_callback_urls(request.request_id)
        card = self._build_adaptive_card(request, workflow_id, urls)

        # Preferred path: authenticated Microsoft Graph. The token comes from the
        # injected/derived TokenCredential — this transport never mints it itself.
        credential = self._resolve_credential()
        if credential is not None and cfg.team_id and cfg.channel_id:
            from victor.core.identity import GRAPH_DEFAULT_SCOPE

            token = (await credential.get_token(GRAPH_DEFAULT_SCOPE)).token
            url = f"{_GRAPH_API_BASE}/teams/{cfg.team_id}/channels/{cfg.channel_id}/messages"
            body = {
                "body": {
                    "contentType": "html",
                    "content": '<attachment id="approval-card"></attachment>',
                },
                "attachments": [
                    {
                        "id": "approval-card",
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": json.dumps(card),
                    }
                ],
            }
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    data = await resp.json()
                    if resp.status >= 300:
                        logger.error(f"Teams Graph API error ({resp.status}): {data}")
                        raise RuntimeError(f"Teams message send failed: {data.get('error', data)}")
                    external_ref = str(data.get("id", request.request_id))
            self._message_ids[request.request_id] = external_ref
            logger.info(f"Sent Teams approval card via Graph for {request.request_id}")
            return external_ref

        # Fallback: legacy incoming webhook — a secret URL, NOT Entra-authenticated.
        if cfg.webhook_url:
            payload = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ],
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(cfg.webhook_url, json=payload) as resp:
                    if resp.status >= 300:
                        logger.error(f"Teams webhook error ({resp.status})")
            logger.warning(
                "Sent Teams approval via incoming webhook (no Entra auth). Configure "
                "AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET + "
                "TEAMS_TEAM_ID/TEAMS_CHANNEL_ID for authenticated delivery."
            )
            return request.request_id

        raise ValueError(
            "Teams transport not configured. Provide Entra client-credentials "
            "(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET) plus a Graph "
            "target (TEAMS_TEAM_ID, TEAMS_CHANNEL_ID), or a TEAMS_WEBHOOK_URL fallback."
        )

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Responses arrive via the approve/reject callback URLs — nothing to poll."""
        return None

    def _build_adaptive_card(
        self,
        request: HITLRequest,
        workflow_id: str,
        urls: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build an Adaptive Card payload for the approval request."""
        body: List[Dict[str, Any]] = [
            {
                "type": "TextBlock",
                "size": "Large",
                "weight": "Bolder",
                "text": "🔔 Approval Required",
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Workflow", "value": workflow_id},
                    {"title": "Request ID", "value": f"{request.request_id[:12]}..."},
                    {"title": "Timeout", "value": f"{request.timeout}s"},
                ],
            },
            {"type": "TextBlock", "text": request.prompt, "wrap": True},
        ]
        if request.context:
            body.append(
                {
                    "type": "FactSet",
                    "facts": [
                        {"title": str(k), "value": str(v)} for k, v in request.context.items()
                    ],
                }
            )

        actions: List[Dict[str, Any]] = []
        if urls.get("approve_url"):
            actions.append(
                {
                    "type": "Action.OpenUrl",
                    "title": "✓ Approve",
                    "url": urls["approve_url"],
                }
            )
        if urls.get("reject_url"):
            actions.append(
                {
                    "type": "Action.OpenUrl",
                    "title": "✗ Reject",
                    "url": urls["reject_url"],
                }
            )

        return {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
            "actions": actions,
        }


class SMSTransport(BaseTransport):
    """SMS-based HITL transport via Twilio."""

    def __init__(self, config: SMSConfig):
        super().__init__(config)
        self.sms_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.SMS

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Send SMS via Twilio."""
        from twilio.rest import Client

        client = Client(
            self.sms_config.account_sid,
            self.sms_config.auth_token,
        )

        urls = self._build_callback_urls(request.request_id)
        message_body = self.sms_config.message_template.format(
            workflow_name=workflow_id,
            prompt=request.prompt[:100],
            approve_url=urls.get("approve_url", ""),
            reject_url=urls.get("reject_url", ""),
        )

        message_sids = []
        for to_number in self.sms_config.to_numbers:
            if not to_number.strip():
                continue
            message = client.messages.create(
                body=message_body,
                from_=self.sms_config.from_number,
                to=to_number.strip(),
            )
            message_sids.append(message.sid)

        logger.info(f"Sent {len(message_sids)} SMS for {request.request_id}")
        return ",".join(message_sids)

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Poll is handled via callback URL - return None."""
        return None


# =============================================================================
# SCM Transports (GitHub, GitLab, Bitbucket)
# =============================================================================


class GitHubPRTransport(BaseTransport):
    """GitHub PR-based HITL transport."""

    def __init__(self, config: GitHubConfig):
        super().__init__(config)
        self.github_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.GITHUB_PR

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Request PR review or add approval-required label."""
        import aiohttp

        headers = {
            "Authorization": f"token {self.github_config.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        owner = self.github_config.owner
        repo = self.github_config.repo
        pr_number = self.github_config.pr_number or request.context.get("pr_number")

        if not pr_number:
            raise ValueError("PR number required for GitHub PR approval")

        base_url = self.github_config.base_url

        async with aiohttp.ClientSession() as session:
            # Add a comment with approval request
            comment_url = f"{base_url}/repos/{owner}/{repo}/issues/{pr_number}/comments"
            comment_body = self._build_pr_comment(request, workflow_id)

            async with session.post(
                comment_url,
                json={"body": comment_body},
                headers=headers,
            ) as resp:
                data = await resp.json()
                comment_id = data.get("id")

            # Request review from specified reviewers
            if self.github_config.required_reviewers:
                review_path = f"/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers"
                review_url = f"{base_url}{review_path}"
                await session.post(
                    review_url,
                    json={"reviewers": self.github_config.required_reviewers},
                    headers=headers,
                )

        logger.info(f"Created GitHub PR comment for {request.request_id}")
        # Internal opaque reference string (not an HTTP response body).
        # nosemgrep: python.flask.security.audit.directly-returned-format-string.directly-returned-format-string
        return f"github:pr:{pr_number}:comment:{comment_id}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Poll for PR approval status."""
        import aiohttp

        # Parse external_ref: github:pr:123:comment:456
        parts = external_ref.split(":")
        if len(parts) < 3:
            return None

        pr_number = parts[2]
        owner = self.github_config.owner
        repo = self.github_config.repo
        base_url = self.github_config.base_url

        headers = {
            "Authorization": f"token {self.github_config.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        async with aiohttp.ClientSession() as session:
            # Get PR reviews
            reviews_url = f"{base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
            async with session.get(reviews_url, headers=headers) as resp:
                reviews = await resp.json()

            # Check for approval or rejection
            for review in reviews:
                state = review.get("state", "").upper()
                if state == "APPROVED":
                    return HITLResponse(
                        request_id=request_id,
                        status=HITLStatus.APPROVED,
                        approved=True,
                        reason=review.get("body", "Approved via GitHub PR review"),
                    )
                elif state == "CHANGES_REQUESTED":
                    return HITLResponse(
                        request_id=request_id,
                        status=HITLStatus.REJECTED,
                        approved=False,
                        reason=review.get("body", "Changes requested via GitHub PR review"),
                    )

        return None

    def _build_pr_comment(self, request: HITLRequest, workflow_id: str) -> str:
        """Build PR comment body."""
        context_md = ""
        if request.context:
            context_items = "\n".join(f"- **{k}:** {v}" for k, v in request.context.items())
            context_md = f"\n\n**Context:**\n{context_items}"

        return f"""
## 🔔 Workflow Approval Required

**Workflow:** `{workflow_id}`
**Request:** {request.prompt}
{context_md}

---

**Actions:**
- ✅ **Approve** this PR to continue the workflow
- ❌ **Request changes** to reject

*Request ID: `{request.request_id[:12]}...` | Timeout: {request.timeout}s*
"""


class GitHubCheckTransport(BaseTransport):
    """GitHub Check Run-based HITL transport for CI gates."""

    def __init__(self, config: GitHubConfig):
        super().__init__(config)
        self.github_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.GITHUB_CHECK

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Create a pending GitHub Check Run."""
        import aiohttp

        headers = {
            "Authorization": f"token {self.github_config.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        owner = self.github_config.owner
        repo = self.github_config.repo
        sha = request.context.get("sha", request.context.get("commit_sha"))

        if not sha:
            raise ValueError("Commit SHA required for GitHub Check")

        base_url = self.github_config.base_url
        check_runs_url = f"{base_url}/repos/{owner}/{repo}/check-runs"

        urls = self._build_callback_urls(request.request_id)

        payload = {
            "name": self.github_config.check_name,
            "head_sha": sha,
            "status": "in_progress",
            "output": {
                "title": f"Approval Required: {workflow_id}",
                "summary": request.prompt,
                "text": self._build_check_details(request, urls),
            },
            "actions": [
                {
                    "label": "Approve",
                    "description": "Approve this workflow step",
                    "identifier": f"approve:{request.request_id}",
                },
                {
                    "label": "Reject",
                    "description": "Reject this workflow step",
                    "identifier": f"reject:{request.request_id}",
                },
            ],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                check_runs_url,
                json=payload,
                headers=headers,
            ) as resp:
                data = await resp.json()
                check_run_id = data.get("id")

        logger.info(f"Created GitHub Check Run {check_run_id} for {request.request_id}")
        # Internal opaque reference string (not an HTTP response body).
        # nosemgrep: python.flask.security.audit.directly-returned-format-string.directly-returned-format-string
        return f"github:check:{check_run_id}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Poll for Check Run completion via webhook."""
        # Check runs are typically updated via GitHub webhooks
        # This would need a webhook handler to process check_run events
        return None

    def _build_check_details(self, request: HITLRequest, urls: Dict[str, str]) -> str:
        """Build check run details markdown."""
        context_md = ""
        if request.context:
            context_items = "\n".join(f"| {k} | {v} |" for k, v in request.context.items())
            context_md = f"\n| Key | Value |\n|-----|-------|\n{context_items}\n"

        return f"""
## Approval Details

{context_md}

### Actions

Use the buttons above or:
- [Approve]({urls.get('approve_url', '#')})
- [Reject]({urls.get('reject_url', '#')})

---
*Request ID: `{request.request_id}` | Timeout: {request.timeout}s*
"""


# =============================================================================
# Custom Hook Transport
# =============================================================================


class CustomHookTransport(BaseTransport):
    """Custom hook-based transport for user-defined integrations."""

    def __init__(self, config: CustomHookConfig):
        super().__init__(config)
        self.hook_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.CUSTOM_HOOK

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        """Call custom send hook."""
        if not self.hook_config.send_hook:
            raise ValueError("send_hook not configured")

        result = await self.hook_config.send_hook(request, workflow_id)
        return str(result)

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Call custom poll hook."""
        if not self.hook_config.poll_hook:
            return None

        return await self.hook_config.poll_hook(request_id, external_ref)

    async def cancel(self, request_id: str, external_ref: str) -> bool:
        """Call custom cancel hook."""
        if not self.hook_config.cancel_hook:
            return False

        return await self.hook_config.cancel_hook(request_id, external_ref)


# =============================================================================
# SCM / ticketing / incident transports
#
# These deliver the approval request to an external system and, where that
# system exposes a queryable approval state, poll it. For systems without a
# native approve/reject signal the response arrives via the signed callback
# links embedded in the message (see BaseTransport._build_callback_urls) and
# poll() returns None. The HTTP calls follow each provider's documented API;
# they are unit-tested against mocked responses and warrant integration testing
# against a real instance before production use.
# =============================================================================


def _approval_message(
    request: HITLRequest,
    workflow_id: str,
    urls: Dict[str, str],
) -> str:
    """Build a plain-text approval body with context and signed action links."""
    lines = [
        "🔔 Workflow approval required",
        f"Workflow: {workflow_id}",
        f"Request: {request.prompt}",
    ]
    for key, value in (request.context or {}).items():
        lines.append(f"- {key}: {value}")
    if urls.get("approve_url"):
        lines.append(f"Approve: {urls['approve_url']}")
    if urls.get("reject_url"):
        lines.append(f"Reject: {urls['reject_url']}")
    lines.append(f"Request ID: {request.request_id[:12]} | Timeout: {request.timeout}s")
    return "\n".join(lines)


class GitHubDeploymentTransport(BaseTransport):
    """GitHub deployment-protection HITL transport.

    GitHub natively notifies the environment's required reviewers when a run
    pauses for deployment, so ``send`` records the run reference and ``poll``
    watches the run's *pending deployments*: once our environment is no longer
    pending the gate has been resolved — a cancelled run means rejected, anything
    else approved. Requires ``context['run_id']`` (the workflow run awaiting the
    environment) and a configured ``environment``.
    """

    def __init__(self, config: GitHubConfig):
        super().__init__(config)
        self.github_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.GITHUB_DEPLOYMENT

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.github_config.token}",
            "Accept": "application/vnd.github+json",
        }

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        cfg = self.github_config
        run_id = request.context.get("run_id") if request.context else None
        env = cfg.environment or (request.context.get("environment") if request.context else None)
        if not run_id:
            raise ValueError("GitHub deployment approval requires context['run_id']")
        logger.info(f"GitHub deployment approval pending for run {run_id} (environment={env})")
        return f"github:deployment:{run_id}:{env or ''}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        import aiohttp

        parts = external_ref.split(":")
        run_id = parts[2] if len(parts) > 2 else None
        env = parts[3] if len(parts) > 3 else None
        if not run_id:
            return None
        cfg = self.github_config
        base = cfg.base_url
        async with aiohttp.ClientSession() as session:
            pend_url = (
                f"{base}/repos/{cfg.owner}/{cfg.repo}/actions/runs/{run_id}/pending_deployments"
            )
            async with session.get(pend_url, headers=self._headers()) as resp:
                pending = await resp.json()
            if isinstance(pending, list):
                still_pending = (
                    any((p.get("environment") or {}).get("name") == env for p in pending)
                    if env
                    else bool(pending)
                )
                if still_pending:
                    return None
            run_url = f"{base}/repos/{cfg.owner}/{cfg.repo}/actions/runs/{run_id}"
            async with session.get(run_url, headers=self._headers()) as resp:
                run = await resp.json()
        if run.get("conclusion") == "cancelled":
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason="Deployment rejected (run cancelled)",
            )
        return HITLResponse(
            request_id=request_id,
            status=HITLStatus.APPROVED,
            approved=True,
            reason="Deployment approved",
        )


class GitLabMRTransport(BaseTransport):
    """GitLab merge-request approval HITL transport.

    Posts an approval note (with signed callback links) on the MR, then polls the
    MR *approvals* endpoint for the native approve decision.
    """

    def __init__(self, config: GitLabConfig):
        super().__init__(config)
        self.gitlab_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.GITLAB_MR

    def _project(self) -> str:
        return urllib.parse.quote(str(self.gitlab_config.project_id), safe="")

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        import aiohttp

        cfg = self.gitlab_config
        mr_iid = cfg.mr_iid or (request.context.get("mr_iid") if request.context else None)
        if not (cfg.project_id and mr_iid):
            raise ValueError("GitLab MR approval requires project_id and mr_iid")
        urls = self._build_callback_urls(request.request_id)
        body = _approval_message(request, workflow_id, urls)
        url = f"{cfg.base_url}/api/v4/projects/{self._project()}/merge_requests/{mr_iid}/notes"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json={"body": body}, headers={"PRIVATE-TOKEN": cfg.token or ""}
            ) as resp:
                data = await resp.json()
        return f"gitlab:mr:{mr_iid}:note:{data.get('id', '')}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        import aiohttp

        parts = external_ref.split(":")
        mr_iid = parts[2] if len(parts) > 2 else self.gitlab_config.mr_iid
        cfg = self.gitlab_config
        url = f"{cfg.base_url}/api/v4/projects/{self._project()}/merge_requests/{mr_iid}/approvals"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"PRIVATE-TOKEN": cfg.token or ""}) as resp:
                data = await resp.json()
        if data.get("approved") or data.get("approved_by"):
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.APPROVED,
                approved=True,
                reason="Merge request approved",
            )
        return None


class GitLabPipelineTransport(BaseTransport):
    """GitLab pipeline manual-gate HITL transport.

    Treats a manual job as the gate: ``send`` records the job reference and
    ``poll`` reads the job status — ``success`` (played) is approval,
    ``failed``/``canceled`` rejection. Requires ``context['job_id']``.
    """

    def __init__(self, config: GitLabConfig):
        super().__init__(config)
        self.gitlab_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.GITLAB_PIPELINE

    def _project(self) -> str:
        return urllib.parse.quote(str(self.gitlab_config.project_id), safe="")

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        job_id = request.context.get("job_id") if request.context else None
        if not (self.gitlab_config.project_id and job_id):
            raise ValueError("GitLab pipeline approval requires project_id and context['job_id']")
        logger.info(f"GitLab pipeline manual gate pending for job {job_id}")
        return f"gitlab:pipeline:job:{job_id}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        import aiohttp

        job_id = external_ref.split(":")[-1]
        cfg = self.gitlab_config
        url = f"{cfg.base_url}/api/v4/projects/{self._project()}/jobs/{job_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"PRIVATE-TOKEN": cfg.token or ""}) as resp:
                data = await resp.json()
        status = data.get("status")
        if status == "success":
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.APPROVED,
                approved=True,
                reason="Manual job played",
            )
        if status in ("failed", "canceled"):
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason=f"Manual job {status}",
            )
        return None


class JiraTransport(BaseTransport):
    """Jira-issue approval HITL transport.

    Creates an approval issue whose description carries the context and signed
    callback links, then polls the issue status; a status name matching the
    configured approval / rejection transition resolves the request.
    """

    def __init__(self, config: JiraConfig):
        super().__init__(config)
        self.jira_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.JIRA

    def _auth(self):
        import aiohttp

        return aiohttp.BasicAuth(self.jira_config.email or "", self.jira_config.api_token or "")

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        import aiohttp

        cfg = self.jira_config
        if not (cfg.base_url and cfg.project_key):
            raise ValueError("Jira approval requires base_url and project_key")
        urls = self._build_callback_urls(request.request_id)
        fields = {
            "project": {"key": cfg.project_key},
            "summary": f"[Approval] {request.prompt[:120]}",
            "description": _approval_message(request, workflow_id, urls),
            "issuetype": {"name": cfg.issue_type},
        }
        url = f"{cfg.base_url}/rest/api/2/issue"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"fields": fields}, auth=self._auth()) as resp:
                data = await resp.json()
        return f"jira:issue:{data.get('key', '')}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        import aiohttp

        key = external_ref.split(":")[-1]
        cfg = self.jira_config
        url = f"{cfg.base_url}/rest/api/2/issue/{key}?fields=status"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, auth=self._auth()) as resp:
                data = await resp.json()
        status_name = (data.get("fields", {}).get("status", {}) or {}).get("name", "")
        lowered = status_name.lower()
        if cfg.approval_transition.lower() in lowered or lowered in (
            "approved",
            "done",
        ):
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.APPROVED,
                approved=True,
                reason=f"Jira issue {status_name}",
            )
        if cfg.rejection_transition.lower() in lowered or lowered in (
            "rejected",
            "won't do",
        ):
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason=f"Jira issue {status_name}",
            )
        return None


class PagerDutyTransport(BaseTransport):
    """PagerDuty HITL transport.

    Triggers an incident via the Events API v2 with the approve/reject callback
    links in the payload; PagerDuty has no native approve/reject, so the response
    arrives through the signed callback endpoint and ``poll`` returns None.
    """

    def __init__(self, config: PagerDutyConfig):
        super().__init__(config)
        self.pagerduty_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.PAGERDUTY

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        import aiohttp

        cfg = self.pagerduty_config
        if not cfg.routing_key:
            raise ValueError("PagerDuty approval requires a routing_key (Events API v2)")
        urls = self._build_callback_urls(request.request_id)
        dedup_key = f"victor-hitl-{request.request_id}"
        payload = {
            "routing_key": cfg.routing_key,
            "event_action": "trigger",
            "dedup_key": dedup_key,
            "payload": {
                "summary": f"Approval required: {request.prompt[:200]}",
                "source": f"victor:{workflow_id}",
                "severity": "warning",
                "custom_details": {
                    "context": request.context or {},
                    "approve_url": urls.get("approve_url"),
                    "reject_url": urls.get("reject_url"),
                    "request_id": request.request_id,
                },
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://events.pagerduty.com/v2/enqueue", json=payload
            ) as resp:
                if resp.status >= 300:
                    logger.error(f"PagerDuty enqueue error ({resp.status})")
        return f"pagerduty:incident:{dedup_key}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Responses arrive via the signed callback links — nothing to poll."""
        return None


class TerraformCloudTransport(BaseTransport):
    """Terraform Cloud run-approval HITL transport.

    Polls a run awaiting confirmation: ``applied``/``planned_and_finished`` is
    approval, ``discarded``/``canceled``/``errored`` rejection. To actually apply
    or discard the run, use TFC or its run apply/discard API. Requires
    ``context['run_id']``.
    """

    def __init__(self, config: TerraformCloudConfig):
        super().__init__(config)
        self.tfc_config = config

    @property
    def mode(self) -> HITLMode:
        return HITLMode.TERRAFORM_CLOUD

    async def send(self, request: HITLRequest, workflow_id: str) -> str:
        run_id = request.context.get("run_id") if request.context else None
        if not run_id:
            raise ValueError("Terraform Cloud approval requires context['run_id']")
        logger.info(f"Terraform Cloud run {run_id} awaiting confirmation")
        return f"tfc:run:{run_id}"

    async def poll(
        self,
        request_id: str,
        external_ref: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        import aiohttp

        run_id = external_ref.split(":")[-1]
        cfg = self.tfc_config
        headers = {
            "Authorization": f"Bearer {cfg.token}",
            "Content-Type": "application/vnd.api+json",
        }
        url = f"{cfg.base_url}/api/v2/runs/{run_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                data = await resp.json()
        status = (data.get("data", {}).get("attributes", {}) or {}).get("status", "")
        if status in ("applied", "planned_and_finished"):
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.APPROVED,
                approved=True,
                reason=f"Run {status}",
            )
        if status in ("discarded", "canceled", "errored"):
            return HITLResponse(
                request_id=request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason=f"Run {status}",
            )
        return None


# =============================================================================
# Transport Registry
# =============================================================================


# Registry of transport classes by mode
_TRANSPORT_REGISTRY: Dict[HITLMode, Type[BaseTransport]] = {
    HITLMode.EMAIL: EmailTransport,
    HITLMode.SMS: SMSTransport,
    HITLMode.SLACK: SlackTransport,
    HITLMode.TEAMS: TeamsTransport,
    HITLMode.GITHUB_PR: GitHubPRTransport,
    HITLMode.GITHUB_CHECK: GitHubCheckTransport,
    HITLMode.GITHUB_DEPLOYMENT: GitHubDeploymentTransport,
    HITLMode.GITLAB_MR: GitLabMRTransport,
    HITLMode.GITLAB_PIPELINE: GitLabPipelineTransport,
    HITLMode.JIRA: JiraTransport,
    HITLMode.PAGERDUTY: PagerDutyTransport,
    HITLMode.TERRAFORM_CLOUD: TerraformCloudTransport,
    HITLMode.CUSTOM_HOOK: CustomHookTransport,
}

# Config classes by mode
_CONFIG_REGISTRY: Dict[HITLMode, Type[BaseTransportConfig]] = {
    HITLMode.EMAIL: EmailConfig,
    HITLMode.SMS: SMSConfig,
    HITLMode.SLACK: SlackConfig,
    HITLMode.TEAMS: TeamsConfig,
    HITLMode.GITHUB_PR: GitHubConfig,
    HITLMode.GITHUB_CHECK: GitHubConfig,
    HITLMode.GITHUB_DEPLOYMENT: GitHubConfig,
    HITLMode.GITLAB_MR: GitLabConfig,
    HITLMode.GITLAB_PIPELINE: GitLabConfig,
    HITLMode.JIRA: JiraConfig,
    HITLMode.PAGERDUTY: PagerDutyConfig,
    HITLMode.TERRAFORM_CLOUD: TerraformCloudConfig,
    HITLMode.CUSTOM_HOOK: CustomHookConfig,
}


def register_transport(
    mode: HITLMode,
    transport_class: Type[BaseTransport],
    config_class: Optional[Type[BaseTransportConfig]] = None,
) -> None:
    """Register a custom transport.

    Args:
        mode: The HITL mode to register for
        transport_class: The transport class
        config_class: Optional config class
    """
    _TRANSPORT_REGISTRY[mode] = transport_class
    if config_class:
        _CONFIG_REGISTRY[mode] = config_class


def get_transport(
    mode: HITLMode,
    config: Optional[BaseTransportConfig] = None,
) -> BaseTransport:
    """Get a transport instance for a mode.

    Args:
        mode: The HITL mode
        config: Transport configuration (creates default from env if None)

    Returns:
        Transport instance

    Raises:
        ValueError: If mode not registered
    """
    if mode not in _TRANSPORT_REGISTRY:
        raise ValueError(
            f"No transport registered for mode {mode}. "
            f"Available: {list(_TRANSPORT_REGISTRY.keys())}"
        )

    transport_class = _TRANSPORT_REGISTRY[mode]

    if config is None:
        config_class = _CONFIG_REGISTRY.get(mode, BaseTransportConfig)
        if hasattr(config_class, "from_env"):
            config = config_class.from_env()
        else:
            config = config_class()

    return transport_class(config)


def list_available_transports() -> Dict[HITLMode, str]:
    """List available transports with descriptions.

    Returns:
        Dict of mode -> description
    """
    return {
        mode: cls.__doc__.split("\n")[0] if cls.__doc__ else str(mode)
        for mode, cls in _TRANSPORT_REGISTRY.items()
    }


__all__ = [
    # Base classes
    "BaseTransportConfig",
    "BaseTransport",
    "HITLTransportProtocol",
    # Config classes
    "EmailConfig",
    "SMSConfig",
    "SlackConfig",
    "TeamsConfig",
    "GitHubConfig",
    "GitLabConfig",
    "JiraConfig",
    "PagerDutyConfig",
    "TerraformCloudConfig",
    "CustomHookConfig",
    # Transport classes
    "EmailTransport",
    "SMSTransport",
    "SlackTransport",
    "TeamsTransport",
    "GitHubPRTransport",
    "GitHubCheckTransport",
    "GitHubDeploymentTransport",
    "GitLabMRTransport",
    "GitLabPipelineTransport",
    "JiraTransport",
    "PagerDutyTransport",
    "TerraformCloudTransport",
    "CustomHookTransport",
    # Registry functions
    "register_transport",
    "get_transport",
    "list_available_transports",
]
