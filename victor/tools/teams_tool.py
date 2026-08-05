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

"""Microsoft Teams integration tool.

This tool provides:
1. Sending messages to channels
2. Searching for messages
3. Listing teams and channels
4. Creating channels

Authentication (app-only, Entra/Azure AD client-credentials) — the tool mints
its own Microsoft Graph token from these environment variables:
- AZURE_CLIENT_ID     (alias: TEAMS_CLIENT_ID)     — Entra app client ID
- AZURE_CLIENT_SECRET (alias: TEAMS_CLIENT_SECRET)  — Entra app client secret
- AZURE_TENANT_ID     (alias: TEAMS_TENANT_ID)      — Entra tenant (directory) ID

Alternatively, a pre-acquired Graph token may be supplied out-of-band as
``context["teams_access_token"]`` (e.g. from a delegated/SSO flow). Token
acquisition is delegated to ``victor.core.identity`` (a TokenCredential).

Setup:
1. Register an app in Azure AD (https://portal.azure.com)
2. Add Microsoft Graph API permissions:
   - ChannelMessage.Send
   - Channel.ReadBasic.All
   - Team.ReadBasic.All
   - ChannelMessage.Read.All (for search)
3. Grant admin consent for the permissions
4. Create a client secret
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

_GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"


def _get_teams_access_token(context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Get Teams access token from execution context.

    Args:
        context: Tool execution context

    Returns:
        Access token if available in context, None otherwise
    """
    if context:
        return context.get("teams_access_token")
    return None


def is_teams_configured(context: Optional[Dict[str, Any]] = None) -> bool:
    """True when Teams is usable: a context token, or Entra creds in the env.

    Used as the tool's availability check (must stay synchronous), so it only
    confirms credentials are *present* — the token itself is minted lazily at
    call time by :func:`_resolve_access_token`.
    """
    if _get_teams_access_token(context):
        return True
    import os

    tenant = os.getenv("AZURE_TENANT_ID") or os.getenv("TEAMS_TENANT_ID")
    client = os.getenv("AZURE_CLIENT_ID") or os.getenv("TEAMS_CLIENT_ID")
    secret = os.getenv("AZURE_CLIENT_SECRET") or os.getenv("TEAMS_CLIENT_SECRET")
    return bool(tenant and client and secret)


def _get_headers(access_token: str) -> Dict[str, str]:
    """Build Graph API request headers for a bearer ``access_token``."""
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


async def _resolve_access_token(
    context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve a Microsoft Graph token.

    Prefers a token supplied out-of-band via ``context["teams_access_token"]``;
    otherwise mints one via the Entra client-credentials
    :class:`~victor.core.identity.TokenCredential` built from the environment.
    Returns ``None`` when neither is available.
    """
    token = _get_teams_access_token(context)
    if token:
        return token

    from victor.core.identity import GRAPH_DEFAULT_SCOPE, graph_credential_from_env

    credential = graph_credential_from_env()
    if credential is None:
        return None
    return (await credential.get_token(GRAPH_DEFAULT_SCOPE)).token


@tool(
    cost_tier=CostTier.MEDIUM,
    category="collaboration",  # Same category as slack for grouping
    priority=Priority.MEDIUM,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.LOW,
    signature_params=["query"],
    stages=["completion", "execution"],  # Same stages as slack
    task_types=["action", "search"],
    execution_category="network",
    keywords=[
        "teams",
        "microsoft",
        "message",
        "chat",
        "channel",
        "notification",
        "collaboration",
    ],
    mandatory_keywords=[
        "send teams message",
        "post to teams",
        "teams channel",
        "microsoft teams",
    ],
    availability_check=is_teams_configured,  # Only available when configured
)
async def teams(
    operation: str,
    team_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    text: Optional[str] = None,
    query: Optional[str] = None,
    channel_name: Optional[str] = None,
    channel_description: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send messages and manage channels in Microsoft Teams.

    Args:
        operation: The operation to perform: 'send_message', 'search_messages',
                   'list_teams', 'list_channels', 'create_channel'.
        team_id: The team ID for operations that require it.
        channel_id: The channel ID for operations that require it.
        text: The text of the message for 'send_message'.
        query: The query for searching messages for 'search_messages'.
        channel_name: Name for 'create_channel' operation.
        channel_description: Description for 'create_channel' operation.
        context: Tool execution context containing teams_access_token.

    Returns:
        A dictionary with the result of the operation.
    """
    if not HTTPX_AVAILABLE:
        return {
            "success": False,
            "error": "httpx not installed. Install with: pip install httpx",
        }

    access_token = await _resolve_access_token(context)
    if not access_token:
        return {
            "success": False,
            "error": (
                "Teams not configured. Supply a token via context['teams_access_token'], "
                "or set AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET "
                "(TEAMS_* aliases accepted)."
            ),
        }

    headers = _get_headers(access_token)

    try:
        async with httpx.AsyncClient() as client:
            if operation == "send_message":
                return await _send_message(client, headers, team_id, channel_id, text)

            elif operation == "search_messages":
                return await _search_messages(client, headers, query)

            elif operation == "list_teams":
                return await _list_teams(client, headers)

            elif operation == "list_channels":
                return await _list_channels(client, headers, team_id)

            elif operation == "create_channel":
                return await _create_channel(
                    client, headers, team_id, channel_name, channel_description
                )

            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}. "
                    f"Valid operations: send_message, search_messages, "
                    f"list_teams, list_channels, create_channel",
                }

    except httpx.HTTPStatusError as e:
        logger.error(f"[teams] HTTP error during operation '{operation}': {e}")
        return {"success": False, "error": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"[teams] Unexpected error during operation '{operation}': {e}")
        return {"success": False, "error": str(e)}


async def _send_message(
    client: "httpx.AsyncClient",
    headers: Dict[str, str],
    team_id: Optional[str],
    channel_id: Optional[str],
    text: Optional[str],
) -> Dict[str, Any]:
    """Send a message to a Teams channel."""
    if not team_id or not channel_id or not text:
        return {
            "success": False,
            "error": "Missing required parameters: team_id, channel_id, and text",
        }

    logger.info(f"[teams] Sending message to team '{team_id}' channel '{channel_id}'")

    url = f"{_GRAPH_API_BASE}/teams/{team_id}/channels/{channel_id}/messages"
    response = await client.post(
        url,
        headers=headers,
        json={
            "body": {
                "content": text,
            }
        },
    )
    response.raise_for_status()
    data = response.json()

    return {
        "success": True,
        "message_id": data.get("id"),
        "created_datetime": data.get("createdDateTime"),
        "message": "Message sent successfully",
    }


async def _search_messages(
    client: "httpx.AsyncClient",
    headers: Dict[str, str],
    query: Optional[str],
) -> Dict[str, Any]:
    """Search for messages across Teams."""
    if not query:
        return {"success": False, "error": "Missing required parameter: query"}

    logger.info(f"[teams] Searching messages with query='{query}'")

    # Microsoft Graph search API
    url = f"{_GRAPH_API_BASE}/search/query"
    response = await client.post(
        url,
        headers=headers,
        json={
            "requests": [
                {
                    "entityTypes": ["chatMessage"],
                    "query": {"queryString": query},
                    "from": 0,
                    "size": 10,
                }
            ]
        },
    )
    response.raise_for_status()
    data = response.json()

    # Parse search results
    results: List[Dict[str, Any]] = []
    hits_containers = data.get("value", [])
    for container in hits_containers:
        hits = container.get("hitsContainers", [])
        for hit_container in hits:
            for hit in hit_container.get("hits", []):
                resource = hit.get("resource", {})
                results.append(
                    {
                        "id": resource.get("id"),
                        "summary": hit.get("summary", "")[:200],
                        "created_datetime": resource.get("createdDateTime"),
                        "from": resource.get("from", {})
                        .get("user", {})
                        .get("displayName", "Unknown"),
                    }
                )

    return {"success": True, "results": results, "count": len(results)}


async def _list_teams(
    client: "httpx.AsyncClient",
    headers: Dict[str, str],
) -> Dict[str, Any]:
    """List all teams the app has access to."""
    logger.info("[teams] Listing teams")

    url = f"{_GRAPH_API_BASE}/groups?$filter=resourceProvisioningOptions/Any(x:x eq 'Team')"
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    teams_data = data.get("value", [])
    results: List[Dict[str, Any]] = [
        {
            "id": team.get("id"),
            "name": team.get("displayName"),
            "description": team.get("description", ""),
            "mail": team.get("mail", ""),
        }
        for team in teams_data[:20]  # Limit to 20 teams
    ]

    return {"success": True, "results": results, "count": len(results)}


async def _list_channels(
    client: "httpx.AsyncClient",
    headers: Dict[str, str],
    team_id: Optional[str],
) -> Dict[str, Any]:
    """List channels in a team."""
    if not team_id:
        return {"success": False, "error": "Missing required parameter: team_id"}

    logger.info(f"[teams] Listing channels for team '{team_id}'")

    url = f"{_GRAPH_API_BASE}/teams/{team_id}/channels"
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    channels_data = data.get("value", [])
    results: List[Dict[str, Any]] = [
        {
            "id": channel.get("id"),
            "name": channel.get("displayName"),
            "description": channel.get("description", ""),
            "membership_type": channel.get("membershipType", "standard"),
        }
        for channel in channels_data[:20]  # Limit to 20 channels
    ]

    return {"success": True, "results": results, "count": len(results)}


async def _create_channel(
    client: "httpx.AsyncClient",
    headers: Dict[str, str],
    team_id: Optional[str],
    channel_name: Optional[str],
    channel_description: Optional[str],
) -> Dict[str, Any]:
    """Create a new channel in a team."""
    if not team_id or not channel_name:
        return {
            "success": False,
            "error": "Missing required parameters: team_id and channel_name",
        }

    logger.info(f"[teams] Creating channel '{channel_name}' in team '{team_id}'")

    url = f"{_GRAPH_API_BASE}/teams/{team_id}/channels"
    response = await client.post(
        url,
        headers=headers,
        json={
            "displayName": channel_name,
            "description": channel_description or "",
            "membershipType": "standard",
        },
    )
    response.raise_for_status()
    data = response.json()

    return {
        "success": True,
        "channel_id": data.get("id"),
        "name": data.get("displayName"),
        "message": f"Channel '{channel_name}' created successfully",
    }
