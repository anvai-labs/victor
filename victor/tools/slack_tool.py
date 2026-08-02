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

"""Slack integration tool.

This tool provides:
1. Sending messages to channels
2. Searching for messages
3. Listing channels
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    SLACK_AVAILABLE = True
except ImportError:
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore
    SLACK_AVAILABLE = False

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


def _get_slack_client(
    context: Optional[Dict[str, Any]] = None,
) -> Optional["WebClient"]:
    """Get Slack client from execution context.

    Args:
        context: Tool execution context

    Returns:
        WebClient if available in context, None otherwise
    """
    if context:
        return context.get("slack_client")
    return None


def is_slack_configured(context: Optional[Dict[str, Any]] = None) -> bool:
    """Check if Slack client is configured and connected."""
    return _get_slack_client(context) is not None


@tool(
    cost_tier=CostTier.MEDIUM,
    category="collaboration",  # Grouped with teams_tool
    priority=Priority.MEDIUM,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.LOW,
    signature_params=["query"],
    stages=["completion", "execution"],
    task_types=["action", "search"],
    execution_category="network",
    keywords=["slack", "message", "chat", "channel", "notification", "team"],
    mandatory_keywords=[
        "send message",
        "search chat",
        "post to slack",
        "slack channel",
    ],
    availability_check=is_slack_configured,  # Only available when configured
)
async def slack(
    operation: str,
    channel: Optional[str] = None,
    text: Optional[str] = None,
    query: Optional[str] = None,
    thread_ts: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send messages and search conversations in Slack channels.

    Args:
        operation: The operation to perform: 'send_message', 'search_messages', 'list_channels'.
        channel: The channel to send the message to (e.g., '#general' or channel ID).
        text: The text of the message for 'send_message'.
        query: The query for searching messages for 'search_messages'.
        thread_ts: Optional thread timestamp to reply in a thread.
        context: Tool execution context containing slack_client.

    Returns:
        A dictionary with the result of the operation.
    """
    if not SLACK_AVAILABLE:
        return {
            "success": False,
            "error": "Slack SDK not installed. Install with: pip install slack_sdk",
        }

    slack_client = _get_slack_client(context)
    if not slack_client:
        return {
            "success": False,
            "error": "Slack client not configured. Provide slack_client in context.",
        }

    try:
        if operation == "send_message":
            if not channel or not text:
                return {
                    "success": False,
                    "error": "Missing required parameters: channel and text",
                }
            logger.info(f"[slack] Sending message to channel '{channel}'")
            kwargs: Dict[str, Any] = {"channel": channel, "text": text}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            response = slack_client.chat_postMessage(**kwargs)
            if response.get("ok"):
                return {
                    "success": True,
                    "ts": response.get("ts"),
                    "channel": response.get("channel"),
                    "message": "Message sent successfully",
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                }

        elif operation == "search_messages":
            if not query:
                return {"success": False, "error": "Missing required parameter: query"}
            logger.info(f"[slack] Searching messages with query='{query}'")
            response = slack_client.search_messages(query=query)
            if response.get("ok"):
                matches = response.get("messages", {}).get("matches", [])
                results: List[Dict[str, Any]] = [
                    {
                        "ts": message.get("ts"),
                        "text": message.get("text", "")[:200],  # Truncate long messages
                        "username": message.get("username", "Unknown"),
                        "channel": message.get("channel", {}).get("name", "Unknown"),
                    }
                    for message in matches[:10]  # Limit to 10 results
                ]
                return {"success": True, "results": results, "count": len(results)}
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                }

        elif operation == "list_channels":
            logger.info("[slack] Listing channels")
            response = slack_client.conversations_list(types="public_channel,private_channel")
            if response.get("ok"):
                channels_data = response.get("channels", [])
                results: List[Dict[str, Any]] = [
                    {
                        "id": ch.get("id"),
                        "name": ch.get("name"),
                        "is_private": ch.get("is_private", False),
                        "num_members": ch.get("num_members", 0),
                    }
                    for ch in channels_data[:20]  # Limit to 20 channels
                ]
                return {"success": True, "results": results, "count": len(results)}
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                }

        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}. "
                f"Valid operations: send_message, search_messages, list_channels",
            }

    except SlackApiError as e:
        logger.error(
            f"[slack] Error during operation '{operation}': {e.response.get('error', str(e))}"
        )
        return {"success": False, "error": e.response.get("error", str(e))}
    except Exception as e:
        logger.error(f"[slack] Unexpected error during operation '{operation}': {e}")
        return {"success": False, "error": str(e)}
