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

"""Jira integration tool.

This tool provides:
1. Issue creation
2. Issue retrieval
3. Issue search
4. Commenting on issues
"""

import logging
from typing import Any, Dict, Optional

try:
    from jira import JIRA, JIRAError

    JIRA_AVAILABLE = True
except ImportError:
    JIRA = None  # type: ignore
    JIRAError = Exception  # type: ignore
    JIRA_AVAILABLE = False

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


def _get_jira_client(context: Optional[Dict[str, Any]] = None) -> Optional["JIRA"]:
    """Get Jira client from execution context.

    Args:
        context: Tool execution context

    Returns:
        JIRA client if available in context, None otherwise
    """
    if context:
        return context.get("jira_client")
    return None


def is_jira_configured(context: Optional[Dict[str, Any]] = None) -> bool:
    """Check if Jira client is configured and connected."""
    return _get_jira_client(context) is not None


@tool(
    cost_tier=CostTier.MEDIUM,
    category="jira",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.LOW,
    signature_params=["jql"],
    stages=["planning", "execution"],
    task_types=["action", "search"],
    execution_category="network",
    keywords=["jira", "issue", "ticket", "bug", "task", "sprint", "backlog", "story"],
    mandatory_keywords=[
        "search issues",
        "find tickets",
        "create ticket",
        "jira search",
    ],
)
async def jira(
    operation: str,
    jql: Optional[str] = None,
    issue_key: Optional[str] = None,
    summary: Optional[str] = None,
    project: Optional[str] = None,
    issue_type: str = "Task",
    description: Optional[str] = None,
    comment: Optional[str] = None,
    max_results: int = 10,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create, update, and search Jira issues and projects.

    Args:
        operation: The operation to perform: 'search_issues', 'create_issue', 'get_issue', 'add_comment'.
        jql: The JQL query for searching issues (required for search_issues).
        issue_key: The key of the issue (e.g., 'PROJ-123') for 'get_issue' and 'add_comment'.
        summary: The summary/title of the issue for 'create_issue'.
        project: The project key for 'create_issue' (e.g., 'PROJ').
        issue_type: The type of issue for 'create_issue' (default: 'Task'). Options: Task, Bug, Story, Epic.
        description: Optional description for 'create_issue'.
        comment: The comment to add for 'add_comment'.
        max_results: The maximum number of results to return for 'search_issues' (default: 10).
        context: Tool execution context containing jira_client.

    Returns:
        A dictionary with the result of the operation.
    """
    if not JIRA_AVAILABLE:
        return {
            "success": False,
            "error": "Jira library not installed. Install with: pip install jira",
        }

    jira_client = _get_jira_client(context)
    if not jira_client:
        return {
            "success": False,
            "error": "Jira client not configured. Provide jira_client in context.",
        }

    try:
        if operation == "search_issues":
            if not jql:
                return {"success": False, "error": "Missing required parameter: jql"}
            logger.info(f"[jira] Searching issues with jql='{jql}'")
            issues = jira_client.search_issues(jql, maxResults=max_results)
            results = [
                {
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": issue.fields.status.name,
                    "assignee": getattr(issue.fields.assignee, "displayName", "Unassigned"),
                    "priority": getattr(issue.fields.priority, "name", "None"),
                }
                for issue in issues
            ]
            return {"success": True, "results": results, "count": len(results)}

        elif operation == "create_issue":
            if not project or not summary:
                return {
                    "success": False,
                    "error": "Missing required parameters: project and summary",
                }
            logger.info(f"[jira] Creating issue in project '{project}' with summary '{summary}'")
            issue_dict: Dict[str, Any] = {
                "project": {"key": project},
                "summary": summary,
                "issuetype": {"name": issue_type},
            }
            if description:
                issue_dict["description"] = description
            new_issue = jira_client.create_issue(fields=issue_dict)
            return {
                "success": True,
                "key": new_issue.key,
                "summary": new_issue.fields.summary,
                "url": (
                    f"{context.get('jira_server', '')}/browse/{new_issue.key}" if context else ""
                ),
            }

        elif operation == "get_issue":
            if not issue_key:
                return {
                    "success": False,
                    "error": "Missing required parameter: issue_key",
                }
            logger.info(f"[jira] Getting issue '{issue_key}'")
            issue = jira_client.issue(issue_key)
            comments = []
            if hasattr(issue.fields, "comment") and issue.fields.comment:
                comments = [
                    {
                        "author": getattr(comment.author, "displayName", "Unknown"),
                        "body": comment.body,
                        "created": str(comment.created),
                    }
                    for comment in issue.fields.comment.comments[:5]  # Limit to 5 most recent
                ]
            return {
                "success": True,
                "key": issue.key,
                "summary": issue.fields.summary,
                "status": issue.fields.status.name,
                "description": issue.fields.description or "",
                "assignee": getattr(issue.fields.assignee, "displayName", "Unassigned"),
                "reporter": getattr(issue.fields.reporter, "displayName", "Unknown"),
                "priority": getattr(issue.fields.priority, "name", "None"),
                "comments": comments,
                "url": (f"{context.get('jira_server', '')}/browse/{issue.key}" if context else ""),
            }

        elif operation == "add_comment":
            if not issue_key or not comment:
                return {
                    "success": False,
                    "error": "Missing required parameters: issue_key and comment",
                }
            logger.info(f"[jira] Adding comment to issue '{issue_key}'")
            jira_client.add_comment(issue_key, comment)
            return {"success": True, "message": f"Comment added to {issue_key}"}

        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}. "
                f"Valid operations: search_issues, create_issue, get_issue, add_comment",
            }

    except JIRAError as e:
        logger.error(f"[jira] Error during operation '{operation}': {e}")
        return {"success": False, "error": str(getattr(e, "text", str(e)))}
    except Exception as e:
        logger.error(f"[jira] Unexpected error during operation '{operation}': {e}")
        return {"success": False, "error": str(e)}
