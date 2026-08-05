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

"""HTTP/API tool for making HTTP requests and testing APIs.

Features:
- All HTTP methods (GET, POST, PUT, PATCH, DELETE)
- Headers and authentication
- Request/response inspection
- JSON, form data, file uploads
- Response validation (test mode)
- Performance metrics
"""

from victor.core.json_utils import json_dumps
import time
from typing import Any, Dict, Optional

import httpx

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.formatters import format_http_response


def _safe_response_headers(response: Any) -> Dict[str, Any]:
    """Return response headers as a plain dict, tolerating lightweight mocks."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return {}
    try:
        return dict(headers)
    except Exception:
        return {}


def _safe_response_status(response: Any) -> str:
    """Return a best-effort reason/status string for a response-like object."""
    reason = getattr(response, "reason_phrase", None)
    if isinstance(reason, str) and reason:
        return reason
    return str(getattr(response, "status_code", ""))


def _safe_response_body(response: Any, response_json: Any) -> Any:
    """Return parsed JSON or a safe text fallback for a response-like object."""
    if response_json is not None:
        return response_json
    text = getattr(response, "text", "")
    if isinstance(text, str):
        return text[:1000]
    return ""


def _normalize_response_json(payload: Any) -> Any:
    """Return payload only when it behaves like real JSON data.

    Test doubles often expose a ``.json()`` method that returns another mock.
    That should not be treated as parsed JSON because downstream formatters may
    try to serialize it.
    """
    try:
        json_dumps(payload)
    except (TypeError, ValueError):
        return None
    return payload


async def _make_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    json_body: Optional[Dict[str, Any]],
    data: Optional[Dict[str, Any]],
    auth: Optional[str],
    follow_redirects: bool,
    timeout: int,
) -> Dict[str, Any]:
    """Internal: Make HTTP request and return response data."""
    method_upper = method.upper()
    request_headers = headers or {}

    # Handle authentication
    if auth:
        if auth.startswith("Bearer ") or auth.startswith("Basic "):
            request_headers["Authorization"] = auth

    start_time = time.time()

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects) as client:
        response = await client.request(
            method=method_upper,
            url=url,
            headers=request_headers,
            params=params,
            json=json_body,
            data=data,
        )

    duration = time.time() - start_time

    # Parse response
    try:
        response_json = _normalize_response_json(response.json())
    except (ValueError, TypeError, AttributeError):
        response_json = None

    return {
        "response": response,
        "response_json": response_json,
        "duration": duration,
        "method": method_upper,
    }


async def _http_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    json_body: Optional[Dict[str, Any]],
    data: Optional[Dict[str, Any]],
    auth: Optional[str],
    follow_redirects: bool,
    timeout: int,
) -> Dict[str, Any]:
    """Internal: Standard HTTP request mode."""
    result = await _make_request(
        method, url, headers, params, json_body, data, auth, follow_redirects, timeout
    )
    response = result["response"]
    response_headers = _safe_response_headers(response)
    response_status = _safe_response_status(response)
    response_body = _safe_response_body(response, result["response_json"])

    # Build response data
    response_data = {
        "success": True,
        "status_code": response.status_code,
        "status": response_status,
        "headers": response_headers,
        "body": response_body,
        "duration_ms": int(result["duration"] * 1000),
        "url": str(response.url),
    }

    # Use unified formatter system
    formatted_result = format_http_response(response_data)

    return {
        "success": True,
        "status_code": response.status_code,
        "status": response_status,
        "headers": response_headers,
        "body": response_body,
        "duration_ms": int(result["duration"] * 1000),
        "url": str(response.url),
        "formatted_output": formatted_result.content,
        "contains_markup": formatted_result.contains_markup,
    }


async def _http_test(
    method: str,
    url: str,
    headers: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    json_body: Optional[Dict[str, Any]],
    data: Optional[Dict[str, Any]],
    auth: Optional[str],
    follow_redirects: bool,
    timeout: int,
    expected_status: Optional[int],
) -> Dict[str, Any]:
    """Internal: API test mode with validation."""
    result = await _make_request(
        method, url, headers, params, json_body, data, auth, follow_redirects, timeout
    )
    response = result["response"]

    # Validate
    validations = []
    all_passed = True

    # Check status code
    if expected_status is not None:
        passed = response.status_code == expected_status
        all_passed = all_passed and passed
        validations.append(
            {
                "test": "Status code",
                "expected": expected_status,
                "actual": response.status_code,
                "passed": passed,
            }
        )

    return {
        "success": all_passed,
        "url": str(response.url),
        "method": result["method"],
        "status_code": response.status_code,
        "duration_ms": int(result["duration"] * 1000),
        "validations": validations,
        "all_passed": all_passed,
        "error": "" if all_passed else "Some validations failed",
    }


@tool(
    cost_tier=CostTier.MEDIUM,
    category="web",
    priority=Priority.MEDIUM,  # Task-specific API testing
    access_mode=AccessMode.NETWORK,  # Makes external HTTP requests
    danger_level=DangerLevel.SAFE,  # No local side effects
    keywords=["http", "request", "api", "rest", "test", "endpoint", "health"],
)
async def http(
    method: str,
    url: str,
    mode: str = "request",
    headers: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    auth: Optional[str] = None,
    follow_redirects: bool = True,
    timeout: int = 30,
    expected_status: Optional[int] = None,
) -> Dict[str, Any]:
    """Send HTTP requests (GET/POST/PUT/DELETE) and inspect responses.

    Modes:
    - "request": Standard HTTP request (default)
    - "test": API testing with validation

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS).
        url: Request URL.
        mode: Operation mode - "request" (default) or "test".
        headers: Request headers (optional).
        params: Query parameters (optional).
        json: JSON body as dictionary (optional).
        data: Form data as dictionary (optional).
        auth: Authentication - 'Bearer TOKEN' or 'Basic USER:PASS' (optional).
        follow_redirects: Follow redirects (default: true).
        timeout: Request timeout in seconds (default: 30).
        expected_status: Expected HTTP status code (for "test" mode).

    Returns:
        For "request" mode:
        - success: Whether request succeeded
        - status_code: HTTP status code
        - status: Status reason phrase
        - headers: Response headers
        - body: Parsed JSON or text response
        - duration_ms: Request duration in milliseconds
        - url: Final URL (after redirects)

        For "test" mode:
        - success: Whether all validations passed
        - url: Final URL tested
        - method: HTTP method used
        - status_code: HTTP status code received
        - duration_ms: Request duration in milliseconds
        - validations: List of validation results
        - all_passed: Whether all validations passed
        - error: Error message if request failed
    """
    if not url:
        return {"success": False, "error": "Missing required parameter: url"}

    mode_lower = mode.lower().strip()

    try:
        if mode_lower == "request":
            return await _http_request(
                method,
                url,
                headers,
                params,
                json,
                data,
                auth,
                follow_redirects,
                timeout,
            )
        elif mode_lower == "test":
            return await _http_test(
                method,
                url,
                headers,
                params,
                json,
                data,
                auth,
                follow_redirects,
                timeout,
                expected_status,
            )
        else:
            return {
                "success": False,
                "error": f"Unknown mode '{mode}'. Use 'request' or 'test'.",
            }

    except httpx.TimeoutException:
        return {"success": False, "error": f"Request timed out after {timeout} seconds"}
    except Exception as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
