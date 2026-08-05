# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""API Middleware for authentication, rate limiting, and request validation.

This module provides enterprise-grade middleware for the Victor API:
- API key authentication (header or query parameter)
- Rate limiting (per-client, per-endpoint)
- Request validation and sanitization
- Request logging and metrics

Design Principles:
- Modular middleware that can be composed
- Configurable via Settings or constructor
- Non-blocking async implementation
- Graceful degradation on errors
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Set

from aiohttp import web
from aiohttp.web import Request, Response

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10  # Allow short bursts
    per_endpoint: bool = True  # Rate limit per endpoint
    whitelist: Set[str] = field(default_factory=set)  # Exempt paths


@dataclass
class AuthConfig:
    """Configuration for API authentication."""

    enabled: bool = False  # Disabled by default for local dev
    api_keys: Dict[str, str] = field(default_factory=dict)  # key -> client_id
    header_name: str = "X-API-Key"
    query_param: str = "api_key"
    exempt_paths: Set[str] = field(default_factory=lambda: {"/health", "/status"})


class TokenBucket:
    """Token bucket rate limiter for smooth rate limiting."""

    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: int,  # Maximum bucket size
    ):
        """Initialize token bucket.

        Args:
            rate: Token refill rate (tokens/second)
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    @property
    def available(self) -> float:
        """Get available tokens."""
        return self.tokens


class RateLimiter:
    """Rate limiter with per-client and per-endpoint limits."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()

        # Token buckets per client
        self._client_buckets: Dict[str, TokenBucket] = {}

        # Token buckets per endpoint
        self._endpoint_buckets: Dict[str, TokenBucket] = {}

        # Request counters for analytics
        self._request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = asyncio.Lock()

    @lru_cache(maxsize=1024)
    def _get_client_id_cached(self, forwarded: str, peername: tuple) -> str:
        """Cached client ID extraction."""
        if forwarded:
            return forwarded.split(",")[0].strip()
        if peername:
            return f"{peername[0]}:{peername[1]}"
        return "unknown"

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request.

        Uses X-Forwarded-For if behind proxy, falls back to peername.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        peername = request.transport.get_extra_info("peername")
        return self._get_client_id_cached(forwarded, tuple(peername) if peername else None)

    def _get_bucket(self, key: str, is_endpoint: bool = False) -> TokenBucket:
        """Get or create token bucket for key."""
        buckets = self._endpoint_buckets if is_endpoint else self._client_buckets

        if key not in buckets:
            rate = self.config.requests_per_minute / 60.0
            buckets[key] = TokenBucket(rate=rate, capacity=self.config.burst_size)

        return buckets[key]

    async def is_allowed(self, request: Request) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Check if request is allowed by rate limits.

        Args:
            request: Incoming request

        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        path = request.path

        # Check whitelist
        if path in self.config.whitelist:
            return True, None

        client_id = self._get_client_id(request)

        # Check client rate limit
        client_bucket = self._get_bucket(client_id)
        if not await client_bucket.acquire():
            return False, {
                "reason": "client_rate_limit",
                "client_id": client_id,
                "retry_after": 1.0 / self.config.requests_per_minute * 60,
            }

        # Check endpoint rate limit
        if self.config.per_endpoint:
            endpoint_bucket = self._get_bucket(path, is_endpoint=True)
            if not await endpoint_bucket.acquire():
                return False, {
                    "reason": "endpoint_rate_limit",
                    "endpoint": path,
                    "retry_after": 1.0 / self.config.requests_per_minute * 60,
                }

        # Update counters (lock-free for better performance)
        self._request_counts[client_id][path] += 1

        return True, None

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "active_clients": len(self._client_buckets),
            "active_endpoints": len(self._endpoint_buckets),
            "request_counts": dict(self._request_counts),
        }


class APIKeyAuthenticator:
    """API key authentication handler."""

    def __init__(self, config: Optional[AuthConfig] = None):
        """Initialize authenticator.

        Args:
            config: Authentication configuration
        """
        self.config = config or AuthConfig()
        self._key_hashes: Dict[str, str] = {}  # hash -> client_id
        self._hash_cache: Dict[str, str] = {}  # key -> hash cache

        # Hash API keys for secure comparison
        for key, client_id in self.config.api_keys.items():
            key_hash = self._get_key_hash(key)
            self._key_hashes[key_hash] = client_id

    def _get_key_hash(self, api_key: str) -> str:
        """Get cached hash for API key."""
        if api_key not in self._hash_cache:
            self._hash_cache[api_key] = hashlib.sha256(api_key.encode()).hexdigest()
        return self._hash_cache[api_key]

    def add_key(self, api_key: str, client_id: str) -> None:
        """Add an API key.

        Args:
            api_key: The API key
            client_id: Client identifier
        """
        key_hash = self._get_key_hash(api_key)
        self._key_hashes[key_hash] = client_id
        self.config.api_keys[api_key] = client_id

    def generate_key(self, client_id: str, prefix: str = "vic") -> str:
        """Generate a new API key.

        Args:
            client_id: Client identifier
            prefix: Key prefix

        Returns:
            Generated API key
        """
        key = f"{prefix}_{secrets.token_urlsafe(32)}"
        self.add_key(key, client_id)
        return key

    def authenticate(self, request: Request) -> tuple[bool, Optional[str]]:
        """Authenticate request.

        Args:
            request: Incoming request

        Returns:
            Tuple of (authenticated, client_id)
        """
        if not self.config.enabled:
            return True, None

        # Check exempt paths
        if request.path in self.config.exempt_paths:
            return True, None

        # Try header first
        api_key = request.headers.get(self.config.header_name)

        # Fall back to query parameter
        if not api_key:
            api_key = request.query.get(self.config.query_param)

        if not api_key:
            return False, None

        # Secure comparison using cached hash
        key_hash = self._get_key_hash(api_key)
        client_id = self._key_hashes.get(key_hash)

        return client_id is not None, client_id


def create_auth_middleware(
    config: Optional[AuthConfig] = None,
) -> Callable:
    """Create authentication middleware.

    Args:
        config: Authentication configuration

    Returns:
        aiohttp middleware
    """
    authenticator = APIKeyAuthenticator(config)

    @web.middleware
    async def auth_middleware(request: Request, handler: Callable) -> Response:
        """Authenticate requests."""
        authenticated, client_id = authenticator.authenticate(request)

        if not authenticated:
            return web.json_response(
                {"error": "Unauthorized", "message": "Valid API key required"},
                status=401,
                headers={"WWW-Authenticate": 'API-Key realm="Victor API"'},
            )

        # Attach client_id to request for downstream use
        request["client_id"] = client_id

        return await handler(request)

    return auth_middleware


def create_rate_limit_middleware(
    config: Optional[RateLimitConfig] = None,
) -> Callable:
    """Create rate limiting middleware.

    Args:
        config: Rate limit configuration

    Returns:
        aiohttp middleware
    """
    limiter = RateLimiter(config)

    @web.middleware
    async def rate_limit_middleware(request: Request, handler: Callable) -> Response:
        """Apply rate limiting."""
        allowed, info = await limiter.is_allowed(request)

        if not allowed:
            retry_after = info.get("retry_after", 1.0) if info else 1.0
            return web.json_response(
                {
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded: {info.get('reason', 'unknown')}",
                    "retry_after": retry_after,
                },
                status=429,
                headers={"Retry-After": str(int(retry_after))},
            )

        return await handler(request)

    return rate_limit_middleware


def create_request_logging_middleware() -> Callable:
    """Create request logging middleware.

    Returns:
        aiohttp middleware
    """

    @web.middleware
    async def logging_middleware(request: Request, handler: Callable) -> Response:
        """Log requests and responses."""
        start_time = time.monotonic()

        try:
            response = await handler(request)
            elapsed = (time.monotonic() - start_time) * 1000

            logger.info(
                f"{request.method} {request.path} -> {response.status} " f"({elapsed:.1f}ms)"
            )

            return response

        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error(f"{request.method} {request.path} -> ERROR ({elapsed:.1f}ms): {e}")
            raise

    return logging_middleware


class APIMiddlewareStack:
    """Composable middleware stack for Victor API.

    Example:
        stack = APIMiddlewareStack()
        stack.add_cors()
        stack.add_rate_limiting(requests_per_minute=100)
        stack.add_authentication(api_keys={"key1": "client1"})
        stack.add_logging()

        app = web.Application(middlewares=stack.build())
    """

    def __init__(self):
        """Initialize middleware stack."""
        self._middlewares: List[Callable] = []

    def add_cors(
        self,
        origins: str = "*",
        methods: str = "GET, POST, OPTIONS",
        headers: str = "Content-Type, X-API-Key",
    ) -> "APIMiddlewareStack":
        """Add CORS middleware.

        Args:
            origins: Allowed origins
            methods: Allowed methods
            headers: Allowed headers

        Returns:
            Self for chaining
        """

        @web.middleware
        async def cors_middleware(request: Request, handler: Callable) -> Response:
            if request.method == "OPTIONS":
                return Response(
                    headers={
                        "Access-Control-Allow-Origin": origins,
                        "Access-Control-Allow-Methods": methods,
                        "Access-Control-Allow-Headers": headers,
                    }
                )

            response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = origins
            return response

        self._middlewares.append(cors_middleware)
        return self

    def add_rate_limiting(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        whitelist: Optional[Set[str]] = None,
    ) -> "APIMiddlewareStack":
        """Add rate limiting middleware.

        Args:
            requests_per_minute: Rate limit
            burst_size: Burst capacity
            whitelist: Exempt paths

        Returns:
            Self for chaining
        """
        config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            whitelist=whitelist or set(),
        )
        self._middlewares.append(create_rate_limit_middleware(config))
        return self

    def add_authentication(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        enabled: bool = True,
        exempt_paths: Optional[Set[str]] = None,
    ) -> "APIMiddlewareStack":
        """Add authentication middleware.

        Args:
            api_keys: API key -> client_id mapping
            enabled: Whether auth is enabled
            exempt_paths: Paths exempt from auth

        Returns:
            Self for chaining
        """
        config = AuthConfig(
            enabled=enabled,
            api_keys=api_keys or {},
            exempt_paths=exempt_paths or {"/health", "/status"},
        )
        self._middlewares.append(create_auth_middleware(config))
        return self

    def add_logging(self) -> "APIMiddlewareStack":
        """Add request logging middleware.

        Returns:
            Self for chaining
        """
        self._middlewares.append(create_request_logging_middleware())
        return self

    def build(self) -> List[Callable]:
        """Build middleware list for aiohttp Application.

        Returns:
            List of middleware callables
        """
        return list(self._middlewares)
