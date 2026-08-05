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

"""Demo: Using Qwen3-Embedding:8b with ChromaDB for code semantic search.

This example demonstrates how to:
1. Set up ChromaDB with Qwen3-Embedding:8b (MTEB #1 model)
2. Index Python files from a codebase
3. Perform semantic search queries
4. Get statistics about the index

Requirements:
- Ollama running locally: ollama serve
- Qwen3 model pulled: ollama pull qwen3-embedding:8b
- Dependencies: pip install chromadb httpx

MTEB Performance:
- Qwen3-Embedding:8b: 70.58 score (#1 multilingual)
- 40K context window (great for large files)
- 4096 embedding dimensions (high quality)
- 100+ languages supported
"""

import asyncio

from victor.storage.vector_stores import EmbeddingConfig, EmbeddingRegistry


async def main():
    """Demo Qwen3-Embedding:8b with ChromaDB."""

    print("=" * 80)
    print("🚀 Qwen3-Embedding:8b Demo - Production-Grade Code Embeddings")
    print("=" * 80)
    print()

    # Configuration for Qwen3-Embedding:8b with ChromaDB
    config = EmbeddingConfig(
        # Vector Store: ChromaDB
        vector_store="chromadb",
        persist_directory="~/.victor/embeddings/qwen3_demo",
        distance_metric="cosine",
        # Embedding Model: Qwen3-Embedding:8b (via Ollama)
        embedding_model_type="ollama",
        embedding_model_name="qwen3-embedding:8b",
        embedding_api_key="http://localhost:11434",  # Ollama server URL
        # Extra configuration
        extra_config={
            "collection_name": "qwen3_demo",
            "dimension": 4096,  # Qwen3 produces 4096-dim embeddings
            "batch_size": 8,  # Lower for large model (adjust based on RAM)
        },
    )

    print("📋 Configuration:")
    print(f"   Vector Store: {config.vector_store}")
    print(f"   Embedding Model: {config.embedding_model_name} ({config.embedding_model_type})")
    print(f"   Dimension: {config.extra_config['dimension']}")
    print(f"   Persist Directory: {config.persist_directory}")
    print()

    # Initialize provider via registry (auto-detects installed backends)
    provider = EmbeddingRegistry.create(config)
    await provider.initialize()
    print()

    # Sample Python code snippets to index
    documents = [
        {
            "id": "auth.py:authenticate_user",
            "content": """def authenticate_user(username: str, password: str) -> Optional[User]:
    '''Authenticate user with username and password.

    Args:
        username: User's username
        password: User's password (will be hashed)

    Returns:
        User object if authentication succeeds, None otherwise
    '''
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.hashed_password):
        return user
    return None""",
            "metadata": {
                "file_path": "app/auth.py",
                "symbol_name": "authenticate_user",
                "symbol_type": "function",
                "line_number": 15,
            },
        },
        {
            "id": "database.py:create_connection",
            "content": """async def create_connection(database_url: str) -> AsyncEngine:
    '''Create async database connection with connection pooling.

    Args:
        database_url: Database connection URL

    Returns:
        SQLAlchemy async engine with connection pool
    '''
    engine = create_async_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )
    return engine""",
            "metadata": {
                "file_path": "app/database.py",
                "symbol_name": "create_connection",
                "symbol_type": "function",
                "line_number": 8,
            },
        },
        {
            "id": "cache.py:cache_decorator",
            "content": """def cache_decorator(ttl: int = 3600):
    '''Decorator to cache function results with TTL.

    Args:
        ttl: Time to live in seconds (default 1 hour)

    Returns:
        Decorated function with caching
    '''
    def decorator(func):
        cache = {}
        timestamps = {}

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            now = time.time()

            # Check if cached and not expired
            if key in cache and now - timestamps[key] < ttl:
                return cache[key]

            # Call function and cache result
            result = await func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = now
            return result

        return wrapper
    return decorator""",
            "metadata": {
                "file_path": "app/utils/cache.py",
                "symbol_name": "cache_decorator",
                "symbol_type": "function",
                "line_number": 22,
            },
        },
        {
            "id": "api.py:get_user_endpoint",
            "content": """@app.get("/api/users/{user_id}")
async def get_user_endpoint(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    '''Get user by ID endpoint.

    Args:
        user_id: User ID to retrieve
        db: Database session
        current_user: Currently authenticated user

    Returns:
        User data as JSON

    Raises:
        HTTPException: If user not found or unauthorized
    '''
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check authorization
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")

    return UserResponse.from_orm(user)""",
            "metadata": {
                "file_path": "app/api/users.py",
                "symbol_name": "get_user_endpoint",
                "symbol_type": "function",
                "line_number": 45,
            },
        },
        {
            "id": "middleware.py:rate_limit_middleware",
            "content": """async def rate_limit_middleware(request: Request, call_next):
    '''Rate limiting middleware using token bucket algorithm.

    Limits requests per IP address to prevent abuse.
    Uses Redis for distributed rate limiting.
    '''
    client_ip = request.client.host
    redis = request.app.state.redis

    # Token bucket parameters
    max_tokens = 100
    refill_rate = 10  # tokens per second

    # Get current tokens
    key = f"rate_limit:{client_ip}"
    tokens = await redis.get(key)

    if tokens is None:
        tokens = max_tokens
    else:
        tokens = float(tokens)

    # Refill tokens based on time elapsed
    now = time.time()
    last_refill = await redis.get(f"{key}:timestamp")

    if last_refill:
        elapsed = now - float(last_refill)
        tokens = min(max_tokens, tokens + elapsed * refill_rate)

    # Check if request allowed
    if tokens >= 1:
        tokens -= 1
        await redis.set(key, tokens)
        await redis.set(f"{key}:timestamp", now)
        response = await call_next(request)
        return response
    else:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")""",
            "metadata": {
                "file_path": "app/middleware/rate_limit.py",
                "symbol_name": "rate_limit_middleware",
                "symbol_type": "function",
                "line_number": 12,
            },
        },
    ]

    # Index documents
    print("📝 Indexing documents...")
    await provider.index_documents(documents)
    print()

    # Get index statistics
    stats = await provider.get_stats()
    print("📊 Index Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Embedding model: {stats['embedding_model_name']}")
    print(f"   Model type: {stats['embedding_model_type']}")
    print(f"   Dimension: {stats['dimension']}")
    print(f"   Distance metric: {stats['distance_metric']}")
    print()

    # Perform semantic search queries
    queries = [
        "How to authenticate a user with username and password?",
        "Database connection with connection pooling",
        "Implement caching with time to live",
        "Rate limiting middleware to prevent abuse",
        "REST API endpoint to get user information",
    ]

    print("🔍 Semantic Search Results:")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        print(f'\n{i}. Query: "{query}"')
        print("-" * 80)

        results = await provider.search_similar(query, limit=3)

        for rank, result in enumerate(results, 1):
            print(f"\n   Rank #{rank} (Score: {result.score:.4f})")
            print(f"   File: {result.file_path}:{result.symbol_name}")
            print(f"   Type: {result.metadata.get('symbol_type', 'unknown')}")
            print(f"   Line: {result.line_number}")

            # Show snippet of content (first 150 chars)
            snippet = result.content.split("\n")[0][:150]
            print(f"   Code: {snippet}...")

    print("\n" + "=" * 80)
    print("✅ Demo completed successfully!")
    print()
    print("💡 Next Steps:")
    print("   1. Index your actual codebase using CodebaseIndex")
    print("   2. Integrate semantic search into your AI assistant")
    print("   3. Try other models: snowflake-arctic-embed2, bge-m3")
    print("   4. Scale to production with larger datasets")
    print()

    # Clean up
    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
