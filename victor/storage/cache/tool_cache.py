import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from victor.storage.cache.tiered_cache import TieredCache
from victor.storage.cache.config import CacheConfig


def _hash_args(args: Dict[str, Any]) -> str:
    """Create a stable hash for tool arguments."""
    try:
        data = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        data = str(args)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class ToolCache:
    """Thin wrapper for caching tool results with per-tool TTL and allowlist."""

    def __init__(
        self,
        ttl: int,
        allowlist: Optional[list[str]] = None,
        cache_config: Optional[CacheConfig] = None,
        cache_eviction_learner: Optional[Any] = None,
    ) -> None:
        self.ttl = ttl
        self.allowlist = set(allowlist or [])

        # Get CacheEvictionLearner from RLCoordinator if not provided
        if cache_eviction_learner is None:
            try:
                from victor.framework.rl.coordinator import get_rl_coordinator

                coordinator = get_rl_coordinator()
                cache_eviction_learner = coordinator.get_learner("cache_eviction")
            except Exception:
                pass  # Graceful fallback to non-RL caching

        self.cache = TieredCache(
            cache_config or CacheConfig(),
            cache_eviction_learner=cache_eviction_learner,
        )
        # track keys by touched paths for selective invalidation
        self._path_index: Dict[str, set[str]] = {}

    def _key(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
        return tool_name, _hash_args(args)

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Any]:
        if tool_name not in self.allowlist:
            return None
        name, hashed = self._key(tool_name, args)
        # Use RL-aware get if learner is available
        return self.cache.get_with_rl(hashed, namespace=name, tool_name=tool_name)

    def set(self, tool_name: str, args: Dict[str, Any], value: Any) -> None:
        if tool_name not in self.allowlist:
            return
        name, hashed = self._key(tool_name, args)
        # Use RL-aware set with tool metadata
        self.cache.set_with_tool(hashed, value, tool_name=tool_name, namespace=name, ttl=self.ttl)
        # index by path if provided
        path = args.get("path") or args.get("root")
        paths = args.get("paths")
        path_values = []
        if path:
            path_values.append(path)
        if isinstance(paths, list):
            path_values.extend(paths)
        # Also include root if provided (for code_search)
        if "root" in args and args["root"] not in path_values:
            path_values.append(args["root"])
        key_ref = f"{name}:{hashed}"
        for p in path_values:
            self._path_index.setdefault(str(p), set()).add(key_ref)

    def clear_all(self) -> None:
        """Clear all cached tool results."""
        self.cache.clear()
        self._path_index.clear()

    def close(self) -> None:
        """Release this cache's connections without deleting persisted entries."""
        self.cache.close(strict=True)

    def clear_namespaces(self, namespaces: list[str]) -> None:
        """Clear specific tool namespaces."""
        for ns in namespaces:
            self.cache.clear(namespace=ns)

    def invalidate_paths(self, paths: list[str]) -> None:
        """Invalidate cache entries associated with given paths."""
        keys_to_clear = set()
        for p in paths:
            keys_to_clear.update(self._path_index.get(str(p), set()))
        if not keys_to_clear:
            # fallback: clear key namespaces
            self.clear_namespaces(list(self.allowlist))
            return
        for full_key in keys_to_clear:
            if ":" in full_key:
                ns, key = full_key.split(":", 1)
                self.cache.delete(key, namespace=ns)
        for p in paths:
            self._path_index.pop(str(p), None)

    def invalidate_by_tool(self, tool_name: str) -> None:
        """Invalidate all cached results for a specific tool.

        Args:
            tool_name: Name of the tool whose cache should be cleared
        """
        self.cache.clear(namespace=tool_name)
        # Clean up path index entries for this tool
        keys_to_remove = []
        for path, key_refs in self._path_index.items():
            updated_refs = {ref for ref in key_refs if not ref.startswith(f"{tool_name}:")}
            if updated_refs != key_refs:
                if updated_refs:
                    self._path_index[path] = updated_refs
                else:
                    keys_to_remove.append(path)
        for path in keys_to_remove:
            del self._path_index[path]
