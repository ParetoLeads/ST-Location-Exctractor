"""
Caching utilities for KMZ Location Scraper.

Provides in-memory caching for OSM queries and administrative hierarchy lookups.
"""
import time
import hashlib
import json
from typing import Dict, Any, Optional, Tuple, List
from config import config


class CacheEntry:
    """Represents a single cache entry with TTL."""
    
    def __init__(self, value: Any, ttl_seconds: int):
        """Initialize cache entry.
        
        Args:
            value: The cached value
            ttl_seconds: Time to live in seconds
        """
        self.value = value
        self.expires_at = time.time() + ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() > self.expires_at


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, ttl_seconds: Optional[int] = None):
        """Initialize cache.
        
        Args:
            ttl_seconds: Default TTL in seconds (defaults to config value)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self.ttl_seconds = ttl_seconds or config.CACHE_TTL_SECONDS
    
    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Create a cache key from arguments."""
        # Create a deterministic key from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        
        if entry.is_expired():
            del self._cache[key]
            return None
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (uses default if not provided)
        """
        ttl = ttl_seconds or self.ttl_seconds
        self._cache[key] = CacheEntry(value, ttl)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def remove_expired(self) -> int:
        """Remove expired entries and return count of removed entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)
    
    def size(self) -> int:
        """Get the number of entries in cache."""
        return len(self._cache)


# Global cache instance (only created if caching is enabled)
_cache_instance: Optional[SimpleCache] = None


def get_cache() -> Optional[SimpleCache]:
    """Get the global cache instance if caching is enabled.
    
    Returns:
        Cache instance or None if caching is disabled
    """
    global _cache_instance
    if not config.ENABLE_CACHE:
        return None
    
    if _cache_instance is None:
        _cache_instance = SimpleCache()
    
    return _cache_instance


def cache_osm_query(
    query_type: str,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float
) -> Optional[List[Dict[str, Any]]]:
    """
    Get cached OSM query result.
    
    Args:
        query_type: Type of query (primary, additional, special)
        min_lat: Minimum latitude
        min_lon: Minimum longitude
        max_lat: Maximum latitude
        max_lon: Maximum longitude
        
    Returns:
        Cached result or None
    """
    cache = get_cache()
    if cache is None:
        return None
    
    key = f"osm_query:{query_type}:{min_lat}:{min_lon}:{max_lat}:{max_lon}"
    return cache.get(key)


def set_osm_query_cache(
    query_type: str,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    result: List[Dict[str, Any]]
) -> None:
    """
    Cache OSM query result.
    
    Args:
        query_type: Type of query (primary, additional, special)
        min_lat: Minimum latitude
        min_lon: Minimum longitude
        max_lat: Maximum latitude
        max_lon: Maximum longitude
        result: Query result to cache
    """
    cache = get_cache()
    if cache is None:
        return
    
    key = f"osm_query:{query_type}:{min_lat}:{min_lon}:{max_lat}:{max_lon}"
    cache.set(key, result)


def cache_hierarchy(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """
    Get cached administrative hierarchy for a location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Cached hierarchy or None
    """
    cache = get_cache()
    if cache is None:
        return None
    
    # Round coordinates to reduce cache misses for nearby points
    lat_rounded = round(latitude, 4)
    lon_rounded = round(longitude, 4)
    key = f"hierarchy:{lat_rounded}:{lon_rounded}"
    return cache.get(key)


def set_hierarchy_cache(
    latitude: float,
    longitude: float,
    hierarchy: Dict[str, Any]
) -> None:
    """
    Cache administrative hierarchy for a location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        hierarchy: Hierarchy data to cache
    """
    cache = get_cache()
    if cache is None:
        return
    
    # Round coordinates to reduce cache misses for nearby points
    lat_rounded = round(latitude, 4)
    lon_rounded = round(longitude, 4)
    key = f"hierarchy:{lat_rounded}:{lon_rounded}"
    cache.set(key, hierarchy)
