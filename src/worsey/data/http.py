"""Shared HTTP helper: GET through the disk cache, with a polite retry on 429/5xx."""

from __future__ import annotations

import time

import requests

from .cache import cached_call


def get_json(source: str, base: str, path: str, params: dict | None = None, headers: dict | None = None,
             retries: int = 3, timeout: float = 30.0, cache_tag: str | None = None, force_network: bool = False,
             cache: bool = True):
    """GET through the disk cache. cache_tag (e.g. the current hour) is part of the cache key but never
    sent, so live quotes refresh once per tag while replays stay offline."""
    def fetch():
        last = None
        hdrs = headers() if callable(headers) else (headers or {})  # lazy: cache hits and replays never need a key
        for attempt in range(retries):
            r = requests.get(base + path, params=params, headers=hdrs, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = r
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        last.raise_for_status()

    if not cache:   # large payloads the caller stores compactly itself (e.g. the ETF price panel)
        from .cache import OfflineCacheMiss, offline
        if offline() and not force_network:
            raise OfflineCacheMiss(f"{source} {path} not cached (cache=False) and WORSEY_OFFLINE=1")
        return fetch()
    # headers carry auth, so they are deliberately not part of the cache key
    key_params = dict(params or {}, _tag=cache_tag) if cache_tag else params
    return cached_call(source, path, key_params, fetch, force_network=force_network)
