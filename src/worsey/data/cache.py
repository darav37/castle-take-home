"""Disk cache for every external API call (SPEC §11, §11a).

Every response is stored as JSON under cache/<source>/<hash>.json, keyed on the
source, path and params. Auth never enters the key or the stored file, so the
cache can be inspected without leaking secrets.

Set WORSEY_OFFLINE=1 to forbid network access: a cache miss then raises instead
of calling out. Tests, the eval and demo replays run this way.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

CACHE_ROOT = Path(os.environ.get("WORSEY_CACHE_DIR", Path(__file__).resolve().parents[3] / "cache"))


class OfflineCacheMiss(RuntimeError):
    """Raised when WORSEY_OFFLINE=1 and the request is not cached."""


def offline() -> bool:
    return os.environ.get("WORSEY_OFFLINE", "") == "1"


def cache_key(source: str, path: str, params: dict | None) -> str:
    blob = json.dumps({"s": source, "p": path, "q": params or {}}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def cached_call(
    source: str,
    path: str,
    params: dict | None,
    fetch: Callable[[], Any],
    root: Path | None = None,
    force_network: bool = False,
) -> Any:
    """Return the cached payload for (source, path, params), or fetch and store it.
    force_network: fetch even when WORSEY_OFFLINE=1 (live quotes the user asked for;)."""
    root = Path(root or CACHE_ROOT)
    f = root / source / f"{cache_key(source, path, params)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["payload"]
    if offline() and not force_network:
        raise OfflineCacheMiss(f"{source} {path} {params} not cached and WORSEY_OFFLINE=1")
    payload = fetch()
    f.parent.mkdir(parents=True, exist_ok=True)
    record = {"source": source, "path": path, "params": params, "fetched_at": time.time(), "payload": payload}
    f.write_text(json.dumps(record, default=str), encoding="utf-8")
    return payload
