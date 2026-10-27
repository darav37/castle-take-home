import pytest

from worsey.data import cache


def test_cache_roundtrip_and_offline(tmp_path, monkeypatch):
    calls = []

    def fetch():
        calls.append(1)
        return {"x": 1}

    assert cache.cached_call("src", "/p", {"a": 1}, fetch, root=tmp_path) == {"x": 1}
    assert cache.cached_call("src", "/p", {"a": 1}, fetch, root=tmp_path) == {"x": 1}
    assert len(calls) == 1  # second call served from disk

    monkeypatch.setenv("WORSEY_OFFLINE", "1")
    with pytest.raises(cache.OfflineCacheMiss):
        cache.cached_call("src", "/p", {"a": 2}, fetch, root=tmp_path)


def test_key_ignores_param_order():
    assert cache.cache_key("s", "/p", {"a": 1, "b": 2}) == cache.cache_key("s", "/p", {"b": 2, "a": 1})
